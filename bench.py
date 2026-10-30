"""Kupanga benchmark: OpenAI's Parameter Golf, main track (10 minutes on 8×H100, 16,000,000-byte artifact).

The subject is the challenge's train_gpt.py (the naive baseline, or a candidate) in subjects/<name>/. On one 8×H100
SXM machine the eval:

  1. Picks the tokenizer variant from train_gpt.py's default DATA_PATH (fineweb10B_sp1024, or fineweb10B_sp4096) and
     downloads that variant's published data once, pinned to one Hugging Face revision: the validation shard, the
     training shards (sp1024: 80, the default of data/cached_challenge_fineweb.py; sp4096: all 143) and the tokenizer,
     into a read-only folder laid out as the downloader leaves it (datasets/fineweb10B_spN/, tokenizers/).
  2. Rebuilds the parent's code by reverse-applying $HARNESS_CHANGES_FILE (what the candidate changed since its
     experiment started) to a copy of the subject. The base node has no parent.
  3. Runs `torchrun --standalone --nproc_per_node=8 train_gpt.py` with DATA_PATH, TOKENIZER_PATH, PGOLF_DATA_ROOT,
     RUN_ID, SEED and MAX_WALLCLOCK_SECONDS=600 for each seed in KX_SEEDS (1337, 42, 2024), and the parent once
     (seed 1337), in the order c p c c so drift on the machine reaches both.
  4. From each run's output, timed line by line here:
       - val_bpb: the last `<label>_exact val_loss:X val_bpb:Y` line
       - train_time: the run's own last `train_time:Nms` (compile and warmup excluded, as on the leaderboard)
       - pre_serialize_s: train_time plus the wall time from the last training line to the first serialization line
         (`Serialized model` / `Total submission size`), less any evaluation logged in between (`eval_time:Nms`), so
         calibration on data after the clock and before that line counts as training
       - eval_time: the wall time from the last training line to the run's exit
       - the artifact: the bytes of train_gpt.py plus the size of the final_model*.ptz file, measured on disk, and
         compared with the last `Total submission size ...: N bytes` line
       - the scored set: val_bpb·ln2/val_loss is the run's scored tokens per byte, which must equal that of a prefix of
         the validation split covering all of it but at most one sequence (computed here from the shard and the
         tokenizer), so skipped tokens or wrong byte counts show
  5. Reports the mean val_bpb (primary, lower is better) with its standard error, the artifact, the times, and
     meets_rules: 1 only if every candidate run meets every rule above (artifact ≤ 16,000,000 bytes and matching the
     logged size; train_time and pre_serialize_s ≤ 600 s; eval_time ≤ 600 s; the score printed after the artifact's
     size line; the scored set; a clean exit). A rule miss is reported with meets_rules = 0 and the reason in rule_misses,
     not raised; only a run with no score line (a crash before it, or a timeout) raises.

The static rules (no network, one file, the metric code) are checked by pgolf-rules before this runs, and a short
one-GPU run by pgolf-smoke.

Environment variables: KX_SEEDS (1337,42,2024), KX_BASELINE_RUNS (1), KX_TRAIN_SHARDS (the variant's default),
KX_RUN_TIMEOUT_S (2100), KX_NPROC (8).

Local checks, no GPU: python bench.py --parse <stdout.txt> ... ; python bench.py --reference <val.bin> <tok.model>
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything runs the code under test


def report(result: dict) -> None:
    """Print the result, and write it to the file Kupanga reads it from (docs/evals.md)."""
    line = json.dumps(result)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


# The challenge's published exports, each pinned to one revision. sp1024 is data/cached_challenge_fineweb.py's
# default repo (the quasi10B export every record since 2026-03-18 used); sp4096 is the export the 2026-04-01 record
# used (MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf). Validation is the same first 50,000 documents in both.
VARIANTS = {
    "sp1024": {"url": "https://huggingface.co/datasets/willdepueoai/parameter-golf/resolve/"
                      "a85b0e6035c3c94bc23685a07c81a8f3bf89db80/datasets",
               "tokenizer": "fineweb_1024_bpe", "train_shards": 80, "max_train_shards": 195},
    "sp4096": {"url": "https://huggingface.co/datasets/kevclark/parameter-golf/resolve/"
                      "c2ba4d294df42144797c3f03ed5736c8dc8f4db7/datasets",
               "tokenizer": "fineweb_4096_bpe", "train_shards": 143, "max_train_shards": 143},
}
LIMIT_BYTES = 16_000_000  # decimal 16 MB: code bytes + compressed model bytes
TRAIN_LIMIT_S = 600.0
TRAIN_GRACE_S = 1.0  # the cap is checked after each step, so a run stops one step past it (records: up to 0.14 s)
SERIALIZE_GRACE_S = 5.0  # saving the raw state dict, EMA/SWA swaps and logging between the stop and serialization
EVAL_LIMIT_S = 600.0
MAX_UNSCORED_TAIL = 16_384  # a script may drop the validation tail that doesn't fill a sequence (≤ its seq_len)
RATIO_RTOL = 3e-8  # the _exact line's 8 decimals of val_loss and val_bpb give tokens/byte to about 7e-9
SEEDS = [int(s) for s in os.environ.get("KX_SEEDS", "1337,42,2024").split(",") if s.strip()]
BASELINE_RUNS = int(os.environ.get("KX_BASELINE_RUNS", "1"))
TRAIN_SHARDS = os.environ.get("KX_TRAIN_SHARDS")
RUN_TIMEOUT_S = int(os.environ.get("KX_RUN_TIMEOUT_S", "2100"))
NPROC = int(os.environ.get("KX_NPROC", "8"))
EVAL_TIMEOUT_S = int(os.environ.get("HARNESS_EVAL_TIMEOUT_S", "9000"))
MIN_RUNS = 2  # fewer candidate runs than this (out of time) aren't scored

SCORE = re.compile(r"^(\S+)_exact val_loss:(\S+) val_bpb:(\S+)(?:\s.*)?$")  # extra fields after val_bpb are allowed
PREQUANT = re.compile(r"^step:(\d+)/(\d+) val_loss:(\S+) val_bpb:(\S+) train_time:\s*(\d+)ms")
TRAIN_LINE = re.compile(r"^(?:step:(\d+)/\d+ .*train_time:\s*(\d+)ms|stopping_early:.*train_time:\s*(\d+)ms)")
EVAL_TIME = re.compile(r"\beval_time:\s*(\d+(?:\.\d+)?)ms")
SERIALIZE = re.compile(r"^(?:Serialized model|Total submission size)")
SIZE = re.compile(r"^Total submission size[^:]*:\s*(\d+) bytes")
TRACEBACK = re.compile(r"^(\[rank\d+\]:\s?)?Traceback \(most recent call last\):", re.M)
DATA_DEFAULT = re.compile(r"""["']DATA_PATH["']\s*,\s*[^)\n]*fineweb10B_(sp\d+)""")


def find_subject() -> Path:
    root = Path(os.environ.get("HARNESS_SUBJECTS_DIR") or Path(__file__).resolve().parent.parent.parent / "subjects")
    found = sorted(p.parent for p in root.glob("*/train_gpt.py"))
    if not found:
        raise SystemExit(f"no Parameter Golf subject (train_gpt.py) under {root}/<subject>/")
    return found[0]


def variant_of(code: Path) -> str:
    """The tokenizer variant a train_gpt.py is written for: the fineweb10B_spN in its DATA_PATH default (sp1024 when
    there's none). pgolf-rules checks the tokenizer and VOCAB_SIZE defaults agree with it."""
    m = DATA_DEFAULT.search((code / "train_gpt.py").read_text(errors="replace"))
    v = m.group(1) if m else "sp1024"
    if v not in VARIANTS:
        raise SystemExit(f"train_gpt.py defaults DATA_PATH to fineweb10B_{v}; this project provides only "
                         f"{', '.join('fineweb10B_' + k for k in VARIANTS)}")
    return v


def fetch(url: str, dest: Path) -> Path:
    """Download url into dest, retrying a failed or truncated download (a cut connection raises
    http.client.IncompleteRead, which isn't an OSError; a shard is also checked against its header)."""
    path = dest / url.rsplit("/", 1)[1]
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(path, "wb") as f:  # noqa: S310
                while chunk := r.read(1 << 24):
                    f.write(chunk)
            if path.suffix == ".bin":
                check_shard(path)
            return path
        except (OSError, http.client.HTTPException, ValueError) as e:
            print(f"download {url} failed ({e!r}), attempt {attempt + 1}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not download {url}")


def check_shard(path: Path) -> int:
    """The shard header train_gpt.py checks (magic 20240520, version 1, a token count matching the size)."""
    import struct

    with open(path, "rb") as f:
        head = f.read(12)
    magic, version, n = struct.unpack("<3i", head) if len(head) == 12 else (0, 0, 0)
    if magic != 20240520 or version != 1 or path.stat().st_size != 1024 + 2 * n:
        raise ValueError(f"downloaded shard {path.name} is corrupt (header {magic}, {version}, {n}, "
                         f"{path.stat().st_size} bytes)")
    return n


def prepare_data(root: Path, variant: str, train_shards: int | None = None) -> tuple[Path, Path]:
    """root/datasets/fineweb10B_<variant>/ and root/tokenizers/, as `python3 data/cached_challenge_fineweb.py --variant
    <variant> --train-shards N` leaves them; read-only. Returns (DATA_PATH, TOKENIZER_PATH)."""
    v = VARIANTS[variant]
    n_train = v["train_shards"] if train_shards is None else min(train_shards, v["max_train_shards"])
    data, tok = root / "datasets" / f"fineweb10B_{variant}", root / "tokenizers"
    data.mkdir(parents=True)
    tok.mkdir(parents=True, exist_ok=True)
    tok.chmod(0o755)
    jobs = [(f"{v['url']}/tokenizers/{v['tokenizer']}.{ext}", tok) for ext in ("model", "vocab")]
    jobs += [(f"{v['url']}/datasets/fineweb10B_{variant}/fineweb_val_000000.bin", data)]
    jobs += [(f"{v['url']}/datasets/fineweb10B_{variant}/fineweb_train_{i:06d}.bin", data) for i in range(n_train)]
    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        paths = list(pool.map(lambda j: fetch(*j), jobs))
    for p in paths:
        p.chmod(0o444)
    data.chmod(0o555)
    tok.chmod(0o555)
    size = sum(p.stat().st_size for p in paths) / 1e9
    print(f"{variant}: downloaded {len(paths)} files ({size:.1f} GB: the val shard, {n_train} train shards, the "
          f"tokenizer) in {time.time() - t0:.0f}s", flush=True)
    return data, tok / f"{v['tokenizer']}.model"


def reference(val_file: Path, tokenizer: Path) -> dict:
    """Tokens per byte of every prefix of the validation targets that leaves at most MAX_UNSCORED_TAIL unscored, with
    the baseline's byte counts (build_sentencepiece_luts): a scored set's val_bpb·ln2/val_loss must be one of these."""
    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    n = int(sp.vocab_size())
    base = np.zeros(n, dtype=np.int64)
    space = np.zeros(n, dtype=bool)
    boundary = np.ones(n, dtype=bool)
    for t in range(n):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t):
            continue
        boundary[t] = False
        if sp.is_byte(t):
            base[t] = 1
            continue
        piece = sp.id_to_piece(t)
        if piece.startswith("▁"):
            space[t] = True
            piece = piece[1:]
        base[t] = len(piece.encode("utf-8"))
    tokens = np.fromfile(val_file, dtype="<u2", offset=1024).astype(np.int64)
    prev, tgt = tokens[:-1], tokens[1:]
    nbytes = base[tgt] + (space[tgt] & ~boundary[prev])
    cum = np.cumsum(nbytes)
    k0 = max(1, len(tgt) - MAX_UNSCORED_TAIL)
    ks = np.arange(k0, len(tgt) + 1)
    ratios = ks / cum[ks - 1]
    return {"targets": len(tgt), "bytes": int(cum[-1]), "k0": k0, "ratios": ratios,
            "lo": float(ratios.min()), "hi": float(ratios.max())}


def scored_set_ok(val_loss: float, val_bpb: float, ref: dict) -> tuple[bool, str]:
    """Whether the run's tokens/byte matches a full-split prefix, and a description of what it matched."""
    import numpy as np

    if not val_loss or val_loss <= 0:
        return False, f"val_loss {val_loss} can't be checked"
    r = val_bpb * math.log(2) / val_loss
    err = np.abs(ref["ratios"] - r) / r
    i = int(err.argmin())
    if err[i] <= RATIO_RTOL:
        return True, f"tokens/byte {r:.8f} = the first {ref['k0'] + i:,} of {ref['targets']:,} targets"
    return False, (f"the scored tokens/byte, val_bpb·ln2/val_loss = {r:.8f}, matches no prefix of the validation split "
                   f"that leaves at most {MAX_UNSCORED_TAIL:,} of its {ref['targets']:,} targets unscored "
                   f"({ref['lo']:.8f}–{ref['hi']:.8f}): some tokens were skipped or scored twice (for example the "
                   "last windows of a sliding-window evaluation), or tokens or bytes were miscounted (count them "
                   "exactly, in int64 or float64)")


def parent_code(subject: Path, work: Path) -> Path | None:
    """A copy of the subject with the experiment's changes reverse-applied (the parent's code), or None on the
    base node."""
    changes = os.environ.get("HARNESS_CHANGES_FILE")
    if not changes or not Path(changes).is_file() or not Path(changes).read_text(errors="replace").strip():
        return None
    diff = Path(changes).read_text(errors="replace")
    if "[harness: the diff is cut here" in diff:
        print("the candidate's changes are too large for Kupanga's diff; no same-machine parent run", flush=True)
        return None
    dest = work / "subjects" / subject.name
    shutil.copytree(subject, dest, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__", "*.ptz", "*.pt"))
    if shutil.which("git") is None:
        subprocess.run("apt-get update -qq && apt-get install -y -qq --no-install-recommends git", shell=True,
                       check=True, capture_output=True, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    p = subprocess.run(["git", "apply", "-R", "--whitespace=nowarn", changes], cwd=work, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"could not rebuild the parent from the candidate's changes, so no parent run: {p.stderr.strip()[:500]}",
              flush=True)
        return None
    return dest


def failure_excerpt(out: str) -> str:
    """The first Python traceback in a torchrun's output (any rank), then its last lines."""
    lines = out.splitlines()
    tail = "\n".join(lines[-25:])
    m = TRACEBACK.search(out)
    if m is None:
        return "Last output:\n" + tail
    start = out[:m.start()].count("\n")
    prefix = m.group(1) or ""
    block = [lines[start]]
    for line in lines[start + 1:]:
        if prefix and not line.startswith(prefix.rstrip()):
            break
        block.append(line)
        if not prefix and line and not line[0].isspace():
            break
    if len(block) > 60:
        block = block[:25] + [f"... ({len(block) - 50} lines of frames) ..."] + block[-25:]
    return "First traceback:\n" + "\n".join(block) + "\n...\nLast output:\n" + tail


def finite(x: str) -> float | None:
    try:
        v = float(x)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse(lines: list[tuple[float, str]], t_exit: float | None = None) -> dict:
    """Numbers from a run's (time, line) output. Keys are missing when the run never printed them."""
    out: dict = {}
    t_train_end = t_serialize = None
    eval_ms = 0.0  # evaluation the run logs (`eval_time:Nms`) between the last training line and serialization
    score_at = size_at = -1
    for i, (t, line) in enumerate(lines):
        if m := SCORE.match(line):
            out |= {"score_label": m.group(1), "val_loss": finite(m.group(2)), "val_bpb": finite(m.group(3))}
            score_at = i
        if m := PREQUANT.match(line):
            out |= {"prequant_val_bpb": finite(m.group(4)), "steps": int(m.group(1))}
        if m := TRAIN_LINE.match(line):
            out["train_time_s"] = max(out.get("train_time_s", 0.0), int(m.group(2) or m.group(3)) / 1000)
            if m.group(1):
                out["steps"] = max(out.get("steps", 0), int(m.group(1)))
            t_train_end, t_serialize, eval_ms = t, None, 0.0
        elif t_train_end is not None and t_serialize is None and (m := EVAL_TIME.search(line)):
            eval_ms += float(m.group(1))
        if SERIALIZE.match(line) and t_train_end is not None and t_serialize is None:
            t_serialize = t
        if m := SIZE.match(line):
            out["printed_bytes"] = int(m.group(1))
            size_at = i
    out["score_after_size"] = score_at > size_at >= 0
    out["t_train_end"] = t_train_end
    if t_exit is not None and t_train_end is not None:
        out["eval_time_s"] = t_exit - t_train_end
        if t_serialize is not None and "train_time_s" in out:
            # a diagnostic evaluation of the trained weights before export isn't training
            out["pre_serialize_s"] = out["train_time_s"] + max(0.0, t_serialize - t_train_end - eval_ms / 1000)
    return out


def run_once(code: Path, env_extra: dict, label: str, timeout: float, nproc: int = NPROC) -> dict:
    """One torchrun of train_gpt.py in `code`. Returns the parsed numbers plus rc, wall_s, the artifact sizes and the
    output text; never raises for the run's own failure (the caller decides)."""
    for f in [*code.glob("final_model*"), *code.glob("logs")]:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    env = {k: v for k, v in os.environ.items() if not k.startswith("HARNESS_")}
    env |= {"PYTHONUNBUFFERED": "1", "NCCL_IB_DISABLE": "1", "RUN_ID": label, **env_extra}
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train_gpt.py"]
    out_path = code.parent / f"{label}.out"
    lines: list[tuple[float, str]] = []
    t0 = time.time()
    with open(out_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace", bufsize=1, start_new_session=True)

        def pump() -> None:
            for line in p.stdout:
                lines.append((time.time(), line.rstrip("\n")))
                f.write(line)
                f.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = -9
        t_exit = time.time()
        reader.join(timeout=30)
    res = parse(lines, t_exit) | {"rc": rc, "wall_s": t_exit - t0, "t_exit": t_exit,
                                  "output": out_path.read_text(errors="replace")}
    code_bytes = len((code / "train_gpt.py").read_bytes())
    ptz = sorted(code.glob("final_model*.ptz"), key=lambda q: q.stat().st_size)
    res["code_bytes"] = code_bytes
    res["model_bytes"] = ptz[-1].stat().st_size if ptz else None
    res["artifact_file"] = ptz[-1].name if ptz else None
    res["artifact_bytes"] = code_bytes + ptz[-1].stat().st_size if ptz else res.get("printed_bytes")
    for f in code.glob("final_model*"):  # the raw state dict is ~67 MB a run
        f.unlink()
    return res


def rule_misses(r: dict, ref: dict | None, train_limit_s: float = TRAIN_LIMIT_S,
                eval_limit_s: float | None = EVAL_LIMIT_S, serialize_limit: bool = True) -> list[str]:
    """Every runtime rule the run broke, in words."""
    misses = []
    if r.get("rc") not in (0, None):
        misses.append(f"train_gpt.py exited with code {r['rc']} after printing its score" if r["rc"] != -9 else
                      "train_gpt.py printed its score but was still running at the run's time limit, and was killed")
    a, printed = r.get("artifact_bytes"), r.get("printed_bytes")
    if r.get("model_bytes") is None:
        misses.append("no final_model*.ptz artifact file was written, so the artifact can't be measured")
    elif a > LIMIT_BYTES:
        misses.append(f"the artifact is {a:,} bytes > {LIMIT_BYTES:,} (train_gpt.py {r['code_bytes']:,} + "
                      f"{r['artifact_file']} {r['model_bytes']:,})")
    if printed is None:
        misses.append("no `Total submission size ...: N bytes` line")
    elif r.get("model_bytes") is not None and printed != a:
        misses.append(f"the logged total submission size {printed:,} differs from the measured artifact {a:,} "
                      f"(train_gpt.py {r['code_bytes']:,} + {r['artifact_file']} {r['model_bytes']:,})")
    t = r.get("train_time_s")
    if t is None:
        misses.append("no train_time was printed")
    elif t > train_limit_s + TRAIN_GRACE_S:
        misses.append(f"training ran {t:.1f} s > {train_limit_s:.0f} s")
    ps = r.get("pre_serialize_s")
    if serialize_limit and ps is not None and ps > train_limit_s + SERIALIZE_GRACE_S:
        misses.append(f"{ps:.1f} s passed from the start of training to serialization (train_time {t:.1f} s plus "
                      f"{ps - t:.1f} s after the last training step, logged evaluation excluded) > {train_limit_s:.0f} "
                      "s: anything that uses data to change the model, such as quantization calibration, must fit in "
                      "the training budget")
    e = r.get("eval_time_s")
    if eval_limit_s is not None and e is not None and e > eval_limit_s:
        misses.append(f"evaluation took {e:.0f} s > {eval_limit_s:.0f} s")
    if not r.get("score_after_size"):
        misses.append("the final `_exact` val_bpb line doesn't come after the `Total submission size` line, so it "
                      "isn't the score of the round-tripped artifact")
    if ref is not None and r.get("val_bpb") is not None:
        ok, what = scored_set_ok(r.get("val_loss"), r["val_bpb"], ref)
        if not ok:
            misses.append(what)
    return misses


def mean_se(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def summarize(cand: list[dict], base: list[dict], ref: dict | None) -> dict:
    bpb, bpb_se = mean_se([r["val_bpb"] for r in cand])
    loss, _ = mean_se([r["val_loss"] for r in cand])
    pre = [r["prequant_val_bpb"] for r in cand if r.get("prequant_val_bpb") is not None]
    misses = [f"seed {r['seed']}: {m}" for r in cand for m in rule_misses(r, ref)]
    out = {
        "val_bpb": bpb, "val_bpb_se": bpb_se, "val_loss": loss,
        "val_bpb_runs": ", ".join(f"{r['seed']}: {r['val_bpb']:.4f}" for r in cand),
        "prequant_val_bpb": sum(pre) / len(pre) if pre else -1.0,
        "artifact_bytes": max(r.get("artifact_bytes") or -1 for r in cand),
        "code_bytes": cand[-1]["code_bytes"],
        "model_bytes": max(r.get("model_bytes") or -1 for r in cand),
        "train_time_s": max(r.get("train_time_s") or -1.0 for r in cand),
        "pre_serialize_s": max(r.get("pre_serialize_s") or -1.0 for r in cand),
        "eval_time_s": max(r.get("eval_time_s") or -1.0 for r in cand),
        "steps": round(sum(r.get("steps", 0) for r in cand) / len(cand)),
        "runs": len(cand), "meets_rules": int(not misses), "rule_misses": "; ".join(misses) or "none",
        "baseline_runs": len(base),
    }
    if base:
        bb, _ = mean_se([r["val_bpb"] for r in base])
        out |= {"baseline_val_bpb": bb, "val_bpb_delta": bpb - bb,
                "baseline_artifact_bytes": max(r.get("artifact_bytes") or -1 for r in base)}
    return out


def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main() -> None:
    t0 = time.time()
    deadline = t0 + EVAL_TIMEOUT_S - 300  # leave time to report
    subject = find_subject()
    gpus = gpu_info()
    print(f"subject: {subject}\nGPUs:\n{gpus}", flush=True)
    names = gpus.splitlines()
    if len(names) < NPROC or not all("H100" in n for n in names):
        raise SystemExit(f"needs {NPROC}×H100, found: {gpus or 'no GPUs'}")
    with tempfile.TemporaryDirectory(prefix="pgolf-") as tmp:
        tmp = Path(tmp)
        work = tmp / "work"
        cand_code = work / "candidate" / subject.name
        shutil.copytree(subject, cand_code, symlinks=True,
                        ignore=shutil.ignore_patterns("logs", "__pycache__", "*.ptz", "*.pt"))
        base_code = parent_code(subject, work / "parent")
        codes = {"candidate": cand_code, "parent": base_code}
        variants = {"candidate": variant_of(cand_code)}
        if base_code is not None:
            try:
                variants["parent"] = variant_of(base_code)
            except SystemExit as e:
                print(f"the parent's code can't run here, so no parent run: {e}", flush=True)
                base_code = codes["parent"] = None
        root = tmp / "dataroot"
        paths, refs = {}, {}
        for v in dict.fromkeys(variants.values()):
            paths[v] = prepare_data(root, v, int(TRAIN_SHARDS) if TRAIN_SHARDS else None)
            refs[v] = reference(paths[v][0] / "fineweb_val_000000.bin", paths[v][1])
            print(f"{v}: {refs[v]['targets']:,} validation targets, {refs[v]['bytes']:,} bytes", flush=True)
        order = [("candidate", s) for s in SEEDS]
        if base_code is not None:
            for i in range(min(BASELINE_RUNS, len(SEEDS))):
                order.insert(1 + 2 * i, ("parent", SEEDS[i]))
        cand: list[dict] = []
        base: list[dict] = []
        longest = 0.0
        for who, seed in order:
            if time.time() + longest * 1.1 > deadline:
                print(f"stopping early: not enough time for another run ({len(cand)} candidate, {len(base)} parent)",
                      flush=True)
                break
            label = f"{who}-seed{seed}"
            data, tokenizer = paths[variants[who]]
            env = {"DATA_PATH": str(data), "TOKENIZER_PATH": str(tokenizer), "PGOLF_DATA_ROOT": str(root),
                   "SEED": str(seed), "MAX_WALLCLOCK_SECONDS": f"{TRAIN_LIMIT_S:.0f}"}
            r = run_once(codes[who], env, label, min(RUN_TIMEOUT_S, max(60.0, deadline - time.time())))
            r["seed"] = seed
            longest = max(longest, r["wall_s"])
            if r.get("val_bpb") is None or r.get("val_loss") is None:
                why = ("timed out" if r["rc"] == -9 else f"exit {r['rc']}" if r["rc"] else
                       "no `<label>_exact val_loss:X val_bpb:Y` line with a finite val_loss and val_bpb")
                msg = f"{label}: train_gpt.py failed ({why}) after {r['wall_s']:.0f}s; "
                if who == "parent":
                    print(msg + "the parent's code fails here, so there is no same-machine delta. "
                          + failure_excerpt(r["output"])[-3000:], flush=True)
                    continue
                raise SystemExit(msg + "no score. " + failure_excerpt(r["output"]))
            misses = rule_misses(r, refs[variants[who]])
            print(f"{label} ({variants[who]}): val_bpb {r['val_bpb']:.4f} ({r['score_label']}), prequant "
                  f"{r.get('prequant_val_bpb')}, steps {r.get('steps')}, train {r.get('train_time_s', -1):.1f}s, "
                  f"to serialization {r.get('pre_serialize_s', -1):.1f}s, eval {r.get('eval_time_s', -1):.0f}s, "
                  f"artifact {r.get('artifact_bytes')} bytes ({r.get('artifact_file')}; logged {r.get('printed_bytes')}), "
                  f"{scored_set_ok(r['val_loss'], r['val_bpb'], refs[variants[who]])[1]}, wall {r['wall_s']:.0f}s"
                  + (f"; RULE MISS: {'; '.join(misses)}" if misses else ""), flush=True)
            (cand if who == "candidate" else base).append(r)
    if len(cand) < MIN_RUNS:
        raise SystemExit(f"only {len(cand)} candidate runs finished in the time limit (need {MIN_RUNS}); no score")
    res = summarize(cand, base, refs[variants["candidate"]])
    res |= {"variant": variants["candidate"], "gpus": len(names), "gpu": names[0].split(",")[0].strip(),
            "driver": names[0].split(",")[-1].strip(), "wall_s": time.time() - t0}
    print(f"candidate: val_bpb {res['val_bpb']:.4f} ± {res['val_bpb_se']:.4f} over {res['runs']} seeds; artifact "
          f"{res['artifact_bytes']:,} bytes; train {res['train_time_s']:.1f}s; eval {res['eval_time_s']:.0f}s; "
          f"meets_rules {res['meets_rules']} ({res['rule_misses']})", flush=True)
    if base:
        print(f"parent on this machine: val_bpb {res['baseline_val_bpb']:.4f}; delta {res['val_bpb_delta']:+.4f}",
              flush=True)
    report(res)


def check_parse(paths: list[str], ref: dict | None = None) -> None:
    """Parse saved outputs as if each were a candidate run (a local check of parse and summarize)."""
    runs = []
    for i, p in enumerate(paths):
        text = Path(p).read_text(errors="replace")
        r = parse([(0.0, line) for line in text.splitlines()]) | {"seed": i, "code_bytes": 0}
        r["artifact_bytes"] = r.get("printed_bytes")
        if r.get("val_bpb") is None:
            raise SystemExit(f"{p}: no `_exact val_loss:X val_bpb:Y` line")
        runs.append(r)
        print(f"{Path(p).name}: {r}")
    print(json.dumps(summarize(runs, runs[:1], ref), indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--parse"]:
        check_parse(sys.argv[2:])
    elif sys.argv[1:2] == ["--reference"]:
        ref = reference(Path(sys.argv[2]), Path(sys.argv[3]))
        print(f"{ref['targets']:,} targets, {ref['bytes']:,} bytes, prefix tokens/byte {ref['lo']:.8f}–{ref['hi']:.8f}")
        for pair in sys.argv[4:]:  # val_loss,val_bpb pairs from saved logs
            loss, bpb = map(float, pair.split(","))
            print(pair, scored_set_ok(loss, bpb, ref))
    else:
        main()
