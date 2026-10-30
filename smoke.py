"""QA for OpenAI's Parameter Golf on one H100: the candidate's train_gpt.py trains, exports, round-trips and scores.

The agent's sandbox has no GPU, so code that only fails on one (a CUDA or compile error, a shape mismatch, an export
that doesn't load back) would otherwise first show in the 8×H100 benchmark, where a crash can't be sent back to the
agent. This runs the candidate as pgolf-8xh100 will (same packages and data layout, bench.py here is a copy of its
bench.py), with `torchrun --nproc_per_node=1` (the baseline then accumulates 8 micro-batches per step), the full
validation shard and two training shards, and MAX_WALLCLOCK_SECONDS=SMOKE_TRAIN_S. It sends back, as critique:

  - a crash or a hang, with the first traceback
  - training that doesn't stop at MAX_WALLCLOCK_SECONDS
  - no `<label>_exact val_loss:X val_bpb:Y` line, or one that isn't after the `Total submission size` line
  - no final_model*.ptz file, an artifact (train_gpt.py + that file) over 16,000,000 bytes, or a logged
    `Total submission size` that differs from the artifact measured on disk
  - a scored set that isn't the whole validation split (tokens/byte from val_loss and val_bpb, see bench.py)
  - network access (an audit hook, audit/sitecustomize.py)
  - a training shard opened after the artifact's size line (training data read during evaluation), or between the
    training clock's stop and that line (calibration) when the run used its whole MAX_WALLCLOCK_SECONDS and so left
    no time for it: calibration on training data must fit in the training budget, as the challenge rules

The artifact here comes from a short training, so its size is an estimate of the full run's (the benchmark measures
that); the eval time on one GPU is about 8× the 8-GPU one and only reported.

Environment variables: SMOKE_TRAIN_S (180), SMOKE_TIMEOUT_S (5400: on one GPU, evaluation takes about 8× its
8-GPU time, so a legal 600 s evaluation can take up to ~4800 s here), SMOKE_TRAIN_SHARDS (2).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything runs the code under test


def report(result: dict) -> None:
    """Print the result, and write it to the file Kupanga reads it from (docs/evals.md)."""
    line = json.dumps(result)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


import bench  # noqa: E402  (pgolf-8xh100's bench.py, shipped with this eval: the same setup it uses)

HERE = Path(__file__).resolve().parent
SMOKE_TRAIN_S = float(os.environ.get("SMOKE_TRAIN_S", "180"))
SMOKE_TIMEOUT_S = int(os.environ.get("SMOKE_TIMEOUT_S", "5400"))
SMOKE_TRAIN_SHARDS = int(os.environ.get("SMOKE_TRAIN_SHARDS", "2"))
STOP_GRACE_S = 10.0  # one step on one GPU is ~8× an 8-GPU step
RESERVE_MIN_S = 1.0  # a run that stopped this far before its cap kept time for calibration after the loop


def stopped(out: str) -> bool:
    """Whether training stopped: the cap's `stopping_early` line or a final `step:N/N val_loss` line."""
    import re

    return "stopping_early" in out or any(a == b for a, b in re.findall(r"^step:(\d+)/(\d+) val_loss", out, re.M))


def audit_events(prefix: Path) -> list[list]:
    events = []
    for f in prefix.parent.glob(prefix.name + ".*"):
        for line in f.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    return sorted(events, key=lambda e: e[1])


def main() -> None:
    t0 = time.time()
    subject = bench.find_subject()
    variant = bench.variant_of(subject)
    print(f"subject: {subject} ({variant})\nGPU: {bench.gpu_info()}", flush=True)
    with tempfile.TemporaryDirectory(prefix="pgolf-smoke-") as tmp:
        tmp = Path(tmp)
        root = tmp / "dataroot"
        data, tokenizer = bench.prepare_data(root, variant, SMOKE_TRAIN_SHARDS)
        ref = bench.reference(data / "fineweb_val_000000.bin", tokenizer)
        code = tmp / "work" / subject.name
        import shutil

        shutil.copytree(subject, code, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__", "*.ptz", "*.pt"))
        audit = tmp / "audit" / "events"
        audit.parent.mkdir()
        env = {"DATA_PATH": str(data), "TOKENIZER_PATH": str(tokenizer), "PGOLF_DATA_ROOT": str(root), "SEED": "1337",
               "MAX_WALLCLOCK_SECONDS": f"{SMOKE_TRAIN_S:.0f}", "PGOLF_AUDIT_LOG": str(audit),
               "PYTHONPATH": os.pathsep.join(filter(None, [str(HERE / "audit"), os.environ.get("PYTHONPATH")]))}
        r = bench.run_once(code, env, "smoke", SMOKE_TIMEOUT_S, nproc=1)
        events = audit_events(audit)
    print("run output, last 30 lines:\n" + "\n".join(r["output"].splitlines()[-30:]), flush=True)
    problems: list[str] = []
    tail = bench.failure_excerpt(r["output"])
    if r["rc"] != 0 or r.get("val_bpb") is None:
        if r["rc"] == -9:
            why = (f"was still training after {SMOKE_TIMEOUT_S} s with MAX_WALLCLOCK_SECONDS={SMOKE_TRAIN_S:.0f}: it "
                   "must stop training at the cap" if not stopped(r["output"]) else
                   f"didn't finish within {SMOKE_TIMEOUT_S} s (training stopped, so export or evaluation hung or is far "
                   "too slow; on 8×H100 evaluation must take at most 600 s)")
        elif r["rc"]:
            why = f"exited with code {r['rc']}"
        else:
            why = "finished without a `<label>_exact val_loss:X val_bpb:Y` line with a finite val_bpb"
        msg = f"On one H100 (torchrun --nproc_per_node=1, MAX_WALLCLOCK_SECONDS={SMOKE_TRAIN_S:.0f}), train_gpt.py {why}. {tail}"
        problems.append(msg if len(msg) <= 8000 else msg[:3000] + "\n...\n" + msg[-5000:])
    else:
        for m in bench.rule_misses(r, ref, train_limit_s=SMOKE_TRAIN_S, eval_limit_s=None, serialize_limit=False):
            if m.startswith("training ran") and r["train_time_s"] <= SMOKE_TRAIN_S + STOP_GRACE_S:
                continue
            problems.append(f"On one H100 with MAX_WALLCLOCK_SECONDS={SMOKE_TRAIN_S:.0f}: {m}." + (
                " (From a short training run; trained weights usually compress a little worse, so leave a margin.)"
                if m.startswith("the artifact is") else ""))
    net = [e for e in events if e[0] in ("connect", "resolve", "urllib")]
    if net:
        problems.append("train_gpt.py used the network: " + "; ".join(f"{e[0]} {e[2]}" for e in net[:5])
                        + ". No network calls or downloads are allowed during training or evaluation.")
    marks = [e[1] for e in events if e[0] == "train_line"]
    t_end = max(marks) if marks else r.get("t_train_end")
    sizes = [e[1] for e in events if e[0] == "size_line"]
    t_size = max(sizes) if sizes else None  # the last `Total submission size` line: the artifact is final
    late = [e for e in events if e[0] == "open" and "fineweb_train_" in e[2] and t_end and e[1] > t_end]
    in_eval = [e for e in late if t_size is not None and e[1] > t_size]
    in_export = [e for e in late if e not in in_eval]
    train_s = r.get("train_time_s")
    reserved = SMOKE_TRAIN_S - train_s if train_s is not None else 0.0
    if in_eval:
        problems.append(f"train_gpt.py opened training data during evaluation ({Path(in_eval[0][2]).name}, "
                        f"{in_eval[0][1] - t_size:.0f} s after the `Total submission size` line). Evaluation may not "
                        "read training data.")
    if in_export and reserved < RESERVE_MIN_S:
        problems.append(f"train_gpt.py opened training data after the training clock stopped "
                        f"({Path(in_export[0][2]).name}, {in_export[0][1] - t_end:.0f} s after the last training "
                        f"step), while its training used the whole MAX_WALLCLOCK_SECONDS={SMOKE_TRAIN_S:.0f} "
                        f"(train_time {train_s} s). Anything that uses training data to change the model, such as "
                        "quantization calibration, belongs in the 600 s training budget: stop the training loop "
                        "early enough to leave time for it within MAX_WALLCLOCK_SECONDS, or calibrate on sequences "
                        "the model generates itself.")
    elif in_export:
        print(f"training data opened {in_export[0][1] - t_end:.0f} s after the training clock stopped, inside the "
              f"{reserved:.0f} s the run reserved of its cap (calibration)", flush=True)
    opens = sum(1 for e in events if e[0] == "open")
    a = r.get("artifact_bytes")
    print(f"rc {r['rc']}, val_bpb {r.get('val_bpb')} ({r.get('score_label')}), prequant {r.get('prequant_val_bpb')}, "
          f"steps {r.get('steps')}, train {r.get('train_time_s')}s, eval (1 GPU) {r.get('eval_time_s')}, artifact {a} "
          f"({r.get('artifact_file')}, logged {r.get('printed_bytes')}), audit: {opens} data opens, {len(net)} network; "
          f"wall {time.time() - t0:.0f}s", flush=True)
    if r.get("val_bpb") is not None:
        print(bench.scored_set_ok(r["val_loss"], r["val_bpb"], ref)[1], flush=True)
    report({"problems": len(problems), "critique": problems, "val_bpb": r.get("val_bpb") or -1.0,
            "prequant_val_bpb": r.get("prequant_val_bpb") or -1.0, "artifact_bytes": a or -1,
            "code_bytes": r["code_bytes"], "model_bytes": r.get("model_bytes") or -1,
            "fits_16mb": int(a is not None and r.get("model_bytes") is not None and a <= bench.LIMIT_BYTES),
            "steps": r.get("steps", 0), "train_time_s": r.get("train_time_s", -1.0),
            "eval_time_1gpu_s": round(r.get("eval_time_s", -1.0), 1), "data_opens_audited": opens,
            "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
