# pgolf-smoke

A Kupanga QA eval for OpenAI's Parameter Golf. It checks that a candidate's `train_gpt.py` trains, exports,
round-trips and scores on one H100. The agent's sandbox has no GPU, so a bug that only shows on one would otherwise
first appear in the 8×H100 benchmark, where a crash can't be sent back to the agent. This eval can send it back:
its critique carries the first traceback. The bugs it catches include a CUDA or compile error, a shape mismatch,
and an artifact that doesn't load back.

It runs the candidate as `pgolf-8xh100` will. `bench.py` here is a copy of that eval's, so the packages, the data
layout and the variant choice are the same. Only the run itself is smaller:
- `torchrun --nproc_per_node=1`: the baseline accumulates 8 micro-batches per step, so it runs the same
  optimization, more slowly.
- the full validation shard, but only 2 training shards
- `MAX_WALLCLOCK_SECONDS=180`

It sends back, as critique:

| Problem | How it's seen |
|---|---|
| A crash, or a hang | The exit code or the time limit, with the first traceback |
| Training that ignores the cap | Still training at the time limit, or `train_time` over the cap by more than 10 s |
| No score, or a score not on the round-tripped artifact | No `<label>_exact val_loss:X val_bpb:Y` line, or none after `Total submission size` |
| No artifact, or a wrong logged size | No `final_model*.ptz`; the logged total ≠ `train_gpt.py` + that file, measured on disk; over 16,000,000 bytes |
| Not the whole validation split | The scored tokens per byte (val_bpb·ln2/val_loss) match no full-split prefix (see pgolf-8xh100's README) |
| Network use | An audit hook (`audit/sitecustomize.py`, on PYTHONPATH for every rank) records Python's `socket.connect`, `getaddrinfo` and `urllib` events to other machines |
| Training data read after training | The same hook times every data file opened, and every training log line as the run writes it; a `fineweb_train_*` opened after the last one is a problem |

It also reports the short run's val_bpb, artifact, steps and times, and its one-GPU export-plus-eval time. That
time is about 8× the 8-GPU time and is only reported.

## Checked

On a 1×H100 spot `sky launch` (GCP a3-highgpu-1g, driver 580.159.04, torch 2.9.1+cu128), before the project
was created:

| Subject | Cap | Result |
|---|---|---|
| The baseline | 180 s | Passes in 301 s: 533 steps, val_bpb 1.4628, artifact 10,283,534 bytes (logged the same), scored set = the first 62,021,632 targets, 10 data opens audited, no network |
| The baseline | 60 s | Passes in 140 s |
| 16 layers instead of 9 (30.1M parameters) | 60 s | **Passes**, with an artifact of 10,299,191 bytes; see below |
| An `urllib` download of a tokenizer | 60 s | 2 problems: the crash with its traceback, and 3 network events |
| `np.fromfile` of a training shard after the training loop | 60 s | 1 problem: training data opened after training ended |

**The size check is weak.** A short run's weights compress far better than a trained model's. The baseline's
artifact is 10.3 MB after 180 s on one GPU, against 15.9 MB after 10 minutes on 8×H100, and the 16-layer copy's
30 MB of int8 payload compresses to 10.3 MB here. So this eval only catches artifacts that are over 16 MB even
untrained, and `fits_16mb` = 1 says little. `pgolf-8xh100` measures the real artifact, and a miss there sets
`meets_rules` = 0.

Settings: `SMOKE_TRAIN_S` (180), `SMOKE_TIMEOUT_S` (1800), `SMOKE_TRAIN_SHARDS` (2).
