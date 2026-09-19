# Reproducing the results

End to end: generate data → split → train → export → verify → benchmark.

You do **not** need to run all of it. The trained model ships in `models/`, so
if you only want to run the agents, [build and go](README.md#build). The
pipeline below is for reproducing the model itself.

---

## Which numbers come from where

This matters more than anything else in this document, so it comes first.

**The paper's headline result is the official tournament**: `DeepSetsBlendBot`
first of eight, `DeepSetsBot` second, at 69.21% and 68.24% over 5460 games each.
Those are win rates **across the whole tournament field**, not against any one
opponent, and the games were run by the competition organisers on their
hardware. Nothing in this repository reproduces them; the harnesses here
reproduce our own local measurements.

**Our local benchmarking ran optimistic in every head-to-head we can compare.**
Matching local measurements against the corresponding tournament head-to-head
results:

| Matchup | Local estimate | Tournament | Gap |
|---|---|---|---|
| `DeepSetsBot` vs SakkirinaSolo | 76.3% | 70.6% | **+5.7** |
| `DeepSetsBot` vs BestMCTS3 | 88.0% | 69.5% | **+18.5** |

Local figures are 400 games per matchup at 10s/turn with seats swapped. The
overestimate is mild against SakkirinaSolo and **severe against BestMCTS3** —
nearly 19 points. Treat any local number in this repository as an upper bound,
and do not assume the bias is uniform across opponents: it plainly is not.

Do **not** compare a local head-to-head figure against 69.21% or 68.24%. Those
are field-wide averages and are not the right denominator for a single matchup.

So, when citing:

- Tournament placings and win rates → the official results, not this repo.
- Anything produced by `tools/benchmark*.py` → a **local estimate**, labelled
  as such, expected to be optimistic by an opponent-dependent margin.
- Ablations and controls comparing our own variants against each other →
  local, but like-for-like, so the bias largely cancels.

## Prerequisites

- .NET 8 SDK
- Python 3.10+ with `torch`, `torch_geometric`, `onnxruntime`, `scikit-learn`,
  `numpy`, `matplotlib`
- `git` and `patch` (for `scripts/fetch_baselines.sh`)

```bash
dotnet build TalesOfTribute.sln -c Release
./scripts/fetch_baselines.sh                 # needed for anything involving baselines
dotnet build TalesOfTribute.sln -c Release   # rebuild, now with the baselines
```

## 0. Regenerate the card database

`training/card_db.py` maps each card to a 76-dimensional effect vector and
supplies 76 of the 99 node-feature dimensions. It is generated from
`Engine/cards.json` and is the **sole** source of truth for card effects (the
`Effects` field in logged game states is always empty, so there is no
independent cross-check).

```bash
cd training
python generate_db.py        # Engine/cards.json -> card_db.py
python verify_card_db.py     # audits the parse; read its output, do not skip
python generate_cs_db.py     # card_db.py -> the C# table
```

`verify_card_db.py` reports every place the parse could silently drop an
effect. It is read-only.

Runtime: seconds.

## 1. Generate training data

Self-play through `GameRunner --log-training-data`, sharded across processes.

Locally:

```bash
tools/generate_data.sh --games 500 --jobs 8 --out-dir /path/to/data \
                       --bot SakkirinaGenNeural --seed-base 20260807
```

On a cluster, edit the `CHANGE_ME_*` placeholders at the top of
`scripts/slurm_generate.sh` and submit it. It runs 32 independent single-core
array tasks, each writing to its own `task_NN/` directory with a
non-overlapping seed range. It is resumable: re-running skips completed tasks.

### The two generation runs, and their seeds

The shipped model was trained on a **combined** dataset from two runs:

| Run | Generating agent | `--seed-base` | Notes |
|---|---|---|---|
| Heuristic | `SakkirinaGen` | `20260803` | Derived from SakkirinaSolo by `scripts/sakkirina_gen.patch`. |
| Neural self-play | `SakkirinaGenNeural` | `20260807` | ~12,000 games. |

`tools/generate_data.sh` adds `task_id * 1000000` to the seed base per task, so
the two runs' seed ranges cannot collide. The benchmark run used seed base
`20260808`.

Because the seed bases are recorded, **the games themselves are reproducible**:
the same seed base and task layout regenerate the same games. What is *not*
reproducible is the `game_id` label, which `GameRunner` builds as
`{processId}_{threadNo}_{counter}` — the OS process id differs on every run.
See [Known limitations](#known-limitations) for why that still matters.

Runtime: ~80s per game per core for the neural agent, ~48s for the heuristic
one on cluster hardware. At the template's defaults, ~6000 games across 32
tasks is **4–5 hours** of wall clock. Regenerating the full combined dataset is
a working day.

Output size: several GB. It is not in this repository and is not meant to be —
it is fully regenerable from the above.

Then gate it before spending training time on it:

```bash
python tools/verify_generated_data.py /path/to/data
```

Read-only. Checks shard integrity, label balance, opening diversity, and the
game-level structure the split depends on.

## 2. Split

```bash
python tools/split_dataset.py /path/to/data /path/to/split --val-fraction 0.1 --seed 0
```

Splits **by game, not by record**, 90/10. This is not a detail. Consecutive
states within one game are highly correlated and share an outcome label, so a
record-level split leaks: the model partially memorises specific trajectories,
and the resulting validation accuracy is inflated without looking anomalous.

Runtime: minutes, dominated by I/O.

## 3. Train

```bash
python training/train_local.py \
    --train-dir /path/to/split/train \
    --val-dir   /path/to/split/val \
    --epochs 3 --batch-size 256 --lr 5e-4 \
    --out-dir /path/to/models
```

The shipped model used 3 epochs at batch 256 and lr 5e-4. Writes a per-epoch
checkpoint, a `best_model.pth`, and `training_metrics.json`.

**We do not have the shipped model's training metrics.** The run that produced
it (8 August 2026) left only the checkpoint and the exported ONNX behind; no
`training_metrics.json` from that run survives, and its wall-clock cost was not
recorded. The metrics we *do* ship,
`models/ablation_heuristic_only_metrics.json`, belong to the
`ablation_heuristic_only.pth` checkpoint — 3 epochs, best validation accuracy
0.8050, best validation loss 0.4034. Do not attribute those figures to the
shipped model.

Runtime: hours, hardware-dependent. No figure is given here rather than a
guessed one.

## 4. Export to ONNX

```bash
python training/export_to_onnx.py \
    --checkpoint /path/to/models/best_model.pth \
    --out models/DeepSetsValueNetwork.onnx
```

The export deliberately bypasses `torch_geometric`'s `global_mean_pool`, which
lowers to a scatter-reduce that ONNX opset 14 cannot express as an average —
the exporter silently falls back to *replace* semantics instead of averaging.
At inference the agent always evaluates exactly one graph, so a plain per-node
mean is equivalent and exports cleanly as `ReduceMean`. The script verifies the
exported graph against the PyTorch model across a range of node counts and
refuses to write a model whose outputs differ by more than 1e-5.

It prints the output's size and SHA-256. Runtime: seconds.

## Reproducing the model file byte for byte

**The shipped `DeepSetsValueNetwork.onnx` was exported with PyTorch 2.2.2.**

An ONNX file embeds `producer_version`. Exporting the same checkpoint on any
other PyTorch version therefore produces a file with a **different SHA-256**,
even though the model is identical. We verified this directly: re-exporting
`models/deepsets_value_network.pth` on PyTorch 2.8.0 yields an identical
operator graph (`Gemm, Relu, Gemm, Relu, ReduceMean, Gemm, Relu, Concat, Gemm,
Relu, Gemm, Relu, Gemm`), identical initializer names, and bit-identical
weights — but a different file hash.

So:

- To reproduce `86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915`
  exactly, pin `torch==2.2.2`.
- On any other version, **a hash mismatch is expected and is not an error.**
  Verify the weights instead: load `models/deepsets_value_network.pth` and
  compare its 12 tensors against the ONNX initializers. They match exactly.

Nothing downstream depends on the file hash except the pinned integrity checks
in the benchmark harnesses, which exist to catch a stale or wrong model
reaching a run — not to assert a particular PyTorch version.

## 5. Verify

### Cross-language feature parity — the important one

```bash
python tools/verify_parity.py --data-dir /path/to/data --num-samples 50
```

The feature schema is defined twice: `training/StateParser.py` for training and
the `FeatureExtractor` class in `Bots/src/DeepSetsCore.cs` for inference. If
they disagree, the agent plays on features the network never saw, and nothing
else in the pipeline will tell you.

This runs **both real implementations** — the Python one directly, the C# one
through `tools/ParityCheck`, which reconstructs genuine `GameState` objects
from logged JSON and calls the actual production method — and diffs node
matrices and global vectors element by element, sampling across early, mid and
late game. It reports per-column max deltas even when everything passes, and
exits nonzero if any column exceeds threshold.

A test where both sides were written from the same spec by the same author
proves nothing. This one runs the code that ships.

Runtime: a minute, plus a Release build of `tools/ParityCheck`.

### Model agreement

```bash
python tools/diagnose_value_net.py --data-path /path/to/split/val --limit 10000
```

Streams real validation samples through both the exported ONNX model and the
PyTorch checkpoint, and reports agreement, calibration and accuracy overall and
bucketed by prestige clock. This is what surfaced the early-game AUC weakness
(0.797 in the lowest prestige-clock bucket against 0.971 in the highest) that
motivated `DeepSetsBlendBot`. Read-only; writes two histogram PNGs.

See [experiments/README.md](experiments/README.md) for the C#-inference
verification chain, which checks the C# path against PyTorch on real in-game
tensors rather than synthetic ones.

## 6. Benchmark

Requires `./scripts/fetch_baselines.sh` first.

Locally:

```bash
tools/benchmark.sh --bot-a DeepSetsBot --bot-b SakkirinaSolo \
                   --games 400 --jobs 8 --full
```

`--full` is the `--timeout 10` preset, which is what the reported numbers used;
`--fast` is `--timeout 1` and is for iteration only, not for numbers you intend
to trust.

On a cluster, edit `scripts/slurm_benchmark.sh`, submit, then aggregate:

```bash
python tools/aggregate_benchmark_results.py --out-dir /path/to/results
```

Both harnesses run **one game per OS process**. `GameRunner` reuses one bot
instance across `--runs N` and its stats counter only reports aggregates per
process, so `--runs 1` is what makes each game's outcome individually
attributable — including losses caused by a timeout or an illegal move rather
than by play.

Seats are swapped for half the games in every matchup. First-player advantage
is real, so an unswapped result is not a clean measurement of anything. The
aggregator prints the swapped and non-swapped rates separately: that is not
decoration, it is the check for a seat-swap inversion bug, which would drag the
aggregate toward a plausible-looking 50% while leaving the two per-seat rates
visibly disagreeing.

Scale and runtime: the cluster configuration is 10 matchups × 400 games = 4000
games, one per array task, ~60–90s per game, roughly **2.5 hours** of wall
clock at 32 concurrent tasks. Resumable — a task with no result file is retried.

**Before trusting any result, confirm the model actually loaded.** Both
harnesses verify the ONNX hash in `GameRunner`'s output before running a single
game, because an agent whose model fails to load does not crash; it falls back
to the heuristic and quietly measures the wrong thing.

## Known limitations

- **The tournament games cannot be reproduced here.** See
  [Which numbers come from where](#which-numbers-come-from-where).
- **The shipped model's training metrics and wall-clock cost were not
  recorded.** Only the checkpoint and the exported ONNX survive from that run.
- **Byte-identical ONNX export requires PyTorch 2.2.2.** The model itself
  reproduces exactly on any version; only the file hash does not.
- **Training data is not distributed** (several GB) but is regenerable, and the
  seed bases are recorded above, so the *games* regenerate identically. What
  does not carry over is the `game_id`, which embeds the OS process id. Since
  `tools/split_dataset.py` partitions on the sorted set of `game_id`s before
  shuffling, a regenerated dataset gets a **different train/val partition even
  with the same `--seed`**. A model retrained from a fresh dataset should
  therefore land close to the shipped one, not identical to it.
