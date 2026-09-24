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

| Matchup | Games | Measured on | Local estimate | Tournament | Gap |
|---|---|---|---|---|---|
| `DeepSetsBot` vs SakkirinaSolo | 400 | cluster | 76.3% | 70.6% | **+5.7** |
| `DeepSetsBot` vs BestMCTS3 | 150 | laptop | 88.0% | 69.5% | **+18.5** |

The overestimate is mild against SakkirinaSolo and **severe against
BestMCTS3** — nearly 19 points. Treat any local number in this repository as an
upper bound, and do not assume the bias is uniform across opponents: it plainly
is not.

The two rows are not equally strong evidence. The BestMCTS3 row is 150 games on
an unmanaged laptop, so it carries both a wider interval and whatever timing
noise a laptop contributes to a wall-clock-budgeted search; the SakkirinaSolo
row is 400 cluster games. The +18.5 gap is large enough that it is unlikely to
be an artefact of either, but it is the less carefully measured of the two and
should be re-measured on the cluster before it is leaned on.

Do **not** compare a local head-to-head figure against 69.21% or 68.24%. Those
are field-wide averages and are not the right denominator for a single matchup.

So, when citing:

- Tournament placings and win rates → the official results, not this repo.
- Anything produced by `tools/benchmark*.py` → a **local estimate**, labelled
  as such, expected to be optimistic by an opponent-dependent margin.
- Ablations and controls comparing our own variants against each other →
  local, but like-for-like, so the bias largely cancels.

## This fork's engine is not identical to the competition's

Two changes were made outside our own code so that the experiment harness could
report what it needs to. Both affect **only how finished games are tallied and
reported — neither changes how a game is played, scored or won.** No rule, no
move legality, no winner determination is touched. Numbers produced here remain
comparable to numbers produced on the competition's engine.

They are recorded here because they mean this fork's engine is no longer
byte-identical to the one the competition runs, and anyone diffing against
upstream should know why.

**`Engine/src/utils/GameEndStatsCounter.cs` — counts `PREPARE_TIME_EXCEEDED`
instead of throwing on it.** `GameEndReason.PREPARE_TIME_EXCEEDED` was missing
from the counter's `switch`, so it fell through to `default` and threw
`ArgumentOutOfRangeException`, killing the whole process instead of counting one
game. It was latent upstream because nothing reached it: a bot's
`PregamePrepare` has the full per-turn timeout to finish, and at the usual 10 s
that is far more than the DeepSets agents need to build an ONNX
`InferenceSession`. [`experiments/configs/time_scaling.json`](experiments/configs/time_scaling.json)
runs `--timeout` as low as 4 s, which makes it reachable on a loaded node. It
now falls into the same "other factors" bucket as the other non-clean end
reasons.

**`GameRunner/Program.cs` — prints one `GAME_END_REASON: <reason> WINNER:
<winner>` line per game.** `GameEndStatsCounter` pools `TURN_TIMEOUT`,
`INCORRECT_MOVE`, `BOT_EXCEPTION`, `INTERNAL_ERROR` and both
`PATRON_SELECTION_*` reasons together as "other factors". That is fine for a
self-play data run and useless for a benchmark at a 2 s budget, where a game
lost to a timeout is not a game lost to play and the two have to be reported
separately. The line is additive — every existing parser matches its own lines
by regex — and `tools/benchmark_cluster.py` uses it to classify each game as
clean / turn_limit / timeout / disqualification.

## Prerequisites

- .NET 8 SDK
- Python 3.9–3.11 with `torch`, `torch_geometric`, `onnxruntime`,
  `scikit-learn`, `numpy`, `matplotlib`
- `git` and `patch` (for `scripts/fetch_baselines.sh`)

```bash
./scripts/setup_python_env.sh                # pinned, CPU-only venv
source .venv/bin/activate

dotnet build TalesOfTribute.sln -c Release
./scripts/fetch_baselines.sh                 # needed for anything involving baselines
dotnet build TalesOfTribute.sln -c Release   # rebuild, now with the baselines
```

`scripts/setup_python_env.sh` pins **torch 2.2.2**, which is the version that
reproduces the shipped ONNX byte hash — see [Reproducing the model file byte for
byte](#reproducing-the-model-file-byte-for-byte). It installs CPU-only wheels
from PyTorch's CPU index; the cluster partition these experiments run on has no
GPUs, and the CUDA builds are several GB that would never be used. Python 3.12+
is rejected, because torch 2.2.2 publishes no wheels for it.

It also installs **`onnx`**, which is a separate package from `onnxruntime` and
is not pulled in by it. `torch.onnx.export` imports it lazily, at call time, so
a missing `onnx` does not surface on `import torch` — it surfaces inside
`training/export_to_onnx.py`, *after* training has finished. The pin is gated by
Python version, because `onnx >= 1.20` requires Python >= 3.10 while the
supported range here is 3.9–3.11: **Python 3.9 → `onnx==1.19.1`, Python 3.10+ →
`onnx==1.21.0`.** Both branches are live, as the table below shows.

### The two environments that produced models

| | Shipped model | Per-seed models |
|---|---|---|
| Where | macOS, local | cluster |
| Python | 3.10.18 | 3.9.25 |
| torch | 2.2.2 | 2.2.2+cpu |
| onnx | 1.21.0 | 1.19.1 |

**The `+cpu` build tag alone changes the exported ONNX hash, even for identical
weights.** It goes into the file's `producer_version` just as the version number
does, so a checkpoint exported under `2.2.2+cpu` and the same checkpoint exported
under `2.2.2` produce byte-different files. The per-seed models' hashes therefore
differ from the shipped model's **for reasons that have nothing to do with their
weights**, and a mismatch between them is not evidence that anything is wrong.

The two environments differ on three axes, and only the torch build tag matters.
Re-exporting `models/deepsets_value_network.pth` through
`training/export_to_onnx.py` reproduces the shipped hash
`86e0f9a8…a915` under all of:

| Python | torch | onnx | Exported hash |
|---|---|---|---|
| 3.10.18 | 2.2.2 | 1.21.0 | shipped hash (this is the shipped model) |
| 3.11.16 | 2.2.2 | 1.21.0 | shipped hash |
| 3.9.13 | 2.2.2 | 1.19.1 | shipped hash |

So **neither the Python minor version nor the `onnx` library version moves the
hash** — which is worth knowing, because it means the version gate above is
about installability only and has no bearing on reproducibility. The torch build
tag is the one axis left, and it is the one the cluster differs on.

This is the same mechanism described under [Reproducing the model file byte for
byte](#reproducing-the-model-file-byte-for-byte); it is called out here because
the two environments differ in practice, so it is a situation you will actually
hit rather than a hypothetical. The practical consequence is the one already
built into the harness: `allowed_onnx_sha256` is a *list*, and a per-seed model's
hash is added to it rather than chased back to the shipped model's.

`fetch_baselines.sh` also derives `SakkirinaScaled` (see
[the experiments](#7-paper-experiments)) alongside `SakkirinaGen` and
`SakkirinaHalf`.

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

| Run | Generating agent | `--seed-base` | Games | Notes |
|---|---|---|---|---|
| Heuristic | `SakkirinaGen` | `20260803` | 6,080 | Derived from SakkirinaSolo by `scripts/sakkirina_gen.patch`. |
| Neural self-play | `SakkirinaGenNeural` | `20260807` | 6,080 | |

**12,160 games combined**, an even split between the two runs.

`tools/generate_data.sh` adds `task_id * 1000000` to the seed base per task, so
the two runs' seed ranges cannot collide. The benchmark run used seed base
`20260808`.

**The seeds regenerate, the games do not.** A seed fixes the initial deal and
the engine's own RNG stream, so a rerun starts from the same position. It does
not fix what gets played: both generating agents are wall-clock budgeted
(`while (s.Elapsed < timeForMoveComputation) TreeSearch(...)`), so how many
search iterations fit into a move depends on machine load, JIT warm-up and GC,
and two runs of the *same binary* on the *same seed* can pick different moves
and then diverge permanently. This is measured, not hypothetical:
`experiments/verify_exp_bot_parity.py --self-check` runs one frozen agent
against itself twice from one seed and the two games diverged at move 5.

So a regenerated dataset is **similar in distribution, not identical**: same
openings, different play, different outcomes. Retraining on it should land close
to the shipped model, not on top of it. See
[Known limitations](#known-limitations).

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
    --epochs 3 --batch-size 256 --lr 5e-4 --seed 0 \
    --out-dir /path/to/models
```

The shipped model used 3 epochs at batch 256 and lr 5e-4. Writes a per-epoch
checkpoint, a `best_model.pth`, `training_metrics.json`, and a `run_config.json`
recording the seed and hyperparameters.

`--seed` (default `0`) seeds torch, numpy, python `random` **and the shuffle
buffer** — `training/stream_dataset.py` shuffles shard order and picks buffer
eviction slots inside the DataLoader worker processes, where the main process's
`random.seed()` does not reach, so `worker_init_fn` seeds each worker from
torch's per-worker seed. This script used to be unseeded, so every rerun
differed; it is now reproducible by default. Identical seeds give identical
weight initialisation and identical data order. They do **not** give bit-exact
loss curves: thread count, BLAS version and hardware all matter and a seed
controls none of them.

### Several seeds, on the cluster

A single trained model's win rate is one sample from a distribution, and
reporting it as the distribution is the usual way a result like this fails to
replicate. `scripts/slurm_train.sh` is an array over seeds that trains, exports
each to ONNX, and writes everything under a per-seed directory in `$HPCWORK`:

```bash
./scripts/setup_python_env.sh --venv-dir "$HPCWORK/tot_venv"   # once, on the login node
# edit the CHANGE_ME_* placeholders in scripts/slurm_train.sh
mkdir -p logs
sbatch scripts/slurm_train.sh                                   # --array=0-4 -> seeds 0..4
```

Each seed leaves `best_model.pth`, the per-epoch checkpoints,
`training_metrics.json`, `run_config.json`, a seed-tagged `.onnx` and a
`SHA256SUMS` under `$HPCWORK/tot_models/seed_NN/`. Nothing is written to `/tmp`:
the script refuses to run if `$HPCWORK` is unset rather than falling back, and
points `TMPDIR` at `$HPCWORK` too, because compute-node `/tmp` is node-local and
is wiped when the job ends.

Unlike the benchmark arrays (1 core per task, because one game is
single-threaded by construction), this asks for 8 cores and pins the thread
count to the allocation — left unset, OpenMP sizes itself from the machine's
total core count rather than the cgroup's.

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
python tools/aggregate_benchmark_results.py \
    --config legacy_paper_benchmark --out-dir /path/to/results
```

The matchup list, game count and `--timeout` now live in
[`experiments/configs/legacy_paper_benchmark.json`](experiments/configs/legacy_paper_benchmark.json),
which reproduces the previously-hardcoded set exactly — same order, same seeds,
same task ids, same result directory names — so an in-flight run resumes across
that change. `tools/benchmark_cluster.sh` defaults to it. **The aggregator needs
the same `--config` the run used.**

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
to the heuristic and quietly measures the wrong thing. The pin is now a *list*
of allowed hashes, since per-seed training produces several legitimate models;
it is still not skippable.

## 7. Paper experiments

Three questions across four configs — equal effort is asked twice, once from
each direction. One harness, one build. Each config is a JSON file in
[`experiments/configs/`](experiments/configs/) — see
[that directory's README](experiments/configs/README.md) for the format, and
[`experiments/README.md`](experiments/README.md) for the agents they use.

All four run `DeepSetsBotExp`, a copy of `DeepSetsBlendBot` with three
environment hooks and nothing else changed. The submitted agents stay
byte-identical; `SOT_ALPHA0=0` makes the copy behave as `DeepSetsBot` and the
default `0.7` makes it behave as `DeepSetsBlendBot`, so one class covers both.

| Config | Question | Tasks |
|---|---|---|
| `alpha_sweep` | How much of the advantage is the blend versus the network alone? | 2000 |
| `time_scaling` | Does the advantage hold as the per-turn budget moves 2 s → 30 s? | 2000 |
| `equal_effort` | Is the network better, or is the baseline just searching more? Treatment sped up. | 400 |
| `equal_effort_baseline_slowed` | The same question, baseline slowed down instead. | 400 |

### Equal effort is run in both directions

`equal_effort` and `equal_effort_baseline_slowed` target the same ratio —
evaluations per turn — and move opposite sides to get there. **Both were run,
because neither is decisive alone.**

One calibration served both: **n = 60 games, ratio of means *r* = 7.08 (roughly
6.4–7.8)**. Each config applies it to a different side. Figures below are
`DeepSetsBotExp` vs `SakkirinaScaled`, evaluations per turn over 400 games:

| Direction | Setting | Achieved |
|---|---|---|
| baseline slowed | `SOT_BASELINE_TIME_SCALE = 0.141` (= 1/*r*), `--timeout 12` | **6,433 vs 6,627** — within 3 % |
| treatment sped up | `SOT_TIME_SCALE = 9.0`, `--timeout 91` | **43,139 vs 49,131** — DeepSets at **0.88** of the baseline's effort |

**Quote the achieved figures, not *r*.** Only the baseline-slowed direction is
matched; the sped-up one is not, and should not be called matched — it left
DeepSets doing 12 % *less* work than the baseline. Note also that `9.0` is not
*r*: evaluations per turn is only approximately linear in the time budget, a
pilot at *r* = 7.08 undershot, and 9.0 was chosen empirically from that pilot.

Both directions ended with DeepSets doing slightly less work than the baseline
(0.97× and 0.88×), so in both the residual mismatch runs **against** the DeepSets
agent. That is the conservative direction for the claim: a win under these
conditions is not explained by the DeepSets side having been handed more search.

The distortions differ in kind. Speeding the treatment up keeps the baseline at
exactly the timing its author tuned it for, but hands `DeepSetsBotExp` an 88.2 s
per-turn budget no tournament would give it, so the agent measured is not the
agent submitted — and it costs roughly 9× the wall clock of any other row.
Slowing the baseline down keeps `DeepSetsBotExp` at exactly its competing timing
and costs no more than a normal row, but runs SakkirinaSolo's search at about a
seventh of the budget it was designed around, where a hand-tuned heuristic may
degrade non-linearly: its rule-based fast paths and tree reuse do not scale with
the clock, so the *shape* of its play changes, not only its depth.

What carries weight is **agreement**. If both move the win rate the same way,
the conclusion is robust to which side was moved. If they disagree, that is the
finding: evaluations per turn is not the right currency for "effort" here, and
the framing needs rethinking before either number is reported. **Lead with the
baseline-slowed direction** — it is both the cheaper and the better-matched of
the two — and let the sped-up one confirm it.

### Cluster commands, in order

```bash
# --- once, on the login node ---
# Clone FRESH, into a NEW directory. Do not pull into an existing
# $HOME/tot/ScriptsOfTribute-Core: that is a different, older repository, and
# none of the experiment configs, agents or harness changes below exist in it.
source $HOME/tot/env.sh
mkdir -p $HOME/tot
git clone -b experiments \
    https://github.com/DorukKaraman/deepsets-tales-of-tribute.git \
    $HOME/tot/deepsets-tales-of-tribute
cd $HOME/tot/deepsets-tales-of-tribute

./scripts/fetch_baselines.sh                      # SakkirinaSolo + SakkirinaScaled
dotnet build Bots/Bots.csproj       -c Release
dotnet build GameRunner/GameRunner.csproj -c Release
mkdir -p logs                                     # SLURM will NOT create this for you

# scripts/slurm_experiment.sh defaults REPO_ROOT to this path. If you cloned
# somewhere else, edit it there too, or every array task will look in the wrong
# place.
export OUT_DIR=$HOME/tot/experiment_results

# --- 1. alpha sweep ---
tools/benchmark_cluster.sh --config alpha_sweep --out-dir "$OUT_DIR/alpha_sweep" --dry-run
sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=0-1999%32 scripts/slurm_experiment.sh
python tools/aggregate_benchmark_results.py --config alpha_sweep --out-dir "$OUT_DIR/alpha_sweep"

# --- 2. time scaling ---
tools/benchmark_cluster.sh --config time_scaling --out-dir "$OUT_DIR/time_scaling" --dry-run
# 0-1599 is the required 2/5/10/20 s rows; 1600-1999 is the optional 30 s row
sbatch --export=ALL,SOT_EXP_CONFIG=time_scaling --array=0-1599%32 --time=00:20:00 scripts/slurm_experiment.sh
sbatch --export=ALL,SOT_EXP_CONFIG=time_scaling --array=1600-1999%32 --time=00:30:00 scripts/slurm_experiment.sh
python tools/aggregate_benchmark_results.py --config time_scaling --out-dir "$OUT_DIR/time_scaling"

# --- 3. equal effort. Both configs are already calibrated and filled in;
#        re-calibrate only if the hardware changed, since the ratio depends on it.
tools/benchmark_cluster.sh --config equal_effort_baseline_slowed \
    --out-dir "$OUT_DIR/equal_effort_baseline_slowed" --calibrate

# --- 3a. baseline slowed down. Cheap, and the better-matched direction. ---
#      SOT_BASELINE_TIME_SCALE=0.141 (= 1/r), timeout 12.
sbatch --export=ALL,SOT_EXP_CONFIG=equal_effort_baseline_slowed \
    --array=0-399%32 --time=00:15:00 scripts/slurm_experiment.sh
python tools/aggregate_benchmark_results.py --config equal_effort_baseline_slowed \
    --out-dir "$OUT_DIR/equal_effort_baseline_slowed"

# --- 3b. treatment sped up. Expensive: ~120 core-hours for 400 games. ---
#      SOT_TIME_SCALE=9.0, timeout 91 (= ceil(9.8 * 9.0) + 2).
sbatch --export=ALL,SOT_EXP_CONFIG=equal_effort \
    --array=0-399%32 --time=00:45:00 scripts/slurm_experiment.sh
python tools/aggregate_benchmark_results.py --config equal_effort --out-dir "$OUT_DIR/equal_effort"
```

**`--time` is not optional on 3b.** `slurm_experiment.sh` defaults to
`00:30:00`, sized for the 30 s rows of `time_scaling`. At `SOT_TIME_SCALE=9.0`
`DeepSetsBotExp` gets an 88.2 s per-turn budget, so a normal-length game takes
roughly 9× the usual wall clock and every task would be killed at the 30-minute
default with no result file. `00:45:00` leaves margin without over-requesting.
Budget **~120 core-hours** for the 400 games; at `%32` that is about 4 hours of
wall clock. The baseline-slowed direction costs a normal row's worth, which is
why it is the one to lead with.

`--array` may exceed the site's `MaxArraySize`
(`scontrol show config | grep -i MaxArraySize`); submit in chunks if so. Every
task is resumable — one whose result file exists is skipped without running
anything — so resubmitting a chunk, or the whole array, is always safe.

### Disqualifications and timeouts are reported separately

`GameRunner` now prints the exact `GameEndReason` per game, and the aggregator
splits the non-clean games into **timeout**, **disqualification** and
**turn limit**, attributed to the side that caused them. At a 2 s budget a game
lost to a timeout is not a game lost to play, and pooling the two (as the
engine's own "other factors" counter does) would make a budget that is simply
too tight look like an agent that is simply worse. `time_scaling.json` sets
engine `--timeout` to budget + 2 s to keep timeouts near zero; **if they are
not, raise the margin and re-run the row rather than reporting its win rate.**

## 8. Held-out evaluation

The benchmark measures agents. This measures *models*, on data none of them
were trained on.

```bash
# 1. generate a held-out set from two DIFFERENT agents
tools/generate_data.sh --games 2000 --out-dir "$HPCWORK/heldout" \
    --bot-a DeepSetsBotExp --bot-b SakkirinaSolo --seed-base 20260925

# 2. score every checkpoint on it, in one pass, on identical samples
python tools/evaluate_checkpoints.py \
    models/deepsets_value_network.pth \
    "$HPCWORK"/tot_models/seed_*/best_model.pth \
    --data-dir "$HPCWORK/heldout"
```

Neither shipped model was trained on games between `DeepSetsBotExp` and
`SakkirinaSolo`, so that pairing is genuinely unseen — which a fresh *self-play*
set from the same generator would not be, however new its games are.
`tools/generate_data.sh` alternates seats across jobs whenever the two bots
differ: first-player advantage is real and correlates with the outcome label, so
a set generated entirely with one agent in seat P1 would hand every metric a
systematic bias.

**That command sets no `SOT_ALPHA0`, so `DeepSetsBotExp` runs at its default
`0.7` — the set is generated by the *blend* agent, not the network-only one.**
That is a deliberate choice, not an oversight: the blend is the submitted
1st-place agent, so its games are the states a deployed model actually has to
evaluate. But it does mean the states are drawn from a policy that consults the
heuristic early, and a model scored on them is being asked about that
distribution specifically.

To generate from the pure-network agent instead, export the variable before
calling the script — `tools/generate_data.py` passes the ambient environment
straight through to `GameRunner`, so it reaches the bot:

```bash
SOT_ALPHA0=0 tools/generate_data.sh --games 2000 --out-dir "$HPCWORK/heldout_a0" \
    --bot-a DeepSetsBotExp --bot-b SakkirinaSolo --seed-base 20260927
```

Note this is the one place in the pipeline where a `SOT_*` variable is read from
the ambient shell. The benchmark harness deliberately strips them
(`tools/benchmark_cluster.py`'s `build_env`) so a stray export cannot leak into
a run that never asked for it; the generation path has no such guard, which
makes it usable here and worth being careful about elsewhere.

`tools/evaluate_checkpoints.py` reports loss, accuracy, AUC and Brier per
checkpoint, overall and by prestige-clock bucket, each against that slice's
majority-class baseline — the same buckets `train_local.py` prints during
training, so the numbers are directly comparable. It streams the dataset once
and evaluates every checkpoint per batch, which is what guarantees they are all
scored on byte-identical samples. The forward pass it uses is the *exported*
one (plain per-node mean rather than `global_mean_pool`), i.e. the path the
agent actually runs.

## Known limitations

- **The tournament games cannot be reproduced here.** See
  [Which numbers come from where](#which-numbers-come-from-where).
- **The shipped model's training metrics and wall-clock cost were not
  recorded.** Only the checkpoint and the exported ONNX survive from that run.
- **Byte-identical ONNX export requires PyTorch 2.2.2.** The model itself
  reproduces exactly on any version; only the file hash does not.
- **Training data is not distributed** (several GB) and regenerates only
  approximately. The seed bases are recorded above, so a rerun starts from the
  same deals — but the generating agents are wall-clock budgeted, so they do not
  replay the same moves, and the games that come out are different games with
  the same openings. The dataset is reproducible in distribution, not in
  content.

  Two independent things therefore differ on a regenerated dataset, and both
  push the same way. The records themselves are different, because the play was
  different. And the `game_id` labels are different, because `GameRunner` builds
  them as `{processId}_{threadNo}_{counter}` and the OS process id changes every
  run — which matters because `tools/split_dataset.py` partitions on the sorted
  set of `game_id`s, so **even the same records would land in a different
  train/val split at the same `--seed`**. A model retrained from a fresh dataset
  should land close to the shipped one, not identical to it, and a difference of
  a point or two is expected rather than evidence of a bug.
- **`DeepSetsBotExp` cannot be proven move-identical to `DeepSetsBlendBot`.**
  The search is wall-clock budgeted, so two runs of the *same* binary on the
  same seed can diverge. `experiments/verify_exp_bot_parity.py --self-check`
  measures that noise floor; the comparison is only meaningful against it.
- **The alpha sweep is not a paired design.** `seed = seed_base + task_id` and
  task ids are contiguous across matchups, so the five conditions play different
  games. They are independent 400-game samples; read small differences against
  the Wilson intervals the aggregator prints.
- **`time_scaling`'s 10 s row is not directly comparable to the legacy
  benchmark.** Every row there uses engine `--timeout` = budget + 2 s so that a
  turn spending its whole allowance is not scored as a timeout; the legacy
  benchmark used `--timeout 10` with no margin.
- **Equal effort is calibrated, not derived.** Evaluations per turn is only
  approximately linear in the time budget — tree reuse and the rule-based fast
  paths do not scale with the clock — so the matching `SOT_TIME_SCALE` is found
  by iterating `--calibrate`, and how closely the two agents actually meet
  should be reported alongside the win rate. Calibrate at `SOT_ALPHA0=0`: above
  0, one counted evaluation runs *both* evaluators inside the blend window, so
  the counter measures the same event on both sides but not the same work.
- **The engine is not byte-identical to the competition's.** Two tallying-only
  changes; see [above](#this-forks-engine-is-not-identical-to-the-competitions).
