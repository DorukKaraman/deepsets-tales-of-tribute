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
`training/export_to_onnx.py --no-flush` (the shipped file predates denormal
flushing, so reproducing it means reproducing that condition too — see [Denormal
flushing](#denormal-flushing-and-why-the-export-is-platform-dependent))
reproduces the shipped hash `86e0f9a8…a915` under all of:

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

#### How the five seed models were actually produced

Recorded because the run needed two manual interventions and one judgement call
about the split, none of which are recoverable from the artefacts alone.

**The split.** It was rebuilt on the cluster rather than transferred, and is
identical to the shipped model's **by construction, not by direct comparison**:
the input data matches by SHA-256 manifest, `tools/split_dataset.py` is
byte-identical across all four copies checked, `random.Random(0).shuffle`
produces the same ordering on Python 3.10.18 and 3.9.25, and the resulting game
counts match. What could *not* be done is the direct check — comparing the two
splits' game IDs — because the original split files had been deleted. Every link
in the chain is verified; the end-to-end equality is inferred from them. If a
seed model ever behaves oddly in a way that points at its training data, this is
the assumption to attack first.

**Environment.** Python 3.9.25, torch 2.2.2+cpu, onnx 1.19.1 — the cluster
column of the [two-environments table](#the-two-environments-that-produced-models)
in Prerequisites, which is also why these ONNX hashes differ from the shipped
model's for reasons unrelated to their weights.

**Memory.** Seed 0 peaked at *exactly* its 16 GB request, which is the signature
of a job pressing against its ceiling rather than one that happened to need
precisely that much — treat 16 GB as a lower bound, not a measurement. Seeds 1–4
were given 24 GB and ran without incident. `scripts/slurm_train.sh` now defaults
to 24 GB.

**Three interventions, none requiring retraining.** Seed 0's in-job export died
on the missing `onnx` package and was exported by hand afterwards; seeds 3 and 4
were rejected by the old fixed export tolerance and were re-exported once that
was fixed; and later all five were re-exported again with denormal flushing,
after the first `seed_benchmark` run turned out to have been measuring models
that ran four times slower than the shipped one. All three causes are now fixed
at the source — `onnx` is pinned in `scripts/setup_python_env.sh`, the tolerance
is relative, and flushing is the export default, all described in [Export to
ONNX](#4-export-to-onnx) — so a rerun should need none of them. The `.pth`
checkpoints were never touched by any of this, which is why the validation
losses below are still the losses of the weights that produced them.

**Results.**

| Seed | Best val loss | ONNX SHA-256 (flushed re-export — the current files) |
|---|---|---|
| 0 | 0.4385 | `b5b8b2bfac57d671b1dbccd0f1f7f60f4ccc95d845462f2a830ebd3922dd1258` |
| 1 | 0.4369 | `1efc662e08fa2cf2cce82da542d0d891fd66a90f3ad64450a89ce93fcb2512df` |
| 2 | 0.4402 | `56eb9f9113439848420e70674b1b2c8771d1b217df883e79dcd3b3a074976d90` |
| 3 | 0.4329 | `cb30b934961dbe37365faf7c6d22df5b3e98054c2f45fbaef1789de8edf32e00` |
| 4 | 0.4470 | `b77c2a6442a0685024f239fedcbe1a00fe1ce32159da633a40cbe57dffb8db84` |

Mean 0.4391, sd 0.005. Those hashes are what go into a config's
`allowed_onnx_sha256` when a seed's model is benchmarked via `SOT_MODEL_PATH`,
and they are the ones
[`experiments/configs/seed_benchmark.json`](experiments/configs/seed_benchmark.json)
carries.

**These are the post-flush hashes, and they superseded an earlier set.** The
original exports (`22558ce2…`, `88a73e81…`, `757bcee9…`, `1e5ad716…`,
`9cd893be…`) carried 14,000–19,000 subnormal weights each and ran roughly 15×
slower; the first `seed_benchmark` run measured those and is invalid. The
re-exports are the *same models* — `tools/compare_onnx_models.py` found zero
output difference across 2000 real states per seed — just without the dead
weights. Do not put the old hashes back into any config; `seed_benchmark.json`'s
`_first_run_was_invalid` block has the full account.

**These val losses are not comparable to the shipped model's 0.4034.** That
figure was measured on a different validation set, so the gap between it and
this table says nothing about either. Compare the five seeds against each other,
or score them all on one common set with `tools/evaluate_checkpoints.py` — that
is what it is for.

The spread is also **tighter than the ±0.02 run-to-run variation measured in
August**, and the two are not measuring the same thing: the August figure
predates the shuffle-buffer seeding fix, so it mixed genuine seed-to-seed
variation with nondeterministic data ordering.

**Both figures are weaker than they look, and for the same reason.** Every val
loss on this page was computed through a DataLoader that did not partition
shards across workers, so each run validated on a different resampled multiset
of `val/` — see
[Known limitations](#known-limitations). The 0.005 spread across the seeds and
the ±0.02 August noise floor therefore both mix model variance with sampling
variance, and neither is a clean measure of the quantity it names. Re-scoring
all six checkpoints on one common set with `tools/evaluate_checkpoints.py`,
which is single-process and unaffected, is what would separate them; that is
pending and has to run on the cluster against the full split's `val/`.

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
exported graph against the PyTorch model across a range of node counts, and
**writes to a temporary file, verifying before renaming into place** — a failed
verification leaves no `.onnx` behind at all, rather than an unchecked one with
no `SHA256SUMS` beside it.

Agreement is judged on a combined tolerance, `|diff| <= 1e-5 + 1e-6 × |torch
output|`, and both the absolute and the relative difference are printed per node
count. A fixed absolute bound does not work here: the network emits a raw logit
that reaches into the tens, and one correctly-rounded float32 bit at an output
of 128 is already 1.5e-5. Two of the per-seed exports were rejected by the old
fixed 1e-5 bound over differences of exactly 1 ULP, with relative errors of
1.19e-7 — float32 epsilon. Those exports were correct; the tolerance was not.

The verification inputs are drawn from a fixed seed, so the check is
deterministic: the same file gives the same verdict and the same printed numbers
on every run, and a failure can be reproduced by whoever has to diagnose it.
(Unseeded, it drew fresh inputs each run — which is why some per-seed exports
tripped the old bound and others did not, on identical code.) The seeding uses a
local `torch.Generator`, not `torch.manual_seed`, so importing this module
cannot perturb anyone else's random stream.

If verification ever does fail, read the **relative** column: ~1e-7 means the
tolerance needs revisiting, while anything orders of magnitude larger means the
graph is wrong, and the pooling operator is the first thing to check.

It prints the output's size and SHA-256. Runtime: seconds.

### Denormal flushing, and why the export is platform-dependent

**By default the export zeroes every weight with `|w| < 1e-30`**, on a copy — the
`.pth` is never modified, so the validation losses reported for a checkpoint
remain the losses of the weights that produced them. `--no-flush` turns it off.

This is a performance fix, not a numerical one. Subnormal arithmetic falls off
onnxruntime's fast path, and on the cluster `seed_00` measured a **median
870.7 µs per inference against 52.4 µs for a flushed copy — roughly 17×**, over
5000 real validation states. In the benchmark that showed up as the seed models
getting ~1,400 evaluations per turn where the shipped model got ~6,400, in a
comparison whose entire premise was that they searched equally. It invalidated
all five seed rows.

**It is platform-dependent, and that is the part worth remembering.** x86
executes subnormals slowly; Apple Silicon flushes them to zero in hardware and
pays nothing. Train on x86 and export without flushing and you get an agent that
is *correct* and several times slower than the one described here — which would
make this repository's throughput figures look unreproducible when the models
are in fact fine. That is the failure mode the default is there to prevent.

**The 17× is a cluster measurement and does not reproduce on a Mac.** Because
Apple Silicon flushes denormals in hardware, the same comparison there shows
roughly 1× — subnormal weights cost nothing, so there is no gap to see. Measured
while building this: on an M1, a float32 matmul against an all-subnormal operand
ran at 0.87× the time of a normal one, and comparing the shipped model against a
copy with 22% of its weights forced subnormal gave a timing ratio of 1.00×.
Checking the claim on a laptop and concluding it was imaginary is the easy
mistake here; run the timing check on the same x86 hardware the benchmark runs
on. `tools/compare_onnx_models.py` detects this situation and warns rather than
letting a 1× ratio read as reassurance — its header quotes 38× for `seed_00`
against the *shipped model*, where the 17× here is `seed_00` against a *flushed
copy of itself*: same phenomenon, different baseline, and the latter is the
cleaner comparison because only the flush differs.

The threshold sits far above the float32 subnormal boundary of ~1.18e-38, which
looks arbitrary and is not: zeroing *only* the subnormal weights left `seed_00`
at 130 µs rather than 52 µs, because weights that are merely near-subnormal
still produce subnormal *intermediates* once multiplied by inputs below 1. 1e-30
is an empirical value that clears it; 1e-20 and 1e-12 were also tested and also
gave zero output change.

Safety was checked on real states, not synthetic ones —
`tools/compare_onnx_models.py` compared `seed_00` before and after over 5000
validation states: max absolute difference exactly 0, zero states differing,
zero prediction changes. The in-export verification is a second, independent
check of the same thing, because it compares the *flushed* ONNX against the
*unflushed* PyTorch model.

> **The shipped model carries the same dead floor.** It contains no subnormal
> weights, but it does contain **16,017 weights (21.9%) between 1.2e-38 and
> 1e-30** — sitting just above the boundary rather than below it — and the flush
> zeroes those too. So an export of that checkpoint has two possible hashes:
>
> | Export | SHA-256 |
> |---|---|
> | `--no-flush` | `86e0f9a8…a915` — the shipped file |
> | default (flushed) | `b6ce22dc…a76e` |
>
> **These are the same model, not two models.** Compared over 800 real states,
> the two exports give a maximum absolute difference of *exactly zero* — no state
> differs at all, no prediction changes. The weights the flush removes are dead:
> they contribute nothing to any output the agent has ever produced. Only the
> bytes differ.
>
> `--no-flush` is therefore not a caveat attached to the flush; it is how you
> reproduce a file that was made before the flush existed. Reproducing
> `86e0f9a8…a915` means reproducing the conditions it was exported under, and
> `--no-flush` is one of those conditions in exactly the way `torch==2.2.2` is.
> Nothing needs re-exporting: the shipped file is the one every
> `allowed_onnx_sha256` and `models/SHA256SUMS` already refer to, and flushing
> would not have changed how it plays.

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
  exactly, pin `torch==2.2.2` and pass `--no-flush`. Both are the same kind of
  requirement: that file was exported before denormal flushing existed, so
  reproducing it means reproducing the conditions it was made under. A flushed
  export of the same checkpoint has a different hash but is **the same model** —
  bit-identical outputs on 800 real states. See [Denormal
  flushing](#denormal-flushing-and-why-the-export-is-platform-dependent).
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

Four questions across five configs — equal effort is asked twice, once from
each direction. One harness, one build. Each config is a JSON file in
[`experiments/configs/`](experiments/configs/) — see
[that directory's README](experiments/configs/README.md) for the format, and
[`experiments/README.md`](experiments/README.md) for the agents they use.

All five run `DeepSetsBotExp`, a copy of `DeepSetsBlendBot` with three
environment hooks and nothing else changed. The submitted agents stay
byte-identical; `SOT_ALPHA0=0` makes the copy behave as `DeepSetsBot` and the
default `0.7` makes it behave as `DeepSetsBlendBot`, so one class covers both.

| Config | Question | Tasks |
|---|---|---|
| `alpha_sweep` | How much of the advantage is the blend versus the network alone? | 2000 |
| `time_scaling` | Does the advantage hold as the per-turn budget moves 2 s → 30 s? | 2000 |
| `equal_effort` | Is the network better, or is the baseline just searching more? Treatment sped up. | 400 |
| `equal_effort_baseline_slowed` | The same question, baseline slowed down instead. | 400 |
| `seed_benchmark` | How much of the win rate is the training seed? Five per-seed models plus the shipped one, same games. | 2400 |

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

### Alpha sweep results

`DeepSetsBotExp` vs SakkirinaSolo, `--timeout 10`, 400 games per row.

| alpha0 | Win rate | 95% CI | as P1 | as P2 | evals/turn |
|---|---|---|---|---|---|
| 0.0 (pure network) | 79.3% | 75.0–82.9 | 88.5% | 70.0% | 6,722 |
| 0.3 | 80.5% | 76.3–84.1 | 87.5% | 73.5% | 7,595 |
| **0.5** | **82.0%** | 77.9–85.5 | 89.0% | 75.0% | 7,955 |
| 0.7 (the submission) | 74.8% | 70.3–78.8 | 89.0% | 60.5% | 8,605 |
| 0.9 | 71.8% | 67.2–75.9 | 82.0% | 61.5% | 9,669 |

Flat or slightly rising to 0.5, then falling. The first three are within noise of
each other (0.0 vs 0.5 is z ≈ 1.0, p ≈ 0.33), so the sweep does not establish a
peak at 0.5 so much as a plateau from 0.0 to 0.5. The fall afterwards is real at
the far end — **0.5 vs 0.9 is 10.2 points, z ≈ 3.4**, which survives a Bonferroni
correction for the 10 pairwise comparisons available (p ≈ 0.006 corrected). 0.5
vs 0.7 is 7.2 points, z ≈ 2.5, which does **not** survive correction (p ≈ 0.13
corrected) and should be read as suggestive only.

**The submitted `alpha0 = 0.7` is past the peak and no better than the pure
network** (0.0 vs 0.7: 79.3% against 74.8%). Note what this does and does not
say: it is a local measurement against one opponent, and `DeepSetsBlendBot` —
the 0.7 agent — is the configuration that won the tournament across the whole
field. This sweep is evidence about this matchup, not a verdict on the
submission.

Reported as an observation rather than a finding: **the drop from 0.5 to 0.7 is
almost entirely in the P2 seat** — 75.0% → 60.5% as P2, while P1 is identical at
89.0% in both. Nothing in the agent is seat-aware, so this is either a real
interaction with first-player advantage or a coincidence in two 200-game
half-samples. It was not predicted in advance and no test here is corrected for
having gone looking.

### Time scaling results

`DeepSetsBotExp` vs `SakkirinaScaled`, the same scale applied to both sides,
400 games per row. **This config runs the blend agent at `alpha0 = 0.7`** (the
default), so it is not the same agent as the equal-effort rows below.

| Budget | Win rate | 95% CI | as P1 | as P2 | ours/turn | baseline/turn | ratio |
|---|---|---|---|---|---|---|---|
| 2 s | 76.0% | 71.6–79.9 | 87.5% | 64.5% | 1,406 | 6,296 | 4.5× |
| 5 s | 79.5% | 75.3–83.2 | 86.0% | 73.0% | 3,981 | 21,110 | 5.3× |
| 10 s | 80.5% | 76.3–84.1 | 90.0% | 71.0% | 7,770 | 44,228 | 5.7× |
| 20 s | 77.0% | 72.6–80.9 | 90.0% | 64.0% | 14,552 | 88,657 | 6.1× |
| 30 s | 75.3% | 70.8–79.2 | 83.0% | 67.5% | 22,224 | 133,129 | 6.0× |

**Every interval overlaps every other.** The largest gap, 10 s vs 30 s at 5.2
points, is z ≈ 1.8 (p ≈ 0.08) before any correction for the ten comparisons
available — not significant. Across the full span the agent does **15.8× more
searching at 30 s than at 2 s and wins 0.7 points less**.

The throughput ratio column is worth its own glance: the baseline evaluates 4.5–6×
more positions per turn than we do at every budget, and that disadvantage is
roughly constant across a 15× range of clock. Whatever is producing the win rate,
it is not search volume.

### Equal effort results

Both directions, `alpha0 = 0` (pure network), 400 games each. See
[Equal effort is run in both directions](#equal-effort-is-run-in-both-directions)
for why both were run.

| Direction | Setting | Win rate | 95% CI | as P1 | as P2 | Achieved effort |
|---|---|---|---|---|---|---|
| Baseline slowed | `SOT_BASELINE_TIME_SCALE = 0.141` | 79.5% | 75.3–83.2 | 88.0% | 71.0% | 6,433 vs 6,627 — within 3 % |
| Treatment sped up | `SOT_TIME_SCALE = 9.0` | 76.5% | 72.1–80.4 | 88.5% | 64.5% | 43,139 vs 49,131 — 0.88× |

**The two directions agree**, which was the design's own criterion for believing
either: they are not significantly different from each other (z ≈ 1.0, p ≈ 0.31),
and neither differs from the stock-timing 79.3% at `alpha0 = 0` from the alpha
sweep (z ≈ 0.07 and z ≈ 0.95).

So equalising search — in either direction, at either end of a 7× clock
adjustment — leaves the win rate where it was. And recall that both directions
landed with the DeepSets agent doing slightly *less* work than the baseline
(0.97× and 0.88×), so the residual mismatch runs against the conclusion rather
than for it.

### Seed benchmark results

Job 4459158 (tasks 0–1999) plus the shipped row kept from job 4457064 (tasks
2000–2399): **240 COMPLETED, 0 WARN**, every game clean — no timeouts, no
disqualifications, nothing excluded.

**Check the throughput column before reading the win rates.** The seed rows
returned 5,771–6,921 evaluations per turn against the shipped row's 6,401, so
they straddle it rather than sitting at a quarter of it. The denormal fix took,
and these rows are comparing models at comparable search. (The first run of this
config failed exactly here; see
[`seed_benchmark.json`](experiments/configs/seed_benchmark.json)'s
`_first_run_was_invalid`, and the paired experiment it accidentally produced,
below.)

| Model | Win rate | 95% CI | as P1 | as P2 | evals/turn | val loss | val acc |
|---|---|---|---|---|---|---|---|
| seed 0 | 75.00% | 70.5–79.0 | 84.5% | 65.5% | 5,986 | **0.4343** | 78.24% |
| seed 1 | 75.00% | 70.5–79.0 | 83.0% | 67.0% | 5,865 | 0.4419 | 78.06% |
| seed 2 | 78.00% | 73.7–81.8 | 83.5% | 72.5% | 5,771 | 0.4437 | 77.95% |
| seed 3 | 76.00% | 71.6–79.9 | 86.0% | 66.0% | 6,207 | 0.4452 | 78.02% |
| seed 4 | 69.25% | 64.6–73.6 | 82.0% | 56.5% | 6,921 | 0.4396 | **78.28%** |
| **shipped** | **80.25%** | 76.1–83.9 | 85.5% | 75.0% | 6,401 | 0.4497 | 77.54% |

The val columns are the **re-scored** ones — all six checkpoints on the full
validation split, 308,809 states, one pass, byte-identical samples:

| Model | loss | accuracy | AUC | Brier |
|---|---|---|---|---|
| shipped | 0.4497 | 77.54% | 0.8714 | 0.1491 |
| seed 0 | **0.4343** | 78.24% | 0.8764 | **0.1441** |
| seed 1 | 0.4419 | 78.06% | 0.8745 | 0.1459 |
| seed 2 | 0.4437 | 77.95% | 0.8735 | 0.1467 |
| seed 3 | 0.4452 | 78.02% | 0.8753 | 0.1468 |
| seed 4 | 0.4396 | **78.28%** | **0.8779** | 0.1448 |
| *(majority baseline)* | — | *50.58%* | — | — |

`ablation_heuristic_only.pth` scores 0.4585 / 76.61% on the same pass, but that
number is **not clean**: it was trained on a different split, so some of these
validation games may be in its training set. It is listed for completeness and
should not be compared with the rows above.

```bash
python tools/evaluate_checkpoints.py models/deepsets_value_network.pth \
    "$HPCWORK"/tot_models/seed_0*/best_model.pth --data-dir "$SPLIT/val"
```

**The ranking changed completely when the loader was fixed.** The val losses
previously recorded here came from the resampled multiset the buggy DataLoader
produced (see [Known limitations](#known-limitations)): 0.4385 / 0.4369 /
0.4402 / 0.4329 / 0.4470, which made seed 3 the best of the five and seed 4 the
worst. On the full set, scored properly, **seed 3 is the worst of the five and
seed 4 is second best**. The two orderings have Spearman ρ = −0.50 — the old
ranking was not a noisy version of the right one, it was unrelated to it. Any
statement that rested on those numbers has to be re-derived, not merely
re-checked.

**The offline ranking does not predict playing strength.** Two inversions are
visible in the joined table above:

- **the shipped model is last of the six on every offline metric** — highest
  loss, lowest accuracy, lowest AUC, highest Brier — **and has the highest win
  rate**, 80.25%;
- **seed 4 is best on accuracy and AUC** and is **the weakest player** of the
  six at 69.25%.

State this modestly. Six models, offline losses spanning 0.0154 and accuracies
spanning 0.74 points, against win rates carrying ±2.2 points of binomial noise
each: rank statistics on that are not stable, and a single swap would move them
a long way. The claim supported here is that **the offline ranking is
uninformative about playing strength on this evidence, and is inverted in two
specific cases** — not that the two are anticorrelated. What it does establish
is that picking a checkpoint by validation loss is not a substitute for playing
it, which is the reason the seed benchmark exists as a separate experiment.

Seed mean **74.65%**, SD **3.26** points against binomial noise of ~2.17 points
per row. Homogeneity across the five seeds: χ² ≈ 9.0, df 4, **p ≈ 0.06** — so
the spread is a little wider than sampling alone predicts, but it does not clear
the conventional threshold, and it is not evenly distributed. Drop seed 4 and
the remaining four give χ² ≈ 1.3 on df 3 (p ≈ 0.7): seeds 0–3 are as tight as
400-game samples of one underlying model would be. Whatever between-seed
variation exists here is essentially one seed's worth.

#### The shipped model is top of the distribution, and this experiment cannot say why

It beats the seed pool by **+5.6 points** (z ≈ 2.4, p ≈ 0.02) — statistically
real. But it is **confounded, and not repairably so within this experiment**: it
differs from the five seeds in training platform and environment (Apple Silicon,
Python 3.10) as well as in seed. "The shipped model got a lucky seed" and "the
shipped model's environment produced a better model" predict exactly this result
and cannot be told apart from it. Do not report the +5.6 as a seed effect, and
do not report it as an environment effect either.

Its **+2.25 over the best individual seed is not significant** (z ≈ 0.8,
p ≈ 0.43). The defensible statement is that the shipped model sits at the top of
a distribution whose spread is roughly three points, not that it is better than
the models in it.

#### Seed 4 is the outlier, and the comparison is post hoc

At 69.25% it is **6.75 points below the other four pooled** (z ≈ 2.8,
p ≈ 0.006). That p-value is the one to distrust: seed 4 was selected for testing
*because* it looked low, so the nominal significance is inflated by an unknown
amount and should be read as "worth a second look", not as a finding.

Noted without over-reading it: seed 4 has both the **highest validation loss**
(0.4470) and the **highest evaluations per turn** (6,921). It searches the most
and plays the worst. With n = 5 that is an observation, not a relationship —
across all five, val loss and win rate do not line up cleanly either (seed 3 has
the best loss and the second-best win rate; seed 2 has the third-best loss and
the best win rate).

#### The invalid first run is an accidental paired experiment

This is the most informative thing in the section, and nobody designed it.

The first run measured the same five models on the same 400 games each, with one
difference: the unflushed models searched roughly **4.5× less** (about 1,400
evaluations per turn against about 6,150 now). The models themselves were
identical — `tools/compare_onnx_models.py` found zero output difference across
2000 real states per seed, so flushing changed the speed and nothing else.

| | evals/turn | seed win rates | mean |
|---|---|---|---|
| First run (unflushed) | ~1,400 | 78.5 / 75.5 / 79.5 / 78.0 / 68.0 | **75.90%** |
| Rerun (flushed) | ~6,150 | 75.0 / 75.0 / 78.0 / 76.0 / 69.25 | **74.65%** |

A 1.25-point difference from quartering the search volume. Because the two runs
played **the same games**, this can be tested pair by pair rather than as two
aggregates, which is the stronger analysis — McNemar over the games whose outcome
changed.

The pairing was verified rather than assumed: both runs have the same task ids,
the same task → seed map, the same seat assignments, and no draws or non-clean
games in either, so all 400 pairs per row match exactly and none are excluded.

| Row | b (first W, rerun L) | c (first L, rerun W) | Discordant | Agreement | χ²(1) | Exact p |
|---|---|---|---|---|---|---|
| seed 0 | 58 | 44 | 102 | 74.5% | 1.66 | 0.198 |
| seed 1 | 51 | 49 | 100 | 75.0% | 0.01 | 0.920 |
| seed 2 | 50 | 44 | 94 | 76.5% | 0.27 | 0.606 |
| seed 3 | 53 | 45 | 98 | 75.5% | 0.50 | 0.480 |
| seed 4 | 65 | 70 | 135 | 66.2% | 0.12 | 0.731 |
| **Pooled** | **277** | **252** | **529** | **73.5%** | **1.09** | **0.297** |

χ² is the continuity-corrected McNemar statistic; the exact p is the two-sided
binomial on the discordant pairs. **No row is significant and neither is the
pool.** Quartering the search volume moved nothing detectable, on the most
sensitive test the data supports.

The pairing helped, though less than one might hope: the standard error on the
difference falls from 1.36 points unpaired to **1.15 points paired**. The reason
is in the agreement column — the two runs agree on only 73.5% of games, against
62.8% expected if they were independent. A shared seed fixes the opening deal,
not the game: the search is wall-clock budgeted and its trajectory diverges
anyway. So the pairing removes the variance due to which cards were dealt, and
leaves the variance due to play. (Seed 4 is again the outlier — lowest agreement
at 66.2%, most discordant pairs at 135.)

**This is the cleanest of the four search-volume results**, because nothing was
deliberately varied. Identical models — `compare_onnx_models.py` verified zero
output difference — identical games, identical everything except a confound that
was accidental and complete. There was no experimenter degree of freedom in it
at all, where the other three each required choosing a scale, a budget or a
matching criterion. See [Search volume: four experiments, one
answer](#search-volume-four-experiments-one-answer).

The first run's per-game files are archived at
`experiment_results/calibration/seed_benchmark_unflushed_seeds/`, with a
`README.txt` marking them invalid as win rates. They remain valid as the other
arm of this comparison, which is the reason to keep them.

### Search volume: four experiments, one answer

Four of the results above bear on the same question from different directions,
and **the convergence is the result — not any single row.**

| Evidence | What was varied | Search change | Win-rate effect |
|---|---|---|---|
| `time_scaling` | Per-turn budget, both sides | 15.8× across the span | −0.7 pts, every interval overlapping |
| `equal_effort`, baseline slowed | Baseline's clock, ÷7 | matched to within 3 % | 79.5% vs 79.3% stock (z ≈ 0.07) |
| `equal_effort`, treatment sped up | Our clock, ×9 | matched to 0.88× | 76.5% vs 79.3% stock (z ≈ 0.95) |
| Seed benchmark, accidental pair | Nothing — inference speed only | ~4.5× | −1.25 pts, McNemar p ≈ 0.30 |

**In this matchup, search volume barely moves the win rate.** Four experiments,
four different mechanisms for changing it, no detectable effect in any of them.

The four are not equally strong, and they fail in different ways, which is what
makes the agreement worth something:

- `time_scaling` varies the budget for **both** agents, so it tests whether the
  matchup as a whole is budget-sensitive, not whether *our* search matters.
- The two `equal_effort` directions each move **one** side, and each distorts
  the agent it moves — one gets a budget no tournament would give it, the other
  runs at a seventh of what it was tuned for.
- The **accidental pair is the cleanest**: identical models, identical games,
  nothing deliberately varied, no experimenter degree of freedom, and the only
  one testable pair-by-pair rather than as two aggregates.

Each could be explained away on its own. Explaining away all four requires four
separate explanations pointing the same direction.

**What this does not license.** These are all one matchup — `DeepSetsBotExp`
against SakkirinaSolo or a rescaled copy of it — measured locally, at 400 games
per cell. "Search volume does not matter in this matchup" is supported. "Search
volume does not matter" is not, and neither is any claim about the tournament
field, where the same agents met seven other opponents on someone else's
hardware. See [Which numbers come from where](#which-numbers-come-from-where).

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

## 9. Flat-MLP ablation

Does the DeepSets *structure* contribute, or do the 99-dim card features carry
the result on their own? The ablation replaces the set encoder with a plain MLP
over the same cards laid out as one fixed-size padded vector, changing the
architecture and nothing about the features.

```bash
python training/train_flat.py --arch matched \
    --train-dir "$SPLIT/train" --val-dir "$SPLIT/val" \
    --epochs 3 --batch-size 256 --lr 5e-4 --seed 0 --out-dir "$OUT/flat_matched"
python training/train_flat.py --arch wide   ... --out-dir "$OUT/flat_wide"

python tools/verify_flat_parity.py --data-dir "$SPLIT/val" --num-samples 300
( cd training && python export_flat_to_onnx.py \
      --checkpoint "$OUT/flat_matched/best_model.pth" \
      --out "$OUT/FlatValueNetwork_matched.onnx" --arch matched )
```

`train_flat.py` takes the same arguments as `train_local.py` and writes the same
artefacts, because it *is* `train_local.py` — `train_model` grew a
`model_factory` parameter rather than being forked, so both arms share one
optimizer, scheduler, metric and checkpointing path and differ only in the
model.

### Two configurations, because one would not settle it

| config | shape | parameters | vs DeepSets |
|---|---|---|---|
| DeepSets | 99→128→128 pooled, 256→128→64→1 | 73,089 | 1.00× |
| `--arch matched` | 12,691→5→128→64→1 | 72,549 | 0.99× |
| `--arch wide` | 12,691→128→128→64→1 | 1,649,409 | 22.57× |

At a 12,691-dim input the first layer costs 12,691 weights per unit, so a
73k budget caps it at `73,089 / 12,692 = 5.75` units *even if the rest of the
network were free*. Five is what fits. That is a severe bottleneck, and beating
a model that starved would prove little — hence `wide`, which pays the capacity
off and asks the structural question separately.

Choosing a smaller `MAX_NODES` does not rescue it: the ceiling is 7 units at 96
nodes, 11 at 60, 12 at 48. There is no cap at which a parameter-matched flat
model over this input is not a single-digit bottleneck. The narrowness is the
cost of having no shared per-card encoder, which is the thing under test.

### Why MAX_NODES is 128 and nothing is truncated

The node count was measured over all 3,116,065 states of both generation runs:

| | min | median | mean | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|
| pooled | 25 | 33 | 34.3 | 44 | 49 | 61 | 77 | **128** |
| heuristic | 25 | 34 | 35.3 | 45 | 50 | 61 | 75 | 96 |
| neural | 25 | 31 | 33.2 | 44 | 49 | 60 | 79 | 128 |

A smaller cap looks nearly free — 96 truncates 0.011% of states, 60 truncates
1.01% — and it is not. Truncation drops nodes from the end of
`json_to_pyg_graph`'s emission order, which is `ENEMY_UNSEEN` (enemy hand and
draw): the largest single contributor at 10.2 nodes per state and 30% of all
nodes, removed *only* in the states that have the most of it. That is not 1% of
states mildly degraded, it is 1% of states with a systematically chosen part of
their input deleted — input the DeepSets model does see. Any accuracy gap could
then be blamed on missing information rather than missing structure, and the
ablation would answer a different question than the one asked.

**The maximum of 128 is one game.** The tail above 96 is flat at 7–9 states per
value all the way up, which is a trajectory rather than a distribution: game
`298494_0_100`, a 1,071-state runaway that contributed 499 of the corpus's 936
states above 85 nodes. Of the 380 games in the shards containing it, 377 never
exceed 86. A fixed-size encoding must budget for that one game and then carry
the empty space through every ordinary state.

**Padding waste is a property of the data, not only a problem for this
baseline.** At `MAX_NODES = 128` the median 33-node state leaves about 74% of
the input vector as structural zeros; the mean leaves 73%, and even the 99th
percentile at 61 nodes leaves 52%. The set encoder never allocates that space.
That is an argument for the architecture, not an artefact of how the baseline
was built.

### Accuracy

Full corpus, on the cluster: the same 90/10 game-aware split every other model
in this document was trained on, three epochs, `--seed 0`, all three arms
identical apart from the network. The DeepSets row is a **control trained
here**, not the shipped model, so that all three share a split, a seed, a loop
and a loader.

Validation loss by epoch, best in bold:

| arm | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| deepsets | 0.4348 | **0.4290** | 0.4304 |
| matched | 0.4727 | **0.4635** | 0.4723 |
| wide | **0.5325** | 0.5562 | 0.5848 |

All three then scored in one pass on the full validation split — 308,809
states, byte-identical samples, majority baseline 50.58%:

| arm | parameters | loss | accuracy | AUC | Brier |
|---|---|---|---|---|---|
| **deepsets** | 73,089 | **0.4290** | **78.65%** | **0.8800** | **0.1419** |
| matched | 72,549 | 0.4635 | 76.73% | 0.8606 | 0.1533 |
| wide | 1,649,409 | 0.5325 | 75.83% | 0.8520 | 0.1670 |

**DeepSets wins all sixteen cells**: best of the three on loss, accuracy, AUC
and Brier, in every one of the four prestige-clock buckets, without exception.

| bucket | n | loss (ds / matched / wide) | accuracy (ds / matched / wide) |
|---|---|---|---|
| [0.00, 0.25) | 130,784 | 0.5643 / 0.5876 / 0.6453 | 69.63% / 67.65% / 66.68% |
| [0.25, 0.50) | 64,605 | 0.4346 / 0.4761 / 0.5414 | 80.33% / 77.93% / 76.80% |
| [0.50, 0.75) | 51,465 | 0.3313 / 0.3735 / 0.4633 | 85.08% / 83.74% / 83.12% |
| [0.75, inf) | 61,955 | 0.2187 / 0.2629 / 0.3429 | 90.59% / 88.82% / 88.07% |

Game-clustered paired tests over the 1,216 validation games, Bonferroni ×4 —
**all four significant**, and not marginally:

| comparison | metric | mean diff | SE | t(1215) | p | games better |
|---|---|---|---|---|---|---|
| deepsets − matched | loss | −0.0293 | 0.0039 | −7.61 | < 1e-9 | 798/1216 |
| deepsets − matched | accuracy | +1.83 pts | 0.28 | +6.47 | < 1e-9 | 699/1118 |
| deepsets − wide | loss | −0.0862 | 0.0064 | −13.48 | < 1e-9 | 799/1216 |
| deepsets − wide | accuracy | +2.92 pts | 0.33 | +8.90 | < 1e-9 | 772/1159 |

```bash
python tools/clustered_significance.py --baseline deepsets ablation_per_state.csv.gz
```

**For scale: the loss gap to `matched` is about 2.7× the entire spread across
the five DeepSets training seeds** (0.0293 against 0.0109 from seed 0's 0.4343
to seed 3's 0.4452). The architecture difference is comfortably larger than the
seed noise it would have to clear.

**Capacity still does not close the gap — it widens it.** `wide` has 22.57× the
parameters of `matched` and is worse on every metric and in every bucket, and
its validation loss rises monotonically across all three epochs (0.5325 →
0.5562 → 0.5848) while it keeps fitting the training set. The starvation
objection to the matched arm does not survive this row, which is the only
reason it was run.

Artefacts, all `--seed 0`, `--num-workers 7`, three epochs, batch 256, lr 5e-4:

| arm | ONNX sha256 | best val loss |
|---|---|---|
| deepsets | `4619288bd96495270b09b1f1e7a1589dded8481a84bfc12fb2faebd4c6ad464a` | 0.4290 |
| matched | `bbbccfb7b060b2390cc4e02c865d150c72e19d253c4e89cf4fcf79422fa40071` | 0.4635 |
| wide | `26c396d87dee943273501504482ef1f260fd0bec1661ec239a0c3dc6b0806db8` | 0.5325 |

**Memory: the DeepSets arm peaked at 19.3 GB in epoch 3, so `--mem=24G` is the
right allocation and 16 GB is not.** The shuffle buffer dominates
(`DEFAULT_SHUFFLE_BUFFER_SIZE = 100,000` parsed graphs, per worker), so this
scales with that constant and the worker count rather than with the dataset.

One operational note for anyone repeating the scoring step:
`tools/evaluate_checkpoints.py` was **OOM-killed** on the first attempt at
scoring all three arms over the full 308,809-state split, and had to be re-run
with more memory. Measured locally, its resident set grows quickly and then
plateaus at roughly 0.6 GB for three checkpoints — it does not grow linearly
with the number of states — so this is an allocation to request explicitly, not
a leak to work around. The per-state CSV writer streams and contributes
nothing to it.

#### The subset pilot

Before the cluster run, the same three arms were trained locally on a 760-game
subset (175,739 train / 20,138 val) to decide whether the full run was worth
the time. It is kept here because it is what the decision was made on, and
because comparing it with the full run is informative about how far a pilot of
that size can be trusted.

| arm | subset loss | subset acc | full-corpus loss | full-corpus acc |
|---|---|---|---|---|
| deepsets | 0.5016 | 75.57% | 0.4290 | 78.65% |
| matched | 0.5556 | 73.17% | 0.4635 | 76.73% |
| wide | 0.5687 | 73.29% | 0.5325 | 75.83% |

The pilot got the **ordering** right and the **significance** wrong: its
DeepSets-versus-matched loss gap was 0.0553 at p = 0.0071, where the full run
gives 0.0293 at p < 1e-9. The pilot overstated the gap by roughly a factor of
two on 76 games, and could not resolve the accuracy difference at all
(p = 0.086 there, p < 1e-9 here). A 6% subset was enough to justify the cluster
time and not enough to report.

### Throughput

`tools/compare_onnx_models.py`, 5,000 real validation states, single-threaded,
**on an x86 compute node** — the hardware the benchmark runs on:

| model | median µs | mean µs | vs DeepSets |
|---|---|---|---|
| DeepSets | 49.0 | 49.5 | — |
| matched | 60.7 | 60.5 | **1.24× slower** |
| wide | 240.0 | 251.7 | **4.7× slower** |

(The DeepSets column is measured afresh in each pairing; it reads 49.0 µs
against `matched` and 51.0 µs against `wide`.)

**Do not quote the local figures.** An earlier measurement of the same three
files on an Apple M1, with the pinned x86_64 Python under Rosetta, put
flat-matched at **1.45× faster** than DeepSets. The x86 figure is 1.24× slower.
Same models, same tool, same states, **opposite conclusion** — the sign of the
result is a property of the host, not of the architecture. Only the x86 numbers
above belong in the paper, for the same reason the denormal measurement in
[section 4](#denormal-flushing-and-why-the-export-is-platform-dependent) does:
an agent's search rate is whatever the cluster gives it.

**The MAC arithmetic failed on both machines.** Counting multiply-accumulates,
DeepSets runs its node encoder once per card — about 1,002,000 MACs at the
median 33-node state — against either flat model's constant ~72,000, predicting
the flat model should be roughly **14× cheaper**. It came out 1.45× faster on
one host and 1.24× slower on the other. Neither is 14×, and the two do not even
agree in direction. A 12,691→5 matvec streams 63,455 weights to produce five
numbers: it is memory-bound, so its arithmetic is nearly free and its loads are
not, while 33 batched 99→128 rows is a shape onnxruntime handles well.
Arithmetic intensity decides this, and a MAC count cannot see it. `wide` loses
for a plainer reason: 1,649,409 weights is 6.6 MB, past any useful cache.

### Most of the flat models' parameters are dead

The export flush (`|w| < 1e-30`, see
[section 4](#denormal-flushing-and-why-the-export-is-platform-dependent))
zeroed:

| arm | parameters | zeroed | % |
|---|---|---|---|
| deepsets | 73,089 | 19,119 | 26.2% |
| matched | 72,549 | 44,872 | **61.9%** |
| wide | 1,649,409 | 1,201,515 | **72.8%** |

A parameter-matched model with 61.9% of its weights identically zero is not
really carrying 72,549 parameters. Three mechanisms produce that, and they were
measured rather than assumed — the obvious explanation turns out to be the
smallest of them:

**1. Dead features, shared by every architecture.** 43 of the 99 card-feature
dimensions are never nonzero on any card in the data — card effects that no
competition card has. **Every architecture zeroes 100% of the weights attached
to them**, against 4.3% (deepsets) to 18.9% (wide) on live features. The
per-feature dead pattern correlates r = +0.92 between DeepSets' node encoder
and the flat models' first layer, and r = +0.998 between the two flat models.
This is a property of the feature schema, not of flattening.

**2. Per-slot specialisation, unique to flattening.** `json_to_pyg_graph`
emits cards in a fixed order — tavern, then hand, then played, and so on — so
each slot only ever holds a narrow band of the feature space, and the flat
model keeps a private copy of all 99 weights for every slot. Slot 0 is always a
tavern card: it uses 1 of the 9 location values, and 88.9% (8/9) of its
location weights are dead. Slot 20 sees 6 locations and 33.3% (3/9) are dead.
The dead fraction tracks `(9 − locations used)/9` almost exactly. DeepSets has
one shared encoder and so pays this once, not 128 times.

**3. Padding, which is real but the smallest of the three.** Dead weights do
rise with slot emptiness exactly as expected — from 51.1% on slots 0–24
(occupied in 100% of states) to 92.5% on slots 96–127 (occupied in 0.01%),
correlation −0.59 between occupancy and deadness. But slots that are *always*
occupied are already half dead from mechanisms 1 and 2. Holding every slot to
the always-occupied rate, padding accounts for **13.7 points of matched's
61.9%** and 19.4 of wide's 72.8%.

The amplifier behind all three is structural: **the flat model spends 87.5% of
its parameter budget on the input layer** (63,455 of 72,549), where input
sparsity bites, while DeepSets spends 17% there (12,672 of 73,089) and reuses
that layer across every card. The same per-feature death rate therefore costs
the flat model five times as much of its budget.

What this means for the comparison, stated carefully:

- **"Parameter-matched" means matched by count.** Both models were given ~73,000
  weights; they are not carrying equivalent effective capacity, and the flat
  model's shortfall is not an artefact of how it was initialised or trained.
- **Much of the flat model's allocated capacity serves inputs that are usually
  empty**, because a fixed-size encoding must budget for the largest state and
  then carry that space through every ordinary one — 74% of the input vector is
  padding at the median state.
- **This is the cost of padding, not a flaw in the comparison.** It is what
  flattening a variable-size set actually costs, and a fixed-size encoder
  cannot avoid it. Reporting the models as parameter-matched and then noting
  that the flat model cannot use its match is the honest description.
- **More capacity does not close the gap.** The `wide` arm has 22.57× the
  parameters, 72.8% of them dead, and loses by more than `matched` does on
  every metric. Whatever the flat model is missing, it is not budget.

### What is not built

No C# encoder and no bot. A flat bot would need no new encoder: the exported
graph takes the same `(node_features, global_features)` inputs as the DeepSets
model and pads internally, so `DeepSetsCore.cs` can load any of the three
unchanged — verified on signature, dtype, output name and scale.
[`experiments/configs/ablation_benchmark.json`](experiments/configs/ablation_benchmark.json)
is ready to run that comparison in games, which is the test that matters given
that the offline ranking of the seed models
[did not predict their playing strength](#seed-benchmark-results).


## Known limitations

- **The tournament games cannot be reproduced here.** See
  [Which numbers come from where](#which-numbers-come-from-where).
- **The shipped model's training metrics and wall-clock cost were not
  recorded.** Only the checkpoint and the exported ONNX survive from that run.
- **The shipped model and the five seed models were each trained on a
  non-uniform subset of the shards, a different one every epoch.**
  `training/stream_dataset.py` shuffled the shard list with each worker's *own*
  RNG state and then took `shards[worker_id::num_workers]`. Slices of different
  permutations are not a partition, so in any given epoch some shards were read
  by several workers and others by none.

  **This is fixed** — the current code partitions first and shuffles within each
  worker's own share, and `tools/test_stream_dataset_sharding.py` is the
  regression test (it fails on the old strategy at every worker count ≥ 2 and
  passes on the new one). The models were not retrained, so the figures below
  describe the runs that produced the artefacts this repo ships.

  Measured over the real 128-shard training corpus, 400 simulated runs using
  the DataLoader's own seeding path, verified against a live multi-worker
  loader:

  | configuration | missed/epoch | duplicated/epoch | distinct/epoch | never in 3 epochs |
  |---|---|---|---|---|
  | `--num-workers 4` — shipped model, 8 Aug | 31.7% ±2.6 | 26.2% ±2.3 | 68.3% | 3.1% ±1.4 |
  | `--num-workers 7` — cluster seed runs | 33.9% ±2.6 | 26.3% ±2.1 | 66.1% | 4.0% ±1.5 |

  Four was `train_local.py`'s `--num-workers` default at the shipped model's
  training commit; seven is `scripts/slurm_train.sh`'s `CPUS_PER_TASK - 1`. Each
  worker picked a given shard independently with probability `1/nw`, so the miss
  rate is `(1 - 1/nw)^nw` — 0.3164 and 0.3399, tending to `1/e`. The measurement
  lands on those values, which is what identifies the mechanism rather than just
  the symptom.

  **This does not affect any win rate.** Every benchmark in
  [section 7](#7-paper-experiments) plays the models as they are, against
  opponents, and measures what they do. A model trained on two-thirds of the
  shards per epoch is simply the model that exists; its 79.3% is its 79.3%. The
  same holds for the seed benchmark, the alpha sweep, the time scaling and both
  equal-effort directions. Nothing in those results is contingent on how the
  training data was sampled.

  **It does affect every recorded validation loss**, because validation ran
  through the same loader: the seed table above, the 0.4058/0.4437 comparison,
  the architecture probes, the 0.4034 attached to
  `ablation_heuristic_only.pth`, and the ±0.02 run-to-run noise floor measured
  in August were each computed on a *resampled* validation multiset, drawn
  differently per run. They are not wrong so much as not mutually comparable —
  a difference between two of them mixes model variance with sampling variance
  in unknown proportion. That bears directly on calling seed 3 the best and seed
  4 the worst, and on the ±0.02 noise floor, which was in part measuring this.

  **Re-scored figures on one common set are pending, and must run on the
  cluster.** `tools/evaluate_checkpoints.py` streams single-process and is
  unaffected, so scoring all six checkpoints on the full split's `val/` settles
  it without retraining anything. It cannot be done locally: the only local
  validation set belongs to [section 9](#9-flat-mlp-ablation)'s subset, which
  was split separately from the full corpus, so its val games may appear in the
  shipped and seed models' *training* games.

  For how large the effect can be, see [section 9](#9-flat-mlp-ablation): that
  ablation was first run at `--num-workers 4` and had to be discarded, because
  the three arms drew *different* shard multisets — the DataLoader seeds its
  workers from the main-process RNG after model construction, and the three
  architectures consume different amounts of it. That run put the
  DeepSets-versus-flat loss gap at 0.103; the clean re-run puts it at 0.054. The
  bug had inflated the apparent advantage roughly twofold, in the direction that
  flattered this paper's own architecture.
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
