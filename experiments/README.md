# Experiments: controls and null results

**Nothing in this directory is a candidate agent.** Each item here exists to
rule something out. They are kept because the paper reports the results, and a
control whose code is not available is not much of a control.

The `.cs` files in `bots/` are compiled into `Bots.dll` by an explicit
`<Compile Include>` in `Bots/Bots.csproj`, so `GameRunner` can load them by
name like any other bot.

The submitted agents — `Bots/src/DeepSetsBot.cs`,
`Bots/src/DeepSetsBlendBot.cs`, `Bots/src/DeepSetsCore.cs` — are frozen
byte-identical to what played the tournament. That is the constraint everything
here is shaped around: nothing in this directory modifies them, and every
experimental knob is an environment variable read by a *copy*, so one build
serves every configuration.

---

## The experiment agent

`bots/DeepSetsBotExp.cs`

A verbatim copy of `DeepSetsBlendBot` with three environment hooks and nothing
else changed. With none of them set it is the submission.

| Variable | Default | What it does |
|---|---|---|
| `SOT_ALPHA0` | `0.7` | Initial blend coefficient. `0.0` makes `Evaluate()` take its pure-network short circuit on every leaf — i.e. **`alpha0 = 0` *is* `DeepSetsBot`**, so one class covers both submitted agents. |
| `SOT_TIME_SCALE` | `1.0` | Multiplies **both** time constants: the 9.8 s `TurnTimeout` and the 0.65 s per-move cap. |
| `SOT_MODEL_PATH` | unset | ONNX model to load; falls back to the normal resolution. |

**Scale both time constants, never one.** `scripts/sakkirina_half.patch`
measured what happens otherwise: scaling only the 0.65 s per-call cap and
leaving `TurnTimeout` at 9.8 s produced ~73 % of stock's eval rate where ~45 %
was intended, because cheaper calls simply meant more of them fit under the
same unscaled per-turn ceiling.

A set-but-unparseable or out-of-range value is **rejected and logged**, not
silently replaced by the default. Values are parsed with `InvariantCulture`, so
`0.5` means the same thing on a German-locale login node.

It also logs evaluations per turn, which is what makes equal-effort matching
measurable — see [`configs/equal_effort.json`](configs/equal_effort.json) and
`tools/benchmark_cluster.py --calibrate`.

### Is it really the same agent?

```bash
python experiments/verify_exp_bot_parity.py              # Exp vs DeepSetsBlendBot
python experiments/verify_exp_bot_parity.py --self-check # the timing-noise baseline
```

Both agents play the same seed against the same opponent and their move
sequences are compared. **Read the divergence index comparatively, not
absolutely.** The search is wall-clock budgeted (`while (s.Elapsed <
timeForMoveComputation)`), so how many iterations fit in 0.65 s depends on
machine load, JIT warm-up and GC — two runs of the *same binary* on the same
seed can pick different moves and then drift apart permanently. `--self-check`
runs `DeepSetsBlendBot` against itself and shows what that noise alone
produces. A divergence no earlier than the self-check baseline is consistent
with the two agents being identical; a reproducible divergence at move 0 or 1
is a real signal.

## The scaled baseline

`SakkirinaScaled` — not a file here; it ships as
[`scripts/sakkirina_scaled.patch`](../scripts/sakkirina_scaled.patch) and is
produced by `scripts/fetch_baselines.sh`, for the same reason `SakkirinaHalf`
does: it is ~97 % verbatim SakkirinaSolo and redistributing it as source would
be republishing someone else's competition entry.

`SakkirinaSolo` with its two time constants multiplied by
`SOT_BASELINE_TIME_SCALE` (default `1.0`, which is timing-identical to stock),
plus an evaluations-per-turn counter. Two things need it:

- **Time scaling** needs a baseline whose budget moves *with* the treatment's.
  Holding it at 9.8 s while the treatment varies would measure the gap between
  two budgets, not the effect of the budget.
- **Equal effort** needs the stock baseline's evaluations per turn, and stock
  `SakkirinaSolo` emits no throughput telemetry at all. At scale `1.0` this
  agent is stock, and it counts.

## The experiment configs

[`configs/`](configs/) — see [`configs/README.md`](configs/README.md) for the
format and the exact cluster commands. In short: the benchmark harness takes a
JSON file listing matchups, each with its own engine `--timeout`, game count
and environment dict, so one build and one harness cover the alpha sweep, the
time-scaling study and the equal-effort control.

---

## The search-volume control

`SakkirinaHalf` — not a file here; it ships as
[`scripts/sakkirina_half.patch`](../scripts/sakkirina_half.patch) and is
produced by `scripts/fetch_baselines.sh`, because it is 96% verbatim
SakkirinaSolo and redistributing it as source would be republishing someone
else's competition entry (see the README's
[derived work versus redistribution](../README.md#derived-work-versus-redistribution)).

**The confound.** `DeepSetsBot` evaluates far fewer positions per game than
`SakkirinaSolo` does — roughly 102,700 against 226,100 — because a neural
forward pass costs more than a hand-written heuristic. So a sceptic can
reasonably ask whether the win rate reflects a better evaluation function or
merely two agents searching different volumes.

**The control.** `SakkirinaHalf` is `SakkirinaSolo` with its per-move time
budget scaled by 0.45 and nothing else changed. It reproduces the throughput
deficit without also swapping the evaluator, which isolates search volume from
evaluation quality.

> **⚠ The 102,700 / 226,100 figures did not reproduce, and `HALF_STRENGTH_FACTOR
> = 0.45` is derived from them.** 0.45 is exactly 102,734 / 226,057. Re-measured
> on 2026-09-22 with the counters verified equivalent (see
> [Counting evaluations](#counting-evaluations) below):
>
> | | historical | re-measured | |
> |---|---|---|---|
> | `DeepSetsBot` (frozen), evals/game | 102,734 | 112,725 | **1.10x** |
> | stock-timing baseline, evals/game | 226,057 | 635,910 | **2.81x** |
> | ratio | 2.20x | 5.64x | |
>
> Pooled over 6 games, `DeepSetsBot` vs `SakkirinaScaled` at
> `SOT_BASELINE_TIME_SCALE=1.0`, `--timeout 10`. **The neural side reproduces;
> the heuristic baseline does not.** The cause is not identified — the repository
> is a single squashed commit, so the measurement conditions behind the
> historical figures (hardware, ONNX Runtime version, engine `--timeout`) cannot
> be recovered, and the agent they were taken from (`SakkirinaNeural`) is no
> longer in the tree.
>
> If the deficit is really ~1/5.6 rather than ~1/2.2, then 0.45 does not
> reproduce it and `SakkirinaHalf` is calibrated to the wrong point (~0.18 would
> be the equivalent). **Re-measure on the cluster before reporting anything that
> depends on 0.45.** The patch is deliberately left untouched pending that
> measurement, so the control keeps producing the same agent it always did.
>
> Per-game variance is large — the ratio ranged 3.28x to 19.64x across those 6
> games — so use a pooled figure over ≥20 games, not a mean of per-game ratios.

## Counting evaluations

`DeepSetsBotExp` and `SakkirinaScaled` both report evaluations per turn, and
equal-effort matching depends on the two counters meaning the same thing. They
do, verified structurally rather than assumed:

| | `experiments/bots/DeepSetsBotExp.cs` | `Bots/src/SakkirinaScaled.cs` |
|---|---|---|
| counter increment | line 565 | line 333 |
| position in `Evaluate()` | **first statement**, before any branch or early return | **first statement**, before any branch or early return |
| `Evaluate()` definition | line 563 | line 331 |
| call sites of `Evaluate()` | exactly one: line 992 | exactly one: line 848 |
| that call site | `return Evaluate(gameState, myPlayerID);`, the last statement of `Simulate()` | identical |
| `Simulate` / `MoveSimulate` / `TreeSearch` | identical to SakkirinaScaled's modulo one trailing comment | — |

The frozen `DeepSetsBot` (line 379) and `DeepSetsBlendBot` increment in the same
place, so current numbers are comparable with anything those agents logged.

**One counted evaluation is not always the same amount of work**, but the
difference is small. At `SOT_ALPHA0 > 0`, `Evaluate()` runs *both* the heuristic
and the network inside a single increment for every leaf in the blend window,
while the baseline's increment is always one heuristic pass. The blend window is
not rare — at `alpha0=0.9`, 48 of 76 turns (63 %) had `resolvedAlpha > 0` — but
the extra pass is nearly free, because the heuristic costs a fraction of a
network forward pass plus its feature extraction. Measured paired over 6 seeds,
throughput went **4,501 → 4,574 evaluations per thinking-second** from
`alpha0=0.0` to `alpha0=0.9`: a 1.6 % change, inside the noise.

Both `equal_effort*.json` configs still pin `SOT_ALPHA0=0`, for the
straightforward reason rather than a throughput one: **the experiment is about
the network as an evaluator, so the agent measured should be the network-only
one, and the calibration must be run on the same agent that will be
benchmarked.**

### What the equal-effort runs actually achieved

Calibration was n = 60 games, ratio of means *r* = 7.08 (roughly 6.4–7.8). Both
directions were then run to 400 games. Figures are `DeepSetsBotExp` vs
`SakkirinaScaled`, evaluations per turn:

| Direction | Setting | Achieved |
|---|---|---|
| baseline slowed | `SOT_BASELINE_TIME_SCALE = 0.141` (= 1/*r*), timeout 12 | **6,433 vs 6,627** — within 3 % |
| treatment sped up | `SOT_TIME_SCALE = 9.0`, timeout 91 | **43,139 vs 49,131** — DeepSets at **0.88** of the baseline's effort |

Only the first is matched. The second is not, and should not be described as
such: it left DeepSets doing 12 % *less* work than the baseline. `9.0` is not
*r* — evaluations per turn is only approximately linear in the time budget, a
pilot at *r* = 7.08 undershot, and 9.0 was chosen empirically from that pilot.

Both directions ended with DeepSets doing slightly less work than the baseline
(0.97× and 0.88×), so in both the residual mismatch runs **against** the DeepSets
agent — the conservative direction for the claim.

> **Correction (2026-09-23).** An earlier version of this section claimed the
> blend inflated the measured ratio by a factor of 1.8, citing 14.95x at
> `alpha0=0.7` against 8.11x at `alpha0=0`. That was wrong. Those two figures
> came from different runs — n=2 and n=10 — drawn from a distribution whose
> per-game ratio spans 2.71x to 21.07x, so the gap between them was sampling
> noise attributed to a cause. The measurement above isolates `alpha0` properly,
> paired on identical seeds, and finds a ~1.6 % throughput effect.

## Why evaluations per turn rises with alpha0

The counter increments once per leaf regardless of `alpha0` (see the table
above), so a rising evals/turn column is not a counting artefact — the agent
really is evaluating more leaves per turn. The quantity decomposes as

```
evaluations per turn  =  thinking seconds per turn  x  evaluations per thinking second
```

and paired over 6 seeds, `alpha0=0.0` against `alpha0=0.9`:

| | alpha0=0.0 | alpha0=0.9 | ratio |
|---|---|---|---|
| evaluations per turn | 9,934 | 11,124 | 1.12x |
| thinking seconds per turn | 2.10 | 2.38 | **1.13x** |
| evaluations per thinking second | 4,501 | 4,574 | 1.02x |
| turns per game | 11.00 | 12.67 | 1.15x |

**All of the movement is in thinking seconds per turn; throughput is flat.**
Thinking time is spent per *move*, not per turn: each move that reaches the
search gets `min(0.65s, remaining/4)` out of a 9.8 s per-turn budget, while
moves resolved by the single-legal-move short circuit or by `RootRuleBasedMove`
cost nothing at all. So a policy that plays longer, more decision-dense turns
spends more of that budget and evaluates more leaves per turn. The column is
measuring a real behavioural difference between the two evaluators' play, not an
accounting quirk.

Caveat on the local measurement: at n=6 the per-seed evals/turn ratios were
0.45, 4.47, 0.70, 0.71, 1.82, 1.98 — three down, three up. It confirms the
mechanism and rules out a counting artefact; it does not on its own establish
the monotone rise. That comes from the 400-game cluster runs.

**The neural agent is not ONNX-bound.** The frozen `DeepSetsBot`'s own
telemetry puts only **13–31 %** of wall clock inside
`EvaluateBoardState` (`fractionOfWallClockInEvaluateBoardState`, 3 games). Most
of the per-evaluation cost is feature extraction and tensor marshalling around
the inference call, not the inference itself — which is worth knowing before
anyone tries to close the throughput gap by optimising the model.

## The dead-code equivalence null result

`bots/DeepSetsBotTrim.cs`, `bots/DeepSetsBlendBotTrim.cs`

Variants of the two agents with genuinely dead code removed: an unused field
(`anyInvalidMoves`), an unused `DeckValue` parameter, an unused
`neutralPatronFavour` local, an unused `OutputState` method, and an unused
`Compare` override — 41 lines across each file, none of them reachable.

**The result is a null result, and that is the point.** They were benchmarked
head-to-head against their un-trimmed originals (see
[`configs/legacy_paper_benchmark.json`](configs/legacy_paper_benchmark.json),
which retains both self-play matchups) and measured as equivalent. The trimming
changes nothing about play; it confirms the dead code was in fact dead.

Do not treat these as improved versions of the agents. The submitted,
tournament-playing agents are `Bots/src/DeepSetsBot.cs` and
`Bots/src/DeepSetsBlendBot.cs`.

## The C#-inference verification chain

`bots/FeatureDumper.cs`, `verify_csharp_inference.py`,
`compare_ingame_vs_val.py`

Establishes that the C# inference path matches the PyTorch reference **on real
in-game tensors**, not synthetic or randomly generated ones.

`FeatureDumper` is opt-in instrumentation: it samples `NeuralEvaluate`'s actual
inputs and outputs to JSONL when `SOT_DUMP_DIR` is set, and costs nothing when
it is not. Each row is the exact node-feature matrix and global vector the ONNX
model was fed for one real evaluation.

```bash
# 1. produce dumps from real games
SOT_DUMP_DIR=/tmp/dumps ./GameRunner DeepSetsBot SakkirinaSolo -n 20 -to 10

# 2. replay them through the PyTorch checkpoint and compare
python experiments/verify_csharp_inference.py --dump-path /tmp/dumps

# 3. compare the in-game feature distribution against validation
python experiments/compare_ingame_vs_val.py --dump-path /tmp/dumps
```

`verify_csharp_inference.py` is the decisive check: if it fails, there is a
real C#-side feature-extraction or inference bug. `compare_ingame_vs_val.py`
answers a different question — whether the states the agent actually
encounters in play are distributed like the ones the network was trained on, or
whether it is being asked to extrapolate.

`tools/ParityCheck` can also emit this JSONL format directly (`infer` mode),
which replaces the live-game sampler with a deterministic pass over chosen
states — better coverage, same comparison.

These two scripts import from `tools/diagnose_value_net.py`, so run them from
the repository root.

## Superseded: pre-schema-v2 data verification

`verify_training_data.py`

Targets the **pre-schema-v2** data format — the single
`Train_/Val_Sakkirina.jsonl.gz` pair with 101-dimensional node and
17-dimensional global features, before the schema rewrite to 99/19. It does not
run against current generated data; `tools/verify_generated_data.py` is the
current equivalent.

Kept because it is the check that was actually run against the older dataset,
and the older dataset is what the earliest reported numbers came from.
