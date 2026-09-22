# Experiment configs

One JSON file per experiment. `tools/benchmark_cluster.py` reads it and turns it
into a SLURM array: **one game per array task, one task per core**, exactly as
the original hardcoded benchmark did.

| Config | What it asks | Matchups × games | Tasks |
|---|---|---|---|
| [`alpha_sweep.json`](alpha_sweep.json) | How much of the advantage is the blend, and how much is the network alone? | 5 × 400 | 2000 |
| [`time_scaling.json`](time_scaling.json) | Does the advantage grow, shrink or hold as the per-turn budget moves from 2 s to 30 s? | 5 × 400 | 2000 |
| [`equal_effort.json`](equal_effort.json) | Is the network better, or is the heuristic agent just searching more? Treatment sped up. | 1 × 400 | 400 |
| [`equal_effort_baseline_slowed.json`](equal_effort_baseline_slowed.json) | The same question, baseline slowed down instead. Cheap; run it first. | 1 × 400 | 400 |
| [`legacy_paper_benchmark.json`](legacy_paper_benchmark.json) | The original 10-matchup benchmark, reproduced exactly. | 10 × 400 | 4000 |

`legacy_paper_benchmark.json` is the previously-hardcoded list: same order, same
seeds, same task ids, same result directory names, so an in-flight run of the
old harness resumes across the config rework. Do not reorder or extend it.

## The format

```jsonc
{
  "name": "alpha_sweep",
  "description": "printed in the plan and in the aggregate",
  "seed_base": 20260922,               // seed = seed_base + task_id; --seed-base overrides
  "patrons": "ANSEI,DUKE_OF_CROWS,...",
  "allowed_onnx_sha256": ["86e0..."],  // replaces the default list wholesale
  "bot_log": "parse",                  // parse | keep | off
  "defaults": { "games": 400, "timeout": 10, "env": {} },
  "matchups": [
    { "label": "alpha0_0.0",           // unique; becomes the result directory name
      "bot_a": "DeepSetsBotExp",
      "bot_b": "SakkirinaSolo",
      "games": 400,                    // falls back to defaults.games
      "timeout": 10,                   // engine --timeout, seconds
      "env": { "SOT_ALPHA0": "0.0" },  // merged over defaults.env, applied to that task's process
      "optional": false,
      "note": "free text, printed in the plan" }
  ],
  "calibration": { "games": 20, "matchups": [ /* ... */ ] }   // optional; see --calibrate
}
```

**Task-id layout.** Matchups are laid out contiguously in config order, each
occupying its own `games` tasks. Offsets are cumulative sums, so **adding a
matchup anywhere but the end renumbers every task after it** and invalidates an
in-flight run's result files. Within a matchup the first half runs `bot_a` as
P1 and the second half swaps seats.

**`env` is per matchup, applied to that task's process.** Both agents run in
the same process, which is why `time_scaling.json` can set `SOT_TIME_SCALE` and
`SOT_BASELINE_TIME_SCALE` together and have them reach opposite sides of the
board. Every `SOT_*` variable the harness manages is stripped from the ambient
environment first, so an exported `SOT_ALPHA0` in the submitting shell cannot
leak into a run that never asked for it.

**`"TBD"` is a refusal, not a default.** Both `equal_effort*.json` configs ship
with placeholders they cannot know until a calibration run has happened. Any task in a
matchup that still carries one refuses to run. It would otherwise fall back to
the bot's default and produce 400 games labelled as an experiment that was
never performed — which, afterwards, is indistinguishable from a real result.
`--dry-run` still prints the plan and `--calibrate` still runs, because those
are how you get the value that fills it in.

**The ONNX pin takes a list.** Per-seed training (`scripts/slurm_train.sh`)
means there is no longer exactly one legitimate model file. `tools/benchmark_cluster.sh`
checks GameRunner's own model copy against `allowed_onnx_sha256`;
`tools/benchmark_cluster.py` additionally checks any `SOT_MODEL_PATH` a matchup
sets, which is the only place that override *can* be checked. Neither is
skippable: a bot whose model fails to load does not crash, it silently falls
back to a heuristic evaluator.

## Running one

Always print the plan first — it gives each matchup's exact task-id range:

```bash
tools/benchmark_cluster.sh --config alpha_sweep --out-dir "$OUT_DIR" --dry-run
```

Then submit, and aggregate with the **same** `--config`:

```bash
sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=0-1999%32 scripts/slurm_experiment.sh
python tools/aggregate_benchmark_results.py --config alpha_sweep --out-dir "$OUT_DIR/alpha_sweep"
```

Resubmitting is always safe: a task whose result file exists is skipped without
running anything.

## Equal effort, in both directions

The two `equal_effort*` configs target the same matched ratio and move opposite
sides to reach it. Both are run, because neither is decisive alone:

| | moves | keeps fixed | distortion | cost |
|---|---|---|---|---|
| `equal_effort` | `SOT_TIME_SCALE = r` | baseline at stock timing | the treatment gets a budget (~78 s/turn) no tournament would give it | ~120 core-hours / 400 games |
| `equal_effort_baseline_slowed` | `SOT_BASELINE_TIME_SCALE = 1/r` | treatment at stock timing | the baseline runs at ~1/8 of the budget it was tuned for, and a hand-tuned heuristic may degrade non-linearly | a normal row |

What carries weight is **agreement between them**. Same direction in both means
the conclusion survives whichever side was moved. Disagreement is itself the
finding: evaluations per turn would not be the right currency for "effort" in
this matchup, and the framing would need rethinking before either number is
reported.

**One calibration serves both.** It measures *r*; the configs apply it to
opposite sides. The harness prints *r* as the suggestion, so take its
**reciprocal** for the baseline-slowed config.

## Calibration

```bash
tools/benchmark_cluster.sh --config equal_effort --out-dir "$OUT_DIR" --calibrate
```

Runs the config's `calibration` matchups sequentially (20 games by default) and
reports mean **evaluations per turn** per agent, plus a suggested
`SOT_TIME_SCALE` for equal effort.

Calibrate at `SOT_ALPHA0=0`, which both equal-effort configs pin. Above 0, one
counted evaluation runs *both* evaluators inside the blend window, so the
counter measures the same event on each side but not the same work — measured at
14.95× per turn at `alpha0=0.7` against 8.11× at `alpha0=0`.

Per *turn*, not per second or per game: evals/sec is swamped by opponent
thinking time and evals/game by game length, and both of those move when the
time budget moves — which is the variable under study. Two agents in one game
play the same number of turns ±1, so per-turn is what is comparable between
them.

The suggestion assumes evals/turn is linear in the budget, which is
approximately but not exactly true (tree reuse and the rule-based fast paths do
not scale with the clock). **Re-run `--calibrate` with the suggested value to
confirm the two figures actually meet** before spending 400 games on it.

## Reading the aggregate

Two columns decide whether a row is a measurement of anything:

- **timeouts** — the clock ran out. A statement about the `--timeout` margin,
  not about play. `time_scaling.json` sets engine `--timeout` to budget + 2 s
  precisely so this stays near zero; if it does not, raise the margin and re-run
  the row rather than reporting its win rate.
- **disqualifications** — an illegal move, an exception, or a failed patron
  selection. A real defect, and a tight budget does not excuse it.

Both are attributed to the side that caused them (the engine awards the win to
the opponent, so the offender is the loser), and both are excluded from the
clean win rate.

The **swapped / not-swapped** rows are the check for a seat-swap inversion bug.
Both moving in the same direction, wide of 50 %, is healthy. Both sitting near
50 % independently, or one at 90 % and the other at 10 %, is the bug.

## A caveat about the sweep

`seed = seed_base + task_id`, and task ids are contiguous across matchups, so
**the five alpha conditions play different games from each other.** They are
independent 400-game samples, not a paired design. Differences between adjacent
alpha values that are small relative to the Wilson intervals should be read as
such; the intervals are what the aggregator prints for exactly this reason.
