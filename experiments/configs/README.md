# Experiment configs

One JSON file per experiment. `tools/benchmark_cluster.py` reads it and turns it
into a SLURM array: **one game per array task, one task per core**, exactly as
the original hardcoded benchmark did.

| Config | What it asks | Matchups × games | Tasks |
|---|---|---|---|
| [`alpha_sweep.json`](alpha_sweep.json) | How much of the advantage is the blend, and how much is the network alone? | 5 × 400 | 2000 |
| [`time_scaling.json`](time_scaling.json) | Does the advantage grow, shrink or hold as the per-turn budget moves from 2 s to 30 s? | 5 × 400 | 2000 |
| [`equal_effort.json`](equal_effort.json) | Is the network better, or is the heuristic agent just searching more? Treatment sped up; landed at 0.88 of the baseline's effort (43,139 vs 49,131 evals/turn). | 1 × 400 | 400 |
| [`equal_effort_baseline_slowed.json`](equal_effort_baseline_slowed.json) | The same question, baseline slowed down instead. Cheaper; matched to within 3 % (6,433 vs 6,627 evals/turn). | 1 × 400 | 400 |
| [`seed_benchmark.json`](seed_benchmark.json) | How much of the win rate is the training seed? Five per-seed models plus the shipped one, same games. **Needs two path substitutions first.** | 6 × 400 | 2400 |
| [`ablation_benchmark.json`](ablation_benchmark.json) | Does the DeepSets structure win *games*, or only validation loss? The three flat-MLP ablation arms (REPRODUCE.md §9). **Needs one path substitution first.** | 3 × 400 | 1200 |
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
  "allowed_onnx_sha256": ["86e0..."],  // models a ROW may load via SOT_MODEL_PATH
  "builtin_onnx_sha256": "86e0...",    // GameRunner's own copy; defaults to the shipped model
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

**`"TBD"` is a refusal, not a default.** A config can carry the literal string
`TBD` for a `timeout` or an `env` value it cannot know until a calibration run
has happened, and any task in such a matchup refuses to run. It would otherwise
fall back to the bot's own default and produce 400 games labelled as an
experiment that was never performed — indistinguishable, afterwards, from a real
result. `--dry-run` still prints the plan and `--calibrate` still runs, because
those are how the value gets filled in. Both `equal_effort*.json` configs were
built this way and are now filled in; nothing currently ships with a
placeholder.

**The ONNX pin has three layers, and the guarantee is per row.** A bot whose
model fails to load does not crash — it logs the failure, falls back, and plays
a complete, plausible-looking game against the wrong evaluator. None of these
is skippable.

| layer | what it checks | where |
|---|---|---|
| `builtin_onnx_sha256` | GameRunner's own copy of the model — a **stale-build** check | `benchmark_cluster.sh`, before anything runs |
| `allowed_onnx_sha256` | the file a row's `SOT_MODEL_PATH` names is one the config knows | `benchmark_cluster.py`, before each game |
| loaded-model check | the sha256 the bot **logged having loaded** equals that row's own expected model | `benchmark_cluster.py`, after each game |

`allowed_onnx_sha256` takes a list because per-seed training
(`scripts/slurm_train.sh`) means there is no longer exactly one legitimate
model file.

**The first two are separate fields on purpose.** They used to be one: the
built-in copy was checked against `allowed_onnx_sha256`, which forced every
config to whitelist the shipped hash even when no row should ever load it — and
a whitelisted shipped hash means a row that silently fell back to the built-in
model *passes the pin*. For a control row running the same architecture at the
same speed, nothing else would have caught it.

**The third is what makes it per row.** The first two are set checks on files:
they can confirm a task pointed at one of the config's known models, never that
seed 3's row ran seed 3's model. Only the bot knows that, and it says so in its
log. A mismatch, or no such line at all, fails the task and writes no result;
the bot log is kept for diagnosis instead of being deleted. Each result JSON
records a `loaded_models` block — path, size and sha256 per bot — so a finished
run is auditable row by row.

Because that check reads the bot log, a config that sets `SOT_MODEL_PATH`
anywhere **must** have `bot_log` set to `parse` or `keep`; the loader refuses
the combination rather than skipping the check quietly. A config with no
overrides at all (`legacy_paper_benchmark.json`) may still run with the log
off, since every bot then loads the built-in copy that layer 1 already pinned.

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

The two `equal_effort*` configs target the same ratio and move opposite sides to
reach it. Both were run, because neither is decisive alone.

One calibration serves both: **n = 60 games gave a ratio of means r = 7.08
(roughly 6.4–7.8)**. The harness prints *r* as a suggested `SOT_TIME_SCALE`, so
the baseline-slowed config takes its **reciprocal**.

| | setting | keeps fixed | achieved over 400 games | distortion | cost |
|---|---|---|---|---|---|
| `equal_effort_baseline_slowed` | `SOT_BASELINE_TIME_SCALE = 0.141` (= 1/*r*), timeout 12 | treatment at stock timing | **6,433 vs 6,627** evals/turn — within 3 % | baseline runs at ~1/7 of the budget it was tuned for, and a hand-tuned heuristic may degrade non-linearly | a normal row |
| `equal_effort` | `SOT_TIME_SCALE = 9.0`, timeout 91 | baseline at stock timing | **43,139 vs 49,131** evals/turn — DeepSets at **0.88** of the baseline's effort | the treatment gets an 88.2 s/turn budget no tournament would give it | ~120 core-hours / 400 games |

Figures are `DeepSetsBotExp` vs `SakkirinaScaled`. **Quote the achieved numbers,
not *r*** — only the baseline-slowed direction is matched, and the sped-up one
is not: it left DeepSets doing 12 % *less* work than the baseline.

**`SOT_TIME_SCALE = 9.0` is not *r*.** Evaluations per turn is only
approximately linear in the time budget — tree reuse and the rule-based fast
paths do not scale with the clock — and a pilot at *r* = 7.08 undershot, so 9.0
was chosen empirically from that pilot rather than derived.

Both directions ended with DeepSets doing slightly less work than the baseline
(0.97× and 0.88×), so in both the residual mismatch runs **against** the DeepSets
agent. That is the conservative direction for the claim: a win under these
conditions is not explained by the DeepSets side having been handed more search.

What carries weight is **agreement between them**. Same direction in both means
the conclusion survives whichever side was moved. Disagreement is itself the
finding: evaluations per turn would not be the right currency for "effort" in
this matchup, and the framing would need rethinking before either number is
reported. Lead with the baseline-slowed direction — it is the cheaper and the
better-matched of the two — and let the sped-up one confirm it.

## Calibration

```bash
tools/benchmark_cluster.sh --config equal_effort --out-dir "$OUT_DIR" --calibrate
```

Runs the config's `calibration` matchups sequentially (20 games by default) and
reports mean **evaluations per turn** per agent, plus a suggested
`SOT_TIME_SCALE` for equal effort.

Calibrate at `SOT_ALPHA0=0`, which both equal-effort configs pin — **the
experiment is about the network as an evaluator, so calibrate on the same agent
you will benchmark.** Above 0 one counted evaluation also runs the heuristic, so
the counter measures the same event on each side but not quite the same work;
that effect is small, though — paired over 6 seeds, throughput moved 1.6 %
between `alpha0=0.0` and `alpha0=0.9`. (An earlier note here put it at a factor
of 1.8, comparing an n=2 run against an n=10 one; that was sampling noise. See
[experiments/README.md](../README.md#counting-evaluations).)

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
