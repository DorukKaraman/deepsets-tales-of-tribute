# Experiments: controls and null results

**Nothing in this directory is a candidate agent.** Each item here exists to
rule something out. They are kept because the paper reports the results, and a
control whose code is not available is not much of a control.

The `.cs` files in `bots/` are compiled into `Bots.dll` by an explicit
`<Compile Include>` in `Bots/Bots.csproj`, so `GameRunner` can load them by
name like any other bot.

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

## The dead-code equivalence null result

`bots/DeepSetsBotTrim.cs`, `bots/DeepSetsBlendBotTrim.cs`

Variants of the two agents with genuinely dead code removed: an unused field
(`anyInvalidMoves`), an unused `DeckValue` parameter, an unused
`neutralPatronFavour` local, an unused `OutputState` method, and an unused
`Compare` override — 41 lines across each file, none of them reachable.

**The result is a null result, and that is the point.** They were benchmarked
head-to-head against their un-trimmed originals (see `MATCHUPS` in
`tools/benchmark_cluster.py`, which retains both self-play matchups) and
measured as equivalent. The trimming changes nothing about play; it confirms
the dead code was in fact dead.

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
