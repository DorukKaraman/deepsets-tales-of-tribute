# DeepSetsBot and DeepSetsBlendBot

Two Monte Carlo tree search agents for *Tales of Tribute* whose state evaluation
comes from a DeepSets neural network instead of a hand-crafted heuristic. They
took **the top two places of eight entrants** in the 2026 Tales of Tribute AI
Competition (IEEE CoG): `DeepSetsBlendBot` first with **69.21%** and
`DeepSetsBot` second with **68.24%**, over 5460 official games each.

This repository accompanies the paper and contains everything needed to
reproduce that work: the agents, the trained model, the training pipeline, and
the verification and benchmarking harnesses. Start with
**[REPRODUCE.md](REPRODUCE.md)** for the end-to-end pipeline.

It is a fork of [ScriptsOfTribute-Core](https://github.com/ScriptsOfTribute/ScriptsOfTribute-Core),
the official competition engine. The upstream project's own README is preserved
at [docs/upstream-README.md](docs/upstream-README.md).

---

## The agents

Each visible card is encoded as a 99-dimensional vector and passed through a
shared per-card encoder; the results are mean-pooled, concatenated with a
19-dimensional encoding of non-card state, and passed through an MLP that
outputs a win probability. The network was trained by supervised learning on
game outcomes, using the competition's six-patron set (ANSEI, DUKE_OF_CROWS,
RAJHIN, ORGNUM, PELIN, SAINT_ALESSIA).

| Agent | Evaluation |
|---|---|
| `DeepSetsBot` | The network alone. |
| `DeepSetsBlendBot` | Blends a heuristic evaluation into the network's during the early game, decaying linearly to pure network by mid-game. |

The blend exists because the network's validation AUC is weakest early (0.797
in the lowest prestige-clock bucket, against 0.971 in the highest), which is
exactly where hand-tuned economy knowledge is most useful.

Patron selection is uniformly random in both agents.

## Results

**Official tournament (2026 Tales of Tribute AI Competition, 5460 games each):**

| Agent | Placing | Win rate |
|---|---|---|
| `DeepSetsBlendBot` | 1st of 8 | 69.21% |
| `DeepSetsBot` | 2nd of 8 | 68.24% |

Those percentages are win rates **across the whole tournament field**, not
against any single opponent. They are the paper's headline result, and nothing
in this repository reproduces them — the harnesses here reproduce our own local
measurements, which ran optimistic in every head-to-head we can compare (by
5.7 points against SakkirinaSolo and 18.5 against BestMCTS3).
[REPRODUCE.md](REPRODUCE.md#which-numbers-come-from-where) gives the full
comparison; please read it before citing any win rate from this repository.

## Attribution

**The search is not ours.** It derives from **SakkirinaSolo**, the 2025
competition winner: move generation, the tree search, tree reuse, the
rule-based fast paths, move deduplication, and the simulation policy that seeds
node priors are all unchanged from it. Our contribution is the evaluation
function and everything feeding it. Specifically, we replaced SakkirinaSolo's
static `Evaluate()` with the neural evaluator and fixed a null-return bug in
`Play()`; `DeepSetsBlendBot` additionally reuses SakkirinaSolo's own heuristic
evaluation as its early-game blend component.

The move-hashing utilities (`MoveComparer`) originate in **BestMCTS3** and
reached our code through SakkirinaSolo, which carries them with its author's
acknowledgement. We carry that acknowledgement forward.

Ours are:

- the DeepSets value network architecture;
- the 99-dimensional per-card and 19-dimensional global state encoding;
- the data-generation, training, export and cross-language verification
  pipeline that produced the shipped weights.

### Derived work versus redistribution

`Bots/src/DeepSetsBot.cs` and `DeepSetsBlendBot.cs` contain SakkirinaSolo's
search and are published here as derived works, attributed above and in
[docs/SUBMISSION.md](docs/SUBMISSION.md); that is the same form in which they
were submitted to the competition. Distributing an agent that is *substantially
unmodified* SakkirinaSolo under a new name would be republishing someone else's
competition entry rather than building on it, so we do not do that: the
baselines are fetched from the official archive rather than copied into this
repository, and the two agents of ours that are near-verbatim SakkirinaSolo
(`SakkirinaGen`, 39 of 885 lines changed; `SakkirinaHalf`, 36 of 872;
`SakkirinaScaled`, 35 of 1192) ship as patches against the author's own file
instead of as source.

## Third-party agents

`SakkirinaSolo` and `BestMCTS3` are other people's competition submissions. They
are **not** in this repository and are **not** covered by its licence; they
remain the work of their respective authors. To reproduce any baseline
comparison, fetch them from the official competition archive:

```bash
./scripts/fetch_baselines.sh
```

This clones the
[ScriptsOfTribute-CompetitionsArchive](https://github.com/ScriptsOfTribute/ScriptsOfTribute-CompetitionsArchive)
at a pinned commit, verifies each file against a recorded SHA-256, copies the
eight files into `Bots/src/`, and derives `SakkirinaGen.cs`,
`SakkirinaHalf.cs` and `SakkirinaScaled.cs` by applying `scripts/*.patch`. The
fetched and derived files are listed in `.gitignore`; do not commit them.

**Building and running `DeepSetsBot` / `DeepSetsBlendBot` does not require
this.** Only baseline comparisons do.

## Build

Requires the .NET 8 SDK.

```bash
dotnet build TalesOfTribute.sln -c Release
```

Run a game:

```bash
cd GameRunner/bin/Release/net8.0
./GameRunner DeepSetsBot DeepSetsBlendBot -n 1 -to 10 \
    -p ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA
```

Set `SOT_LOG=1` to have the agents write a log; the `PregamePrepare` line
records the resolved model path and its SHA-256.

### The OnnxRuntime native library caveat

**If the model fails to load, the agents do not crash — they silently fall back
to the heuristic evaluator and play substantially worse.** A run that looks
fine can therefore be measuring the wrong agent entirely. This is the single
most likely thing to go wrong, so it is worth understanding.

`GameRunner` loads `Bots.dll` dynamically by reflection rather than through a
compile-time project reference, so NuGet's normal dependency resolution never
places OnnxRuntime's assemblies next to the running executable. Three things
must reach `GameRunner`'s output directory:

- `Microsoft.ML.OnnxRuntime.dll`
- `Newtonsoft.Json.dll`
- `runtimes/<rid>/native/libonnxruntime.*`

`GameRunner.csproj` in this repository already arranges that by referencing the
`Microsoft.ML.OnnxRuntime` package directly (CPU inference only; no GPU or CUDA
execution provider is used). If you build the agents into a different host,
you must arrange it yourself.

### Where the model has to sit

The loader tries three candidates in order:

1. `AppContext.BaseDirectory/DeepSetsValueNetwork.onnx`
2. `./DeepSetsValueNetwork.onnx` (current working directory)
3. `../Bots/DeepSetsValueNetwork.onnx`

**It resolves on candidate 1, the primary path — not on a fallback.** We
verified this explicitly by running from a working directory where neither
candidate 2 nor 3 exists; the model still loaded. This matters because
`models/` is not itself on that list: the model reaches the agent because
`GameRunner.csproj` copies `models/DeepSetsValueNetwork.onnx` into
`GameRunner`'s output directory at build time, and for a bot loaded
dynamically into `GameRunner`, `AppContext.BaseDirectory` is **the directory
the `GameRunner` executable itself sits in** — not the directory `Bots.dll`
was loaded from.

So the layout does not depend on the working directory you launch from, and
does not depend on the fallbacks. If you build the agents into a different
host, put the model next to that host's executable.

To confirm the model loaded rather than silently falling back, run with
`SOT_LOG=1` and check the `PregamePrepare` line reports
`sha256=86e0f9a8...`.

## Model artefacts

`models/SHA256SUMS` records all three; verify with `shasum -a 256 -c SHA256SUMS`.

| File | SHA-256 | What it is |
|---|---|---|
| `DeepSetsValueNetwork.onnx` | `86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915` | The shipped model. This exact file produced the tournament result. |
| `deepsets_value_network.pth` | `3bd14d4e069d47267721cc4765d1d6e94fe5c1b702fa552fe3c445a175f7b66b` | The PyTorch checkpoint the ONNX was exported from. All 12 weight tensors are bit-identical to the ONNX's initializers. |
| `ablation_heuristic_only.pth` | `0fff37d6c89cc74e86f1c8b2b314aca31752295c66daacdd217f351257568774` | **Not the submitted model.** Trained on heuristic-generated data alone; one arm of an ablation the paper reports. |
| `ablation_heuristic_only_metrics.json` | see `SHA256SUMS` | Training curves for the checkpoint above — 3 epochs, best val accuracy 0.8050, best val loss 0.4034. |

No training metrics survive for the shipped model; the figures above belong to
the ablation arm and should not be attributed to it. See
[REPRODUCE.md](REPRODUCE.md#3-train).

The ONNX file reports `producer: pytorch 2.2.2`. Reproducing its *byte* hash
requires that version; see
[REPRODUCE.md](REPRODUCE.md#reproducing-the-model-file-byte-for-byte).

## Layout

Ours:

| Path | Contents |
|---|---|
| `Bots/src/DeepSets*.cs` | The two agents and their shared infrastructure (`DeepSetsCore.cs`: feature extraction, ONNX inference, card metadata). |
| `Bots/src/SakkirinaGenNeural.cs` | Self-play data-generation agent. |
| `models/` | The trained model, its checkpoint, the ablation checkpoint, and their hashes. |
| `training/` | Feature schema (`StateParser.py`), network definition, training, ONNX export, card-database generation and audit. |
| `tools/` | Data generation, dataset splitting, cross-language parity verification, benchmarking. |
| `experiments/` | Control conditions, null results, the paper's experiment agent, and the experiment configs. See [experiments/README.md](experiments/README.md). |
| `scripts/` | Baseline fetching, the three SakkirinaSolo patches, the pinned Python environment, and SLURM templates. |

Upstream (unmodified unless noted): `Engine/`, `gRPC/`, `Tests/`,
`ModuleTests/`, `BotsTests/`, `Bots/src/` (all other bots).
`GameRunner/` is upstream plus our `--log-training-data` / `--data-dir` options,
the OnnxRuntime wiring described above, and a one-line-per-game
`GAME_END_REASON:` print so a benchmark can tell a timeout apart from an illegal
move (the engine's own stats counter pools both as "other factors").
`Engine/src/utils/GameEndStatsCounter.cs` additionally counts
`PREPARE_TIME_EXCEEDED`, which it previously threw on.

### The dual-language feature schema

The node and global feature vectors are defined **twice** — in
`training/StateParser.py` for training, and in the `FeatureExtractor` class in
`Bots/src/DeepSetsCore.cs` for inference. They must agree exactly or the agent
plays on features the network was never trained on. `tools/verify_parity.py`
checks this by running both real implementations against the same logged states
and diffing the outputs column by column. Any change to either file must be
mirrored in the other, and the parity check must pass.

## Licence

This repository is MIT licensed (see [LICENSE](LICENSE)), inherited from
upstream ScriptsOfTribute-Core, © 2022 Ematerasu. The licence covers upstream's
code and ours. It does **not** cover the third-party competition agents
described under [Third-party agents](#third-party-agents), which are not
distributed here.
