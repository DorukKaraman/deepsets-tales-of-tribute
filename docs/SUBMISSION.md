# DeepSetsBot / DeepSetsBlendBot — IEEE CoG 2026 Tales of Tribute submission

## What they are

Both are Monte Carlo tree search agents whose state evaluation comes from a
DeepSets neural network instead of a hand-crafted heuristic. Each visible
card is encoded as a 99-dimensional vector and passed through a shared
per-card encoder, the results are mean-pooled, concatenated with a 19-dimensional
encoding of non-card state, and passed through a MLP that outputs a
win probability. The network was trained by supervised learning on outcomes
from roughly 12,000 self-play games,
using the competition's six-patron set.

- `DeepSetsBot` uses the network alone.
- `DeepSetsBlendBot` additionally blends in a heuristic evaluation during the
  early game, linearly decaying to zero (pure network) by mid-game.

Both bots are in `namespace Bots`.
Patron selection (`SelectPatron`) is uniformly random in both bots.

## Attribution

The search itself is derived from **SakkirinaSolo, the 2025 competition
winner**. I replaced its static `Evaluate()` function with the neural
evaluator and fixed a null-return bug in `Play()`. Everything else in the
search (tree reuse, move selection, rule-based fast paths, etc.) is
unchanged from SakkirinaSolo. `DeepSetsBlendBot` additionally reuses
SakkirinaSolo's own heuristic evaluation function for its early-game blend
component.

My own contributions are:
- The DeepSets value network architecture.
- The 99-dimensional per-card / 19-dimensional global state encoding it
  consumes.
- The data-generation and training pipeline that produced the shipped
  model weights.

## Files in this submission

This package ships as six flat files, not a directory tree. In a checkout of
the competition template, the five build files go here:

```
ScriptsOfTribute-Core/
└── Bots/
    ├── Bots.csproj                  (replaces the existing file)
    ├── DeepSetsValueNetwork.onnx
    └── src/
        ├── DeepSetsBot.cs
        ├── DeepSetsBlendBot.cs
        └── DeepSetsCore.cs
```

File by file:

- `DeepSetsBot.cs` → `Bots/src/DeepSetsBot.cs` (the network-only bot).
- `DeepSetsBlendBot.cs` → `Bots/src/DeepSetsBlendBot.cs` (the blended bot).
- `DeepSetsCore.cs` → `Bots/src/DeepSetsCore.cs`. Shared infrastructure both
  bots depend on: state-to-feature encoding, the ONNX inference wrapper, and
  card metadata.
- `DeepSetsValueNetwork.onnx` → `Bots/DeepSetsValueNetwork.onnx`, the
  trained model (see below).
- `Bots.csproj` → `Bots/Bots.csproj`, replacing the existing file. Two
  additions relative to an untouched checkout, nothing else changed:
  ```xml
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.26.0" />
  ```
  ```xml
  <None Update="DeepSetsValueNetwork.onnx">
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  </None>
  ```
- `SUBMISSION.md`: this file, not part of the build.

## Dependency

`Microsoft.ML.OnnxRuntime` **1.26.0**, CPU inference only — no GPU / CUDA
execution provider is referenced or required.

## Model file

| | |
|---|---|
| File | `DeepSetsValueNetwork.onnx` |
| Size | 294,979 bytes |
| SHA-256 | `86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915` |

**Placement:** the loader in `PregamePrepare()` searches, in order:
`AppContext.BaseDirectory`, the current working directory, and a `Bots/`
subdirectory of each. I verified directly that `AppContext.BaseDirectory`
is what resolves in practice for a bot loaded dynamically into `GameRunner`.
That means **the same directory the built `GameRunner` executable itself
sits in**, not the directory `Bots.dll` gets loaded from. Put
`DeepSetsValueNetwork.onnx` there.

**Failure behavior:** if the model can't be loaded (file missing, corrupt,
or the OnnxRuntime assemblies below are missing), both bots catch the error
in `PregamePrepare()` and fall back to the original SakkirinaSolo heuristic
instead of crashing, so the game completes but the bot plays substantially
worse. To confirm the model loaded, run with `SOT_LOG=1` and look for the
`PregamePrepare` line in the log.

**Runtime dependency note:** `Microsoft.ML.OnnxRuntime.dll`,
`Newtonsoft.Json.dll`, and the matching
`runtimes/<rid>/native/libonnxruntime.*` must sit in the same directory as
`Bots.dll` in whatever build output actually runs the match. This is
because `GameRunner` loads `Bots.dll` dynamically by reflection rather than
a compile-time project reference, so NuGet's normal dependency resolution
never places them there automatically. All three come from the
`Microsoft.ML.OnnxRuntime` NuGet package (version 1.26.0) once it's
restored for `Bots.csproj`.

## Measured results

400 games per matchup, 10s/turn, competition 6-patron set (ANSEI,
DUKE_OF_CROWS, RAJHIN, ORGNUM, PELIN, SAINT_ALESSIA), seats swapped between
halves and results aggregated with a 95% Wilson interval:

| Bot | Opponent | Win rate | 95% CI |
|---|---|---|---|
| DeepSetsBot | SakkirinaSolo | 76.3% | [71.8, 80.2] |
| DeepSetsBot | MCTSBot | 88.5% | [85.0, 91.3] |
| DeepSetsBlendBot | SakkirinaSolo | 77.0% | [72.6, 80.9] |
| DeepSetsBlendBot | MCTSBot | 88.3% | [84.7, 91.1] |

## Resource use

Peak RSS ~162 MB for the whole `GameRunner` process (both bots plus the
engine running one game), measured with `/usr/bin/time -l`.

## Build

```
dotnet build Bots/Bots.csproj -c Release
dotnet build GameRunner/GameRunner.csproj -c Release
```
0 errors.
