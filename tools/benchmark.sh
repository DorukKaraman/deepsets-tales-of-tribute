#!/usr/bin/env bash
# Measure a bot's win rate against a baseline via GameRunner, with enough games
# that the confidence interval is smaller than the effects being chased.
#
# PARALLELISM IS PROCESS-LEVEL ONLY. GameRunner reuses one bot instance across
# --runs N within a single process; concurrent games sharing an instance would
# corrupt the ONNX session and telemetry counters. This script never passes
# GameRunner's own --threads flag -- instead it builds once, then spawns one OS
# process per game (via tools/benchmark_runner.py, bounded by --jobs concurrent
# processes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAME_RUNNER_DIR="$REPO_ROOT/GameRunner"
BOTS_DIR="$REPO_ROOT/Bots"

# Hardcoded, not a flag: this script exists to produce numbers worth trusting,
# and a Debug build silently disables JIT optimizations for the whole
# wall-clock-budgeted search -- there is no legitimate reason to benchmark
# anything but Release.
CONFIGURATION="Release"
BOTS_TFM="netstandard2.1"

DETECTED_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

# The value network is trained exclusively on the competition patron pool --
# benchmarking against the engine's default 9-patron pool would test it on
# three decks (PSIJIC, HLAALU, RED_EAGLE) it has never seen and that cannot
# occur in competition. This default keeps benchmarks representative of what
# will actually be played; override with --patrons if you deliberately want
# a different pool for some other comparison.
DEFAULT_PATRONS="ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA"

BOT_A=""
BOT_B=""
GAMES=100
JOBS="$DETECTED_JOBS"
TIMEOUT=""
SEED_BASE=""
PATRONS="$DEFAULT_PATRONS"
CSV_PATH="$SCRIPT_DIR/out/benchmark_log.csv"
REPEAT_SEED=""
REPEAT_SWAPPED=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --bot-a <name> --bot-b <name> (--fast | --full | --timeout <sec>) [options]

Required:
  --bot-a <name>       Bot whose win rate is being measured (e.g. ISMCTSBot)
  --bot-b <name>       Opponent / baseline bot (e.g. MaxPrestigeBot)
  One of:
    --fast             Preset: --timeout 1  (quick iteration -- not for trusted numbers)
    --full             Preset: --timeout 10 (for numbers you'll actually trust)
    --timeout <sec>    Explicit per-move timeout in seconds
                       (no silent default -- picking one is required so a fast,
                       untrustworthy run can never be mistaken for a full one)

Options:
  --games <n>          Total games to play, split 50/50 with first-player
                       swapped between halves (default: 100)
  --jobs <n>           Max concurrent OS processes (default: detected CPU count = $DETECTED_JOBS)
  --seed-base <n>      First seed to use (default: derived from current time)
  --patrons <list>     Comma-separated patron pool passed to GameRunner
                       (default: $DEFAULT_PATRONS -- the
                       competition set the value network is actually trained
                       on; the engine's own default is a different 9-patron
                       pool including three decks the network has never seen)
  --csv <path>         CSV log path (default: tools/out/benchmark_log.csv)
  --repeat-seed <n>    Re-run exactly one game with this seed alone (jobs=1 in
                       effect, since only one game runs), bypassing normal
                       planning/CSV logging -- for testing whether a failure
                       found during a larger run is deterministic or load-
                       dependent. Full stdout+stderr of a failure is always
                       saved to tools/out/failures/{seed}_{swapped}.txt.
  --repeat-swapped     With --repeat-seed, use bot-b as P1 instead of bot-a.
  -h, --help           Show this help

Notes:
  - Builds GameRunner once (dotnet build -c Release) before spawning any
    worker process; workers invoke the built binary directly, never
    'dotnet run', so concurrent workers can't race on the same build output.
  - SOT_LOG / SOT_LOG_FILE / SOT_DUMP_DIR are stripped from every worker's
    environment regardless of what's set in this shell -- dumping at
    benchmark scale would write gigabytes.
  - Games that end via a bot exception/timeout/illegal move are reported
    separately from clean wins/losses (see the script's own output); the
    headline win rate and CSV row only reflect clean, decided games.

Examples:
  $(basename "$0") --bot-a ISMCTSBot --bot-b MaxPrestigeBot --games 4 --fast
  $(basename "$0") --bot-a ISMCTSBot --bot-b MaxPrestigeBot --games 300 --full
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --bot-a) BOT_A="${2:-}"; shift 2 ;;
    --bot-b) BOT_B="${2:-}"; shift 2 ;;
    --games) GAMES="${2:-}"; shift 2 ;;
    --jobs) JOBS="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --fast) TIMEOUT=1; shift ;;
    --full) TIMEOUT=10; shift ;;
    --seed-base) SEED_BASE="${2:-}"; shift 2 ;;
    --patrons) PATRONS="${2:-}"; shift 2 ;;
    --csv) CSV_PATH="${2:-}"; shift 2 ;;
    --repeat-seed) REPEAT_SEED="${2:-}"; shift 2 ;;
    --repeat-swapped) REPEAT_SWAPPED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$BOT_A" ] || [ -z "$BOT_B" ]; then
  echo "ERROR: --bot-a and --bot-b are required." >&2
  usage
  exit 1
fi

if [ -z "$TIMEOUT" ]; then
  echo "ERROR: specify one of --fast, --full, or --timeout <sec>." >&2
  echo "       There is no silent default, so an iteration-speed run can never" >&2
  echo "       be accidentally trusted as a full one." >&2
  usage
  exit 1
fi

# Defense in depth: benchmark_runner.py also strips these per-worker, but make
# sure this shell isn't the source of an accidental leak either.
unset SOT_LOG SOT_LOG_FILE SOT_DUMP_DIR 2>/dev/null || true

echo "=== Building Bots ($CONFIGURATION) explicitly -- GameRunner has no compile-time"
echo "    reference to it, so building GameRunner alone does not guarantee this ==="
( cd "$BOTS_DIR" && dotnet build -c "$CONFIGURATION" )
echo

echo "=== Building GameRunner ($CONFIGURATION) once, before any parallel workers start ==="
( cd "$GAME_RUNNER_DIR" && dotnet build -c "$CONFIGURATION" )
echo

BINARY="$(find "$GAME_RUNNER_DIR/bin/$CONFIGURATION" -maxdepth 2 -type f -name GameRunner 2>/dev/null | head -n 1)"
if [ -z "$BINARY" ] || [ ! -x "$BINARY" ]; then
  echo "ERROR: could not find a built, executable GameRunner binary under $GAME_RUNNER_DIR/bin/$CONFIGURATION" >&2
  exit 1
fi
RUNNER_OUT_DIR="$(dirname "$BINARY")"
echo "Using binary: $BINARY"
echo

# --- Pre-flight guard: abort if the Bots.dll (or onnx) the runner will actually
# load doesn't match what was just freshly built in $CONFIGURATION. Recency
# (mtime) is NOT checked here on purpose -- a fresh Debug DLL under a Release
# runner must still fail this check, so identity is compared by size+sha256.
EXPECTED_BOTS_DLL="$BOTS_DIR/bin/$CONFIGURATION/$BOTS_TFM/Bots.dll"
ACTUAL_BOTS_DLL="$RUNNER_OUT_DIR/Bots/Bots.dll"

if [ ! -f "$EXPECTED_BOTS_DLL" ]; then
  echo "ERROR: expected Bots.dll not found at $EXPECTED_BOTS_DLL -- did the Bots build above fail silently?" >&2
  exit 1
fi
if [ ! -f "$ACTUAL_BOTS_DLL" ]; then
  echo "ERROR: GameRunner output has no Bots.dll at $ACTUAL_BOTS_DLL" >&2
  exit 1
fi

EXPECTED_BOTS_SIZE="$(wc -c < "$EXPECTED_BOTS_DLL" | tr -d ' ')"
ACTUAL_BOTS_SIZE="$(wc -c < "$ACTUAL_BOTS_DLL" | tr -d ' ')"
EXPECTED_BOTS_SHA="$(shasum -a 256 "$EXPECTED_BOTS_DLL" | awk '{print $1}')"
ACTUAL_BOTS_SHA="$(shasum -a 256 "$ACTUAL_BOTS_DLL" | awk '{print $1}')"

if [ "$EXPECTED_BOTS_SIZE" != "$ACTUAL_BOTS_SIZE" ] || [ "$EXPECTED_BOTS_SHA" != "$ACTUAL_BOTS_SHA" ]; then
  echo "ERROR: GameRunner's Bots.dll does not match the freshly built $CONFIGURATION Bots.dll." >&2
  echo "  expected ($EXPECTED_BOTS_DLL):" >&2
  echo "    size=$EXPECTED_BOTS_SIZE sha256=$EXPECTED_BOTS_SHA" >&2
  echo "  actual   ($ACTUAL_BOTS_DLL):" >&2
  echo "    size=$ACTUAL_BOTS_SIZE sha256=$ACTUAL_BOTS_SHA" >&2
  echo "  A stale or wrong-configuration Bots.dll is shadowing the build. Aborting" >&2
  echo "  before any games run -- benchmarking the wrong binary is worse than not" >&2
  echo "  benchmarking at all." >&2
  exit 1
fi

ONNX_SRC="$REPO_ROOT/models/DeepSetsValueNetwork.onnx"
ONNX_DST="$RUNNER_OUT_DIR/DeepSetsValueNetwork.onnx"
if [ ! -f "$ONNX_SRC" ]; then
  echo "ERROR: expected onnx model not found at $ONNX_SRC" >&2
  exit 1
fi
if [ ! -f "$ONNX_DST" ]; then
  echo "ERROR: GameRunner output has no onnx model at $ONNX_DST" >&2
  exit 1
fi
ONNX_SRC_SHA="$(shasum -a 256 "$ONNX_SRC" | awk '{print $1}')"
ONNX_DST_SHA="$(shasum -a 256 "$ONNX_DST" | awk '{print $1}')"
if [ "$ONNX_SRC_SHA" != "$ONNX_DST_SHA" ]; then
  echo "ERROR: GameRunner's onnx model does not match models/DeepSetsValueNetwork.onnx." >&2
  echo "  source sha256: $ONNX_SRC_SHA ($ONNX_SRC)" >&2
  echo "  output sha256: $ONNX_DST_SHA ($ONNX_DST)" >&2
  exit 1
fi

echo "Verified: Bots.dll and onnx model in GameRunner's output match the freshly built $CONFIGURATION artifacts."
echo "  Bots.dll : $ACTUAL_BOTS_DLL (size=$ACTUAL_BOTS_SIZE, sha256=$ACTUAL_BOTS_SHA)"
echo "  onnx     : $ONNX_DST (sha256=$ONNX_DST_SHA)"
echo

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

RUNNER_ARGS=(--binary "$BINARY" --bot-a "$BOT_A" --bot-b "$BOT_B" --games "$GAMES" \
             --timeout "$TIMEOUT" --jobs "$JOBS" --csv "$CSV_PATH" --patrons "$PATRONS" \
             --bots-dll-path "$ACTUAL_BOTS_DLL" --configuration "$CONFIGURATION")
if [ -n "$SEED_BASE" ]; then
  RUNNER_ARGS+=(--seed-base "$SEED_BASE")
fi
if [ -n "$REPEAT_SEED" ]; then
  RUNNER_ARGS+=(--repeat-seed "$REPEAT_SEED")
  if [ "$REPEAT_SWAPPED" = "1" ]; then
    RUNNER_ARGS+=(--repeat-swapped)
  fi
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_runner.py" "${RUNNER_ARGS[@]}"
