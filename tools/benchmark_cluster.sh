#!/usr/bin/env bash
# Cluster-scale benchmark harness: one game per SLURM array task, through
# GameRunner directly (no --log-training-data). See tools/benchmark_cluster.py
# for the matchup list, task-id layout, and seat-swap handling.
#
# DIRECT BINARY INVOCATION ONLY, same reasoning as tools/generate_data.sh:
# this script builds once, every array task invokes the built binary directly.
#
# RESUMABLE. tools/benchmark_cluster.py writes a result JSON file only after a
# game completes successfully; a task whose result file already exists is
# skipped without running anything. A killed/crashed/still-running task has no
# result file and is simply retried by resubmitting the same array index.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAME_RUNNER_DIR="$REPO_ROOT/GameRunner"
BOTS_DIR="$REPO_ROOT/Bots"

# Hardcoded, not a flag, same reasoning as tools/benchmark.sh and
# tools/generate_data.sh: a Debug build silently disables JIT optimizations
# for the whole wall-clock-budgeted search, and these numbers need to be
# trustworthy.
CONFIGURATION="Release"
BOTS_TFM="netstandard2.1"
BINARY="$GAME_RUNNER_DIR/bin/$CONFIGURATION/net8.0/GameRunner"

# MANDATORY pin, not optional like tools/generate_data.sh's --expect-onnx-sha256.
# A bot that fails to load its model does not crash or refuse to play -- it
# silently falls back to a heuristic evaluator and keeps going, so a 4000-game
# benchmark run against the wrong (or no) model would produce a full set of
# plausible-looking numbers for the wrong experiment, with nothing anywhere
# flagging it. This check is therefore unconditional, every single task,
# with no way to disable it short of editing this constant.
DEFAULT_EXPECT_ONNX_SHA256="86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915"

TASK_ID=""
OUT_DIR=""
SEED_BASE=""
EXPECT_ONNX_SHA256="$DEFAULT_EXPECT_ONNX_SHA256"
DRY_RUN=0
SKIP_BUILD=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --out-dir <path> --seed-base <n> [--task-id <n>] [options]

Required:
  --out-dir <path>      Output directory for result JSON files
  --seed-base <n>       Fixed seed base (seed = seed-base + task-id). Must be
                         given explicitly and stay fixed across the whole
                         array and any resubmission -- see
                         tools/benchmark_cluster.py's --seed-base help.

Options:
  --task-id <n>          Global task id, 0..3999 (10 matchups x 400 games).
                          Required unless --dry-run is given without it, which
                          prints the whole plan instead of one task's.
  --expect-onnx-sha256 <hash>
                          Override the pinned onnx sha256 (default:
                          $DEFAULT_EXPECT_ONNX_SHA256).
                          The check itself is never skippable, only its target.
  --skip-build            Skip the dotnet build steps -- for SLURM array tasks,
                          where the binary is already built once on the login
                          node beforehand. The Bots.dll/onnx sha256 checks
                          still run unconditionally.
  --dry-run               Print the plan; run nothing. Skips the build and
                           integrity checks too, so the plan can be previewed
                           without a built binary.
  -h, --help              Show this help

Fixed by design (not flags): the 10-matchup list, 400 games/matchup,
--timeout 10, --patrons ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA.
See tools/benchmark_cluster.py if any of these ever need to change.

Examples:
  $(basename "$0") --out-dir /tmp/bench_test --seed-base 20260808 --dry-run
  $(basename "$0") --out-dir /tmp/bench_test --seed-base 20260808 --task-id 0 --dry-run
  $(basename "$0") --out-dir \$OUT_DIR --seed-base 20260808 --task-id "\$SLURM_ARRAY_TASK_ID" --skip-build
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --task-id) TASK_ID="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --seed-base) SEED_BASE="${2:-}"; shift 2 ;;
    --expect-onnx-sha256) EXPECT_ONNX_SHA256="${2:-}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$OUT_DIR" ] || [ -z "$SEED_BASE" ]; then
  echo "ERROR: --out-dir and --seed-base are required." >&2
  usage
  exit 1
fi

if [ "$DRY_RUN" = "0" ] && [ -z "$TASK_ID" ]; then
  echo "ERROR: --task-id is required (except for a plan-only --dry-run)." >&2
  usage
  exit 1
fi

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

RUNNER_ARGS=(--out-dir "$OUT_DIR" --seed-base "$SEED_BASE" --configuration "$CONFIGURATION")
if [ -n "$TASK_ID" ]; then
  RUNNER_ARGS+=(--task-id "$TASK_ID")
fi

if [ "$DRY_RUN" = "1" ]; then
  RUNNER_ARGS+=(--dry-run)
  exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_cluster.py" "${RUNNER_ARGS[@]}"
fi

if [ "$SKIP_BUILD" = "1" ]; then
  echo "=== --skip-build: not building. Binary and Bots.dll must already exist"
  echo "    from a prior build (e.g. on the SLURM login node) ==="
  echo
else
  echo "=== Building Bots ($CONFIGURATION) explicitly -- GameRunner has no compile-time"
  echo "    reference to it, so building GameRunner alone does not guarantee this ==="
  ( cd "$BOTS_DIR" && dotnet build -c "$CONFIGURATION" )
  echo

  echo "=== Building GameRunner ($CONFIGURATION) once, before any parallel workers start ==="
  ( cd "$GAME_RUNNER_DIR" && dotnet build -c "$CONFIGURATION" )
  echo
fi

if [ ! -x "$BINARY" ]; then
  echo "ERROR: expected a built, executable GameRunner binary at $BINARY" >&2
  if [ "$SKIP_BUILD" = "1" ]; then
    echo "       (--skip-build was given -- build it once first, e.g. on the login node)" >&2
  else
    echo "       (build succeeded but the output path doesn't match -- TFM changed?)" >&2
  fi
  exit 1
fi
RUNNER_OUT_DIR="$(dirname "$BINARY")"

# --- Pre-flight guard 1/2: Bots.dll -- same check, same reasoning as
# tools/generate_data.sh. Only printed once per task's log, but cheap enough
# (a stat + a sha256 of a few hundred KB) that running it on every one of the
# 4000 tasks is not worth special-casing away.
EXPECTED_BOTS_DLL="$BOTS_DIR/bin/$CONFIGURATION/$BOTS_TFM/Bots.dll"
ACTUAL_BOTS_DLL="$RUNNER_OUT_DIR/Bots/Bots.dll"

if [ ! -f "$EXPECTED_BOTS_DLL" ]; then
  echo "ERROR: expected Bots.dll not found at $EXPECTED_BOTS_DLL -- did the Bots build fail silently, or was it never built?" >&2
  exit 1
fi
if [ ! -f "$ACTUAL_BOTS_DLL" ]; then
  echo "ERROR: GameRunner output has no Bots.dll at $ACTUAL_BOTS_DLL" >&2
  exit 1
fi

EXPECTED_BOTS_SHA="$(shasum -a 256 "$EXPECTED_BOTS_DLL" | awk '{print $1}')"
ACTUAL_BOTS_SHA="$(shasum -a 256 "$ACTUAL_BOTS_DLL" | awk '{print $1}')"

if [ "$EXPECTED_BOTS_SHA" != "$ACTUAL_BOTS_SHA" ]; then
  echo "ERROR: GameRunner's Bots.dll does not match the current $CONFIGURATION Bots.dll." >&2
  echo "  expected ($EXPECTED_BOTS_DLL): sha256=$EXPECTED_BOTS_SHA" >&2
  echo "  actual   ($ACTUAL_BOTS_DLL): sha256=$ACTUAL_BOTS_SHA" >&2
  echo "  A stale or wrong-configuration Bots.dll is shadowing the build. Aborting" >&2
  echo "  before this game runs -- a benchmark against the wrong binary is worse" >&2
  echo "  than not running it at all." >&2
  exit 1
fi

# --- Pre-flight guard 2/2: onnx MANDATORY pin. Unlike tools/generate_data.sh,
# there is no "unpinned" mode here -- see DEFAULT_EXPECT_ONNX_SHA256 above.
# All four benchmarked agents (DeepSetsBot, DeepSetsBlendBot and the two Trim
# variants -- see MATCHUPS in benchmark_cluster.py) load the same filename,
# DeepSetsValueNetwork.onnx. A bot whose model fails to load does not crash;
# it silently falls back to the heuristic evaluator, so this check is the only
# thing standing between a typo and 4000 games of measuring the wrong agent.
for ONNX_NAME in DeepSetsValueNetwork.onnx; do
  ONNX_DST="$RUNNER_OUT_DIR/$ONNX_NAME"
  if [ ! -f "$ONNX_DST" ]; then
    echo "ERROR: GameRunner output has no onnx model at $ONNX_DST" >&2
    exit 1
  fi
  ONNX_DST_SHA="$(shasum -a 256 "$ONNX_DST" | awk '{print $1}')"
  if [ "$ONNX_DST_SHA" != "$EXPECT_ONNX_SHA256" ]; then
    echo "ERROR: onnx model sha256 does not match the pinned value -- ABORTING before running." >&2
    echo "  file    : $ONNX_DST" >&2
    echo "  expected: $EXPECT_ONNX_SHA256" >&2
    echo "  actual  : $ONNX_DST_SHA" >&2
    echo "  A bot whose model fails a load check like this one still plays --" >&2
    echo "  it just silently falls back to a heuristic evaluator, which would" >&2
    echo "  produce a full set of plausible-looking wrong numbers with nothing" >&2
    echo "  flagging it. That is why this is not skippable." >&2
    exit 1
  fi
done

RUNNER_ARGS+=(--binary "$BINARY" --onnx-sha256 "$EXPECT_ONNX_SHA256" --bots-dll-sha256 "$ACTUAL_BOTS_SHA")

exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_cluster.py" "${RUNNER_ARGS[@]}"
