#!/usr/bin/env bash
# Cluster benchmark harness: one game per SLURM array task, run through the
# built GameRunner binary (no --log-training-data). Matchups, per-matchup engine
# --timeout, game counts and per-task environment come from a JSON experiment
# config; see tools/benchmark_cluster.py for the format and task-id layout, and
# experiments/configs/ for the configs.
#
# Builds once; every array task then invokes the binary directly.
#
# Resumable: benchmark_cluster.py writes a result file only after a game
# completes, and a task whose result file exists is skipped. Resubmitting the
# same array index retries a task that has none.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAME_RUNNER_DIR="$REPO_ROOT/GameRunner"
BOTS_DIR="$REPO_ROOT/Bots"

# Always Release: a Debug build disables JIT optimizations under a wall-clock
# search budget.
CONFIGURATION="Release"
BOTS_TFM="netstandard2.1"
BINARY="$GAME_RUNNER_DIR/bin/$CONFIGURATION/net8.0/GameRunner"

DEFAULT_CONFIG="legacy_paper_benchmark"

# Model pins, checked on every task. A bot that fails to load its model falls
# back to a heuristic evaluator without any error, so a wrong model yields
# plausible numbers for the wrong experiment.
#
# GameRunner's built-in copy is checked here against the config's
# "builtin_onnx_sha256" (guard 2/2). Models a row loads through SOT_MODEL_PATH
# are checked by tools/benchmark_cluster.py against "allowed_onnx_sha256" plus
# any --allow-onnx-sha256, since only it knows which task is running.

CONFIG="$DEFAULT_CONFIG"
TASK_ID=""
OUT_DIR=""
SEED_BASE=""
EXTRA_HASHES=()
DRY_RUN=0
SKIP_BUILD=0
CALIBRATE=0
CALIBRATION_GAMES=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --out-dir <path> [--config <name|path>] [--task-id <n>] [options]

Required:
  --out-dir <path>      Output directory for result JSON files

Options:
  --config <name|path>  Experiment config (default: $DEFAULT_CONFIG). A path,
                         or a bare name resolved in experiments/configs/.
                         Available:
$(ls "$REPO_ROOT/experiments/configs"/*.json 2>/dev/null | sed 's#.*/##; s/\.json$//; s/^/                           /' || echo "                           (none found)")
  --task-id <n>          Global task id. Required unless --dry-run is given
                          without it (which prints the whole plan), or
                          --calibrate is given.
  --seed-base <n>        Override the config's seed_base. Must stay fixed
                          across the whole array and any resubmission, or
                          retried tasks would silently regenerate different
                          games than originally planned.
  --calibrate            Run the config's calibration matchups (20 games by
                          default) and report mean evaluations per turn per
                          agent, so an equal-effort time scale can be picked.
                          Sequential; does not touch the main results.
  --calibration-games <n>  Override the calibration game count.
  --allow-onnx-sha256 <hash>
                          Add an extra allowed onnx sha256 on top of the
                          config's list. Repeatable. The check itself is never
                          skippable, only its target.
  --skip-build            Skip the dotnet build steps -- for SLURM array tasks,
                          where the binary is already built once on the login
                          node beforehand. The Bots.dll/onnx sha256 checks
                          still run unconditionally.
  --dry-run               Print the plan; run nothing. Skips the build and
                           integrity checks too, so the plan can be previewed
                           without a built binary.
  -h, --help              Show this help

Examples:
  $(basename "$0") --config alpha_sweep --out-dir "\$OUT_DIR" --dry-run
  $(basename "$0") --config equal_effort --out-dir "\$OUT_DIR" --calibrate
  $(basename "$0") --config alpha_sweep --out-dir "\$OUT_DIR" \\
      --task-id "\$SLURM_ARRAY_TASK_ID" --skip-build
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2 ;;
    --task-id) TASK_ID="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --seed-base) SEED_BASE="${2:-}"; shift 2 ;;
    --calibrate) CALIBRATE=1; shift ;;
    --calibration-games) CALIBRATION_GAMES="${2:-}"; shift 2 ;;
    --allow-onnx-sha256) EXTRA_HASHES+=("${2:-}"); shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$OUT_DIR" ]; then
  echo "ERROR: --out-dir is required." >&2
  usage
  exit 1
fi

if [ "$DRY_RUN" = "0" ] && [ "$CALIBRATE" = "0" ] && [ -z "$TASK_ID" ]; then
  echo "ERROR: --task-id is required (except for a plan-only --dry-run, or --calibrate)." >&2
  usage
  exit 1
fi

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

RUNNER_ARGS=(--config "$CONFIG" --out-dir "$OUT_DIR" --configuration "$CONFIGURATION")
[ -n "$TASK_ID" ] && RUNNER_ARGS+=(--task-id "$TASK_ID")
[ -n "$SEED_BASE" ] && RUNNER_ARGS+=(--seed-base "$SEED_BASE")
[ "$CALIBRATE" = "1" ] && RUNNER_ARGS+=(--calibrate)
[ -n "$CALIBRATION_GAMES" ] && RUNNER_ARGS+=(--calibration-games "$CALIBRATION_GAMES")
for h in ${EXTRA_HASHES+"${EXTRA_HASHES[@]}"}; do
  RUNNER_ARGS+=(--allow-onnx-sha256 "$h")
done

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

# Pre-flight guard 1/2: Bots.dll, as in tools/generate_data.sh.
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

# Pre-flight guard 2/2: GameRunner's built-in model must match the config's
# "builtin_onnx_sha256". It is kept separate from "allowed_onnx_sha256": if the
# shipped hash had to be on the allowed list, a row that fell back to the
# built-in model would pass its pin.
BUILTIN_HASH="$("$PYTHON_BIN" -c '
import json, sys, os
sys.path.insert(0, sys.argv[1])
import benchmark_cluster as bc
try:
    cfg = bc.load_config(sys.argv[2], None)
except Exception as e:
    print("CONFIG_ERROR " + str(e).replace("\n", " "))
    sys.exit(0)
print(cfg["builtin_onnx_sha256"])
' "$SCRIPT_DIR" "$CONFIG")"

if [[ "$BUILTIN_HASH" == CONFIG_ERROR* ]]; then
  echo "ERROR: could not read the experiment config: ${BUILTIN_HASH#CONFIG_ERROR }" >&2
  exit 1
fi

ONNX_DST="$RUNNER_OUT_DIR/DeepSetsValueNetwork.onnx"
if [ ! -f "$ONNX_DST" ]; then
  echo "ERROR: GameRunner output has no onnx model at $ONNX_DST" >&2
  exit 1
fi
ONNX_DST_SHA="$(shasum -a 256 "$ONNX_DST" | awk '{print $1}')"

if [ "$ONNX_DST_SHA" != "$BUILTIN_HASH" ]; then
  echo "ERROR: GameRunner's built-in onnx model is not the one this config expects" >&2
  echo "       -- ABORTING before running." >&2
  echo "  file    : $ONNX_DST" >&2
  echo "  actual  : $ONNX_DST_SHA" >&2
  echo "  expected: $BUILTIN_HASH  (config's builtin_onnx_sha256)" >&2
  echo "  This is the stale-build check: it says the binary you are about to" >&2
  echo "  run was built against a different model than the config was written" >&2
  echo "  for. Rebuild, or set builtin_onnx_sha256 in the config if the change" >&2
  echo "  is deliberate." >&2
  echo "  It is NOT the row pin. Models a matchup loads via SOT_MODEL_PATH go" >&2
  echo "  in allowed_onnx_sha256 (or --allow-onnx-sha256), and are additionally" >&2
  echo "  verified per game against what the bot logs having loaded." >&2
  exit 1
fi

RUNNER_ARGS+=(--binary "$BINARY" --onnx-sha256 "$ONNX_DST_SHA" --bots-dll-sha256 "$ACTUAL_BOTS_SHA")

exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_cluster.py" "${RUNNER_ARGS[@]}"
