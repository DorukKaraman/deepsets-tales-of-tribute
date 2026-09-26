#!/usr/bin/env bash
# Cluster-scale benchmark harness: one game per SLURM array task, through
# GameRunner directly (no --log-training-data). The matchup list, per-matchup
# engine --timeout, game counts and per-task environment all come from a JSON
# experiment config -- see tools/benchmark_cluster.py for the format and the
# task-id layout, and experiments/configs/ for the configs themselves.
#
# The default config reproduces the original hardcoded 10-matchup benchmark
# exactly, so an existing run resumes unchanged across the config rework.
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

DEFAULT_CONFIG="legacy_paper_benchmark"

# MANDATORY pin, not optional like tools/generate_data.sh's --expect-onnx-sha256.
# A bot that fails to load its model does not crash or refuse to play -- it
# silently falls back to a heuristic evaluator and keeps going, so a 400-game
# matchup against the wrong (or no) model would produce a full set of
# plausible-looking numbers for the wrong experiment, with nothing anywhere
# flagging it. This check is therefore unconditional, on every single task.
#
# The pin is a LIST now, not a single hash: per-seed training
# (scripts/slurm_train.sh) produces several legitimate models, so "the one
# correct model" is no longer a well-defined thing. The list comes from the
# config's "allowed_onnx_sha256" (read here with python3, so the config stays
# the single source of truth), plus anything added with
# --allow-onnx-sha256. What is NOT negotiable is that the model GameRunner
# will actually load has to be ON the list.
#
# Note this checks GameRunner's own copy. A matchup that sets SOT_MODEL_PATH
# points its bot somewhere else entirely; tools/benchmark_cluster.py checks
# that one against the same list, because this script cannot -- it does not
# know which task is about to run.

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

# --- Pre-flight guard 1/2: Bots.dll -- same check, same reasoning as
# tools/generate_data.sh. Only printed once per task's log, but cheap enough
# (a stat + a sha256 of a few hundred KB) that running it on every task is not
# worth special-casing away.
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

# --- Pre-flight guard 2/2: the STALE-BUILD check. GameRunner ships its own
# copy of the model, and this confirms that copy is the one the config expects.
#
# It is checked against "builtin_onnx_sha256", NOT against
# "allowed_onnx_sha256", and the separation is the point. allowed_onnx_sha256
# lists the models a ROW may load through SOT_MODEL_PATH; the built-in copy is
# a different thing that happens to be a model too. Checking the built-in copy
# against the allowed list forced every config to whitelist the shipped hash
# even when no row should ever load it -- and once the shipped hash is on the
# allowed list, a row that silently fell back to the built-in model passes the
# pin. For a control row running the same architecture at the same speed,
# nothing else would have caught it.
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
