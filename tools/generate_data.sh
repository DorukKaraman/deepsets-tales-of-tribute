#!/usr/bin/env bash
# Cluster-scale training-data generation: <bot> vs <bot> (default
# SakkirinaGenNeural, our current best agent, 68% vs SakkirinaSolo -- see
# --bot) through GameRunner's --log-training-data, sharded across --jobs OS
# processes.
#
# DIRECT BINARY INVOCATION ONLY. At thousands of games, `dotnet run`'s MSBuild
# up-to-date check on every invocation is unacceptable overhead -- this script
# builds once, then every worker process invokes the built GameRunner binary
# directly (see tools/generate_data.py).
#
# RESUMABLE. Each worker (one process running --runs N) writes a JSON marker
# on successful completion. Relaunching this script with the SAME --games
# --jobs --seed-base --out-dir skips every worker whose marker matches the
# plan and only (re)runs the rest -- a cluster job killed at hour 4 does not
# lose the first 4 hours of shards. Relaunching with DIFFERENT arguments
# reshuffles job boundaries and will re-run everything; markers are only
# trusted when they match the current plan exactly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAME_RUNNER_DIR="$REPO_ROOT/GameRunner"
BOTS_DIR="$REPO_ROOT/Bots"

# Hardcoded, not a flag, same reasoning as tools/benchmark.sh: this exists to
# produce training data worth trusting, and a Debug build silently disables
# JIT optimizations for the whole wall-clock-budgeted search.
CONFIGURATION="Release"
BOTS_TFM="netstandard2.1"
BINARY="$GAME_RUNNER_DIR/bin/$CONFIGURATION/net8.0/GameRunner"

DETECTED_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
DEFAULT_BOT="SakkirinaGenNeural"

GAMES=""
JOBS="$DETECTED_JOBS"
OUT_DIR=""
SEED_BASE=""
BOT="$DEFAULT_BOT"
EXPECT_ONNX_SHA256=""
DRY_RUN=0
SKIP_BUILD=0
TASK_ID=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --games <n> --out-dir <path> [options]

Required:
  --games <n>          Total games to generate across all jobs
  --out-dir <path>      Output directory for shards + resume markers (no
                        default on purpose -- a multi-thousand-game run
                        should never land in a generic default location)

Options:
  --bot <name>          Bot to self-play, both sides (default: $DEFAULT_BOT,
                        our current best agent). SakkirinaGen (the 2025
                        winner's heuristic, no ONNX model needed) is still a
                        valid value if you want that dataset instead.
  --jobs <n>            Max concurrent OS processes (default: detected CPU count = $DETECTED_JOBS)
  --seed-base <n>       First seed to use (default: derived from current time)
  --expect-onnx-sha256 <hash>
                        Abort unless the onnx model's sha256 matches exactly.
                        Optional -- the unconditional existence/consistency
                        check below runs either way; this additionally pins a
                        SPECIFIC model, e.g. so a cluster run can't silently
                        pick up a model swapped in after this was written.
  --skip-build          Skip the dotnet build steps -- for SLURM array tasks,
                        where the binary is already built once on the login
                        node beforehand and every array task must not rebuild
                        it (concurrent builds from N array tasks would race on
                        the same output). The Bots.dll/onnx sha256 pre-flight
                        check still runs unconditionally -- it's read-only and
                        is exactly what would catch a stale/wrong binary that
                        a skipped build can't.
  --task-id <n>         SLURM array task index. When set: appended to
                        --out-dir as a task_NN subdirectory (so tasks never
                        write to the same shard files), the seed is derived as
                        seed-base + task_id * 1000000 (so tasks can never
                        generate identical games), and --jobs is forced to 1
                        (one array task = one core = one process). Requires
                        --seed-base to be given explicitly, since the
                        derivation has to happen here, before generate_data.py
                        ever runs.
  --dry-run             Plan the work and print every job's command line;
                        run nothing. Skips the build and integrity checks too,
                        so the plan can be previewed without a built binary.
  -h, --help            Show this help

Fixed by design (not flags): --timeout 10,
--patrons ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA,
--log-training-data. See tools/generate_data.py if a different timeout or
patron set is ever needed.

Examples:
  $(basename "$0") --games 4 --jobs 2 --out-dir /tmp/sakgen_data --dry-run
  $(basename "$0") --games 10000 --out-dir /data/sakgen_run1
  $(basename "$0") --games 300 --out-dir /data/sakgen_run1 --seed-base 42 \\
      --task-id "\$SLURM_ARRAY_TASK_ID" --skip-build
  $(basename "$0") --games 10000 --out-dir /data/sakgenneural_run1 \\
      --expect-onnx-sha256 71d999201b57974477f9ef1b57eb681a4ce5e54aea52293766378f93b8077fd6
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --games) GAMES="${2:-}"; shift 2 ;;
    --bot) BOT="${2:-}"; shift 2 ;;
    --jobs) JOBS="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --seed-base) SEED_BASE="${2:-}"; shift 2 ;;
    --expect-onnx-sha256) EXPECT_ONNX_SHA256="${2:-}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --task-id) TASK_ID="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$GAMES" ] || [ -z "$OUT_DIR" ]; then
  echo "ERROR: --games and --out-dir are required." >&2
  usage
  exit 1
fi

if [ -n "$TASK_ID" ]; then
  if [ -z "$SEED_BASE" ]; then
    echo "ERROR: --task-id requires --seed-base to be given explicitly -- the" >&2
    echo "       per-task seed is derived here (seed-base + task_id * 1000000)," >&2
    echo "       before generate_data.py ever runs, so it can't fall back to a" >&2
    echo "       time-derived default without risking two array tasks colliding." >&2
    exit 1
  fi
  OUT_DIR="$OUT_DIR/task_$(printf '%02d' "$TASK_ID")"
  SEED_BASE=$((SEED_BASE + TASK_ID * 1000000))
  JOBS=1
  echo "=== SLURM array task ==="
  echo "Task ID        : $TASK_ID"
  echo "Resolved seed  : $SEED_BASE"
  echo "Resolved out-dir: $OUT_DIR"
  echo "Jobs forced to : 1"
  echo
fi

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

RUNNER_ARGS=(--binary "$BINARY" --bot "$BOT" --games "$GAMES" --jobs "$JOBS" --out-dir "$OUT_DIR" \
             --configuration "$CONFIGURATION")
if [ -n "$SEED_BASE" ]; then
  RUNNER_ARGS+=(--seed-base "$SEED_BASE")
fi

if [ "$DRY_RUN" = "1" ]; then
  RUNNER_ARGS+=(--dry-run)
  exec "$PYTHON_BIN" "$SCRIPT_DIR/generate_data.py" "${RUNNER_ARGS[@]}"
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
echo "Using binary: $BINARY"
echo

# --- Pre-flight guard: abort if the Bots.dll or onnx model the runner will
# actually load don't match the current $CONFIGURATION source artifacts.
# Unconditional even with --skip-build (in fact ESPECIALLY with --skip-build):
# this is the exact failure mode that would silently corrupt a multi-hour/
# multi-task cluster generation run -- see tools/benchmark.sh for the same
# checks with the same reasoning.
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

EXPECTED_BOTS_SIZE="$(wc -c < "$EXPECTED_BOTS_DLL" | tr -d ' ')"
ACTUAL_BOTS_SIZE="$(wc -c < "$ACTUAL_BOTS_DLL" | tr -d ' ')"
EXPECTED_BOTS_SHA="$(shasum -a 256 "$EXPECTED_BOTS_DLL" | awk '{print $1}')"
ACTUAL_BOTS_SHA="$(shasum -a 256 "$ACTUAL_BOTS_DLL" | awk '{print $1}')"

if [ "$EXPECTED_BOTS_SIZE" != "$ACTUAL_BOTS_SIZE" ] || [ "$EXPECTED_BOTS_SHA" != "$ACTUAL_BOTS_SHA" ]; then
  echo "ERROR: GameRunner's Bots.dll does not match the current $CONFIGURATION Bots.dll." >&2
  echo "  expected ($EXPECTED_BOTS_DLL):" >&2
  echo "    size=$EXPECTED_BOTS_SIZE sha256=$EXPECTED_BOTS_SHA" >&2
  echo "  actual   ($ACTUAL_BOTS_DLL):" >&2
  echo "    size=$ACTUAL_BOTS_SIZE sha256=$ACTUAL_BOTS_SHA" >&2
  echo "  A stale or wrong-configuration Bots.dll is shadowing the build. Aborting" >&2
  echo "  before any games run -- generating thousands of games against the wrong" >&2
  echo "  binary is worse than not generating them at all." >&2
  exit 1
fi

echo "Verified: Bots.dll in GameRunner's output matches the $CONFIGURATION artifact."
echo "  Bots.dll : $ACTUAL_BOTS_DLL (size=$ACTUAL_BOTS_SIZE, sha256=$ACTUAL_BOTS_SHA)"
echo

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
  echo "  source ($ONNX_SRC): sha256=$ONNX_SRC_SHA" >&2
  echo "  output ($ONNX_DST): sha256=$ONNX_DST_SHA" >&2
  exit 1
fi

echo "Verified: onnx model in GameRunner's output matches models/DeepSetsValueNetwork.onnx."
echo "  onnx     : $ONNX_DST (sha256=$ONNX_DST_SHA)"
echo

# Optional pin, on top of the unconditional existence/consistency check
# above: SakkirinaGen needed no model at all, but SakkirinaGenNeural (and any
# future ONNX-backed bot passed via --bot) does -- if it fails to load, the
# bot silently falls back to a heuristic evaluator and keeps playing, so a
# multi-thousand-game cluster run would generate nothing but quietly
# wrong data with no error anywhere. This lets a cluster invocation pin the
# exact model it was validated against, so a model swapped in later (even a
# legitimately newer one) can't silently change what an in-flight or
# about-to-launch run generates.
if [ -n "$EXPECT_ONNX_SHA256" ] && [ "$ONNX_DST_SHA" != "$EXPECT_ONNX_SHA256" ]; then
  echo "ERROR: onnx model sha256 does not match --expect-onnx-sha256." >&2
  echo "  expected: $EXPECT_ONNX_SHA256" >&2
  echo "  actual  : $ONNX_DST_SHA ($ONNX_DST)" >&2
  exit 1
fi

RUNNER_ARGS+=(--onnx-sha256 "$ONNX_DST_SHA")

RUNNER_ARGS+=(--bots-dll-sha256 "$ACTUAL_BOTS_SHA")

exec "$PYTHON_BIN" "$SCRIPT_DIR/generate_data.py" "${RUNNER_ARGS[@]}"
