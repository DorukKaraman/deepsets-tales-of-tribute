#!/usr/bin/env bash
# SLURM array template for the paper's experiment configs (experiments/configs/):
# one array task = one game = one process, exactly as scripts/slurm_benchmark.sh
# does for the legacy 10-matchup benchmark. The only difference is that WHICH
# experiment runs comes from a config file, chosen at submission time.
#
# WHY ONE GAME PER TASK: GameRunner reuses a bot instance across --runs N and
# its GameEndStatsCounter only reports aggregate counts, with no way to tell
# which end reason produced which winner. --runs 1 per process is what lets
# every game's outcome -- including a loss caused by a timeout rather than by
# play -- be attributed exactly. Concurrency comes entirely from SLURM running
# multiple array elements at once (the %N throttle), not from anything here.
#
# CLUSTER DETAILS (RWTH CLAIX): partition c23ms, 1 core/task, ~2GB/core, no
# --account needed, .NET via `source $HOME/tot/env.sh`, repo at
# $HOME/tot/ScriptsOfTribute-Core. Edit below if yours differs.
#
# ---------------------------------------------------------------------------
# CHOOSING THE EXPERIMENT
#
#   sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep  --array=0-1999%32 scripts/slurm_experiment.sh
#   sbatch --export=ALL,SOT_EXP_CONFIG=time_scaling --array=0-1999%32 scripts/slurm_experiment.sh
#
# --export=ALL,... is required, not optional: without ALL, SLURM replaces the
# whole environment rather than adding to it, and `source $HOME/tot/env.sh`
# below would run in a stripped shell.
#
# ALWAYS PRINT THE PLAN FIRST. It gives the exact task-id range of every
# matchup, which is what you put in --array:
#
#   tools/benchmark_cluster.sh --config <name> --out-dir "$OUT_DIR" --dry-run
#
# ARRAY SIZE / CHUNKING: the arrays here (2000 tasks) may exceed this cluster's
# MaxArraySize (`scontrol show config | grep -i MaxArraySize` on the login
# node -- a site-wide setting this script cannot see or work around). Submit in
# chunks if so; each chunk is a fully independent submission with no
# coordination needed:
#   sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=0-999%32    scripts/slurm_experiment.sh
#   sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=1000-1999%32 scripts/slurm_experiment.sh
#
# --time IS PER TASK (one game), and one game's cost scales with the config's
# per-turn budget. The default below is sized for the most expensive row of
# time_scaling (30s/turn); it is wildly generous for alpha_sweep and costs
# nothing but scheduling priority to leave that way. If your fair-share
# punishes over-requesting, submit alpha_sweep with `sbatch --time=00:15:00`.
#
#   alpha_sweep   (10s/turn)  ~60-90s per game
#   time_scaling  (2s/turn)   ~20-30s per game
#   time_scaling  (30s/turn)  ~5-8 min per game
#
# RESUMABLE. Every task writes a result JSON file only after its one game
# completes successfully. A task whose result file already exists is skipped
# immediately without running anything, so re-submitting any chunk (or the
# whole array) is always safe -- only genuinely missing/failed tasks do work.
#
# SETUP (do this BEFORE sbatch-ing, not after):
#   1. source $HOME/tot/env.sh
#      cd $HOME/tot/ScriptsOfTribute-Core
#      git pull
#      ./scripts/fetch_baselines.sh          # SakkirinaSolo + SakkirinaScaled
#      dotnet build Bots/Bots.csproj -c Release
#      dotnet build GameRunner/GameRunner.csproj -c Release
#   2. mkdir -p $HOME/tot/ScriptsOfTribute-Core/logs
#      SLURM does NOT create the directory for #SBATCH --output/--error; if it
#      does not exist at submission time, every task fails before this script's
#      body even runs.
#   3. Preview the plan (see ALWAYS PRINT THE PLAN FIRST above).
#   4. Submit.

#SBATCH --job-name=deepsets_exp
#SBATCH --partition=c23ms
#SBATCH --array=0-399%32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

# --output/--error above are RELATIVE (to wherever `sbatch` is invoked from) on
# purpose: #SBATCH directives are parsed by sbatch itself and do not reliably
# expand shell variables like $HOME, so an absolute path would need a literal,
# pre-resolved home directory hardcoded here. This is why the SETUP steps have
# you `cd $HOME/tot/ScriptsOfTribute-Core` before sbatch-ing.

set -euo pipefail

# --- Edit if your setup differs from the CLUSTER DETAILS above ---
REPO_ROOT="$HOME/tot/ScriptsOfTribute-Core"
CONFIG="${SOT_EXP_CONFIG:-}"             # set via --export=ALL,SOT_EXP_CONFIG=...
OUT_DIR_BASE="$HOME/tot/experiment_results"  # small JSON files, not bulk data,
                                              # so $HOME is fine here -- unlike
                                              # the /hpcwork/... paths the
                                              # data-generation runs need
# Seed base comes from the config file itself (each experiment has its own), so
# it is fixed across the whole array and every resubmission by construction.
# Override here only if you deliberately want a second independent replicate.
SEED_BASE=""
# -------------------------------------------------------------

if [ -z "$CONFIG" ]; then
  echo "ERROR: no experiment config selected." >&2
  echo "       Submit with, e.g.:" >&2
  echo "         sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=0-1999%32 $0" >&2
  echo "       Available configs:" >&2
  ls "$REPO_ROOT/experiments/configs"/*.json 2>/dev/null | sed 's#.*/##; s/\.json$//; s/^/         /' >&2
  exit 1
fi

# One output directory per config: two experiments must never pool their result
# files, and the per-matchup subdirectory names alone would not stop them
# (nothing forbids two configs from using the same label).
OUT_DIR="$OUT_DIR_BASE/$CONFIG"

source "$HOME/tot/env.sh"

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID starting on $(hostname)"
echo "CONFIG=$CONFIG  REPO_ROOT=$REPO_ROOT  OUT_DIR=$OUT_DIR"

ARGS=(--config "$CONFIG"
      --task-id "$SLURM_ARRAY_TASK_ID"
      --out-dir "$OUT_DIR"
      --skip-build)
[ -n "$SEED_BASE" ] && ARGS+=(--seed-base "$SEED_BASE")

exec "$REPO_ROOT/tools/benchmark_cluster.sh" "${ARGS[@]}"
