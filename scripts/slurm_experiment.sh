#!/usr/bin/env bash
# SLURM array template for the paper's experiment configs (experiments/configs/):
# one array task = one game = one process, as in scripts/slurm_benchmark.sh, with
# the experiment chosen by a config file at submission time.
#
# One game per task because GameRunner reuses a bot instance across --runs N and
# GameEndStatsCounter reports only aggregate counts. --runs 1 per process
# attributes every game's outcome exactly, including a loss by timeout.
# Concurrency comes from SLURM running array elements in parallel (the %N
# throttle).
#
# Cluster details (RWTH CLAIX): partition c23ms, 1 core/task, ~2GB/core, no
# --account, .NET via `source $HOME/tot/env.sh`, repo at
# $HOME/tot/deepsets-tales-of-tribute. Edit below if yours differ.
#
# ---------------------------------------------------------------------------
# Choosing the experiment
#
#   sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep  --array=0-1999%32 scripts/slurm_experiment.sh
#   sbatch --export=ALL,SOT_EXP_CONFIG=time_scaling --array=0-1999%32 scripts/slurm_experiment.sh
#
# --export=ALL,... is required: without ALL, SLURM replaces the environment
# instead of adding to it, and `source $HOME/tot/env.sh` runs in a stripped shell.
#
# Print the plan first; it gives each matchup's task-id range for --array:
#
#   tools/benchmark_cluster.sh --config <name> --out-dir "$OUT_DIR" --dry-run
#
# Array size: 2000-task arrays may exceed the site's MaxArraySize
# (`scontrol show config | grep -i MaxArraySize` on the login node). If so,
# submit independent chunks:
#   sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=0-999%32    scripts/slurm_experiment.sh
#   sbatch --export=ALL,SOT_EXP_CONFIG=alpha_sweep --array=1000-1999%32 scripts/slurm_experiment.sh
# If the account limits submitted jobs, use scripts/slurm_experiment_batched.sh.
#
# --time is per task (one game), and a game's cost scales with the config's
# per-turn budget. The default below covers time_scaling's 30s/turn row and is
# generous for alpha_sweep; if over-requesting costs fair-share, submit
# alpha_sweep with `sbatch --time=00:15:00`.
#
#   alpha_sweep   (10s/turn)  ~60-90s per game
#   time_scaling  (2s/turn)   ~20-30s per game
#   time_scaling  (30s/turn)  ~5-8 min per game
#
# Resumable: a task writes its result JSON only after its game completes, and a
# task whose result file exists is skipped, so resubmitting any chunk is safe.
#
# Setup, before sbatch:
#   1. Clone into a new directory, not an existing $HOME/tot/ScriptsOfTribute-Core
#      checkout, which is a different repository without the experiment configs
#      and agents.
#        source $HOME/tot/env.sh
#        mkdir -p $HOME/tot
#        git clone -b experiments \
#            https://github.com/DorukKaraman/deepsets-tales-of-tribute.git \
#            $HOME/tot/deepsets-tales-of-tribute
#        cd $HOME/tot/deepsets-tales-of-tribute
#        ./scripts/fetch_baselines.sh        # SakkirinaSolo + SakkirinaScaled
#        dotnet build Bots/Bots.csproj -c Release
#        dotnet build GameRunner/GameRunner.csproj -c Release
#   2. mkdir -p $HOME/tot/deepsets-tales-of-tribute/logs
#      SLURM does not create the directory for #SBATCH --output/--error; if it is
#      missing, every task fails before this script runs.
#   3. Preview the plan (see above).
#   4. Submit.

#SBATCH --job-name=deepsets_exp
#SBATCH --partition=c23ms
#SBATCH --array=0-399%32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

# --output/--error are relative to the directory sbatch runs in, because sbatch
# does not reliably expand shell variables like $HOME in #SBATCH directives.
# Submit from $HOME/tot/deepsets-tales-of-tribute.

set -euo pipefail

# --- Edit if your setup differs from the CLUSTER DETAILS above ---
REPO_ROOT="$HOME/tot/deepsets-tales-of-tribute"   # the fresh clone; see SETUP above
CONFIG="${SOT_EXP_CONFIG:-}"             # set via --export=ALL,SOT_EXP_CONFIG=...
OUT_DIR_BASE="$HOME/tot/experiment_results"  # small JSON files, so $HOME is fine
# The seed base comes from the config, so it is fixed across the array and every
# resubmission. Override only for a second independent replicate.
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

# One output directory per config, so two experiments never pool result files
# even if they share a matchup label.
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
