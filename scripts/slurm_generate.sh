#!/usr/bin/env bash
# SLURM array template for cluster self-play data generation. The default bot is
# SakkirinaGenNeural; set BOT_NAME below for another, e.g. SakkirinaGen (the 2025
# winner's heuristic) for the earlier dataset.
#
# 32 independent single-core array tasks rather than one 32-process job. Each
# task is its own process (tools/generate_data.sh --task-id N forces --jobs 1)
# with its own task_NN/ subdirectory and a seed range no other task uses. A
# failed task is rerun on its own, and rerunning the array skips finished tasks
# through tools/generate_data.py's resume markers.
#
# Setup, before sbatch:
#   1. Build once, on the login node:
#        cd CHANGE_ME_REPO_ROOT
#        dotnet build Bots/Bots.csproj -c Release
#        dotnet build GameRunner/GameRunner.csproj -c Release
#   2. Create the logs directory. SLURM does not create the directory for
#      #SBATCH --output/--error, and every task fails before the script runs if
#      it is missing:
#        mkdir -p CHANGE_ME_REPO_ROOT/logs
#   3. Replace every remaining CHANGE_ME_* placeholder below (partition, account,
#      repo root, games per task); the guard below refuses to run otherwise.
#      OUT_DIR, SEED_BASE, BOT_NAME and --time have defaults for the
#      SakkirinaGenNeural run.
#   4. Submit:
#        sbatch scripts/slurm_generate.sh
#
# Total games = 32 * GAMES_PER_TASK; choose it together with --time.
# tools/generate_data.sh plays at full strength (--timeout 10). SakkirinaGenNeural
# runs ONNX inference on every rollout and took ~79.6s/game locally, against
# SakkirinaGen's ~48s on cluster hardware. At ~6000 games, 32 tasks at ~80s/game
# take roughly 4-5 hours, inside the 10-hour --time below.

#SBATCH --job-name=sakgen
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=10:00:00
#SBATCH --output=CHANGE_ME_REPO_ROOT/logs/sakgen_%A_%a.out
#SBATCH --error=CHANGE_ME_REPO_ROOT/logs/sakgen_%A_%a.err

set -euo pipefail

# --- Fill in before submitting ---
REPO_ROOT="CHANGE_ME_REPO_ROOT"          # e.g. /home/you/tot/deepsets-tales-of-tribute
GAMES_PER_TASK="CHANGE_ME_GAMES_PER_TASK"  # integer, e.g. 300 -- total games = 32 * this

# --- Defaults for the SakkirinaGenNeural run; editing is optional ---
OUT_DIR="/hpcwork/yfl79180/tot_data_neural"  # apart from the SakkirinaGen run's
                                              # /hpcwork/yfl79180/tot_data
SEED_BASE="20260807"                     # the SakkirinaGen run used 20260803;
                                          # must differ or the games repeat.
                                          # generate_data.sh adds task_id*1000000
                                          # per task. Never time-derived, so a
                                          # retried task keeps its seed range
BOT_NAME="SakkirinaGenNeural"            # its ONNX model is pinned below via
                                          # EXPECT_ONNX_SHA256
EXPECT_ONNX_SHA256="71d999201b57974477f9ef1b57eb681a4ce5e54aea52293766378f93b8077fd6"
# ----------------------------------

# Refuse to run with an unedited placeholder instead of failing later (e.g. on a
# non-numeric GAMES_PER_TASK). Placeholders in #SBATCH lines are rejected by
# sbatch itself; this catches the shell variables.
for name in REPO_ROOT GAMES_PER_TASK; do
  value="${!name}"
  if [[ "$value" == CHANGE_ME_* ]]; then
    echo "ERROR: $name was never edited from its placeholder value ($value)." >&2
    echo "       Edit every CHANGE_ME_* placeholder in this script before submitting." >&2
    exit 1
  fi
done

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID starting on $(hostname)"
echo "REPO_ROOT=$REPO_ROOT  OUT_DIR=$OUT_DIR  GAMES_PER_TASK=$GAMES_PER_TASK  SEED_BASE=$SEED_BASE  BOT_NAME=$BOT_NAME"

exec "$REPO_ROOT/tools/generate_data.sh" \
  --bot "$BOT_NAME" \
  --games "$GAMES_PER_TASK" \
  --out-dir "$OUT_DIR" \
  --seed-base "$SEED_BASE" \
  --expect-onnx-sha256 "$EXPECT_ONNX_SHA256" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --skip-build
