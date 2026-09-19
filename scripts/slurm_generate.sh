#!/usr/bin/env bash
# SLURM array template for cluster-scale self-play data generation. Default
# bot is SakkirinaGenNeural, our current best agent (68% vs SakkirinaSolo) --
# see BOT_NAME below to generate from a different bot instead (e.g.
# SakkirinaGen, the 2025 winner's heuristic, for the earlier dataset).
#
# DESIGN: 32 independent single-core array tasks, NOT one 32-process job.
# Each task is its own OS process (tools/generate_data.sh --task-id N forces
# --jobs 1), writing to its own task_NN/ subdirectory with a seed range that
# can never collide with any other task's. This is deliberately simpler than
# a single multi-process SLURM job: if task 17 dies, only task 17 needs
# re-running (SLURM's own array retry, or a manual re-submit of just that
# index), and the resumability tools/generate_data.py already has means
# re-running the same array a second time skips every task that already
# finished.
#
# SETUP (do this BEFORE sbatch-ing, not after):
#   1. Build once, on the login node:
#        cd CHANGE_ME_REPO_ROOT
#        dotnet build Bots/Bots.csproj -c Release
#        dotnet build GameRunner/GameRunner.csproj -c Release
#   2. Create the logs directory yourself -- SLURM does NOT create the
#      directory for #SBATCH --output/--error; if it doesn't already exist
#      when this is submitted, every task fails immediately before the
#      script body even runs:
#        mkdir -p CHANGE_ME_REPO_ROOT/logs
#   3. Replace every remaining CHANGE_ME_* placeholder below (partition,
#      account, repo root, games-per-task). The script refuses to run (see
#      the guard below) if any are left unedited. OUT_DIR, SEED_BASE, BOT_NAME
#      and --time already have real defaults for this (the second,
#      SakkirinaGenNeural) cluster run -- edit them too if you want different
#      ones, but they won't block submission if left as-is.
#   4. Submit:
#        sbatch scripts/slurm_generate.sh
#
# Total games generated = 32 * GAMES_PER_TASK. Pick GAMES_PER_TASK and --time
# together: tools/generate_data.sh runs the bot at full strength (--timeout
# 10, no speed shortcuts), so budget generously per game -- see the per-game
# peak-memory/wall-clock measurement notes from the local dry run before
# picking a --time value, rather than guessing. SakkirinaGenNeural runs ONNX
# inference on every rollout, so it is noticeably slower than SakkirinaGen was
# (~79.6s/game measured locally vs. SakkirinaGen's ~48s on cluster hardware --
# expect somewhere in that range, not directly comparable since local and
# cluster hardware differ). At the default GAMES_PER_TASK below: ~6000 games /
# 32 tasks * ~80s/game is roughly 4-5 hours -- comfortably inside the 10-hour
# --time below, not up against it.

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
REPO_ROOT="CHANGE_ME_REPO_ROOT"          # e.g. /home/you/ScriptsOfTribute-Core
GAMES_PER_TASK="CHANGE_ME_GAMES_PER_TASK"  # integer, e.g. 300 -- total games = 32 * this

# --- Have real defaults for this (second, SakkirinaGenNeural) run -- edit if
# you want different ones, but submission won't be blocked if you don't ---
OUT_DIR="/hpcwork/yfl79180/tot_data_neural"  # separate from the first run's
                                              # /hpcwork/yfl79180/tot_data, so
                                              # the two datasets cannot mix --
                                              # shard filenames already embed
                                              # the bot name too, so both are
                                              # self-describing either way
SEED_BASE="20260807"                     # the first (SakkirinaGen) run used
                                          # 20260803 -- must differ, or this
                                          # run would just regenerate
                                          # identical games. Fixed across the
                                          # WHOLE array (tools/generate_data.sh
                                          # adds task_id*1000000 per task); do
                                          # not use a time-derived value here,
                                          # or a re-submitted/retried task
                                          # could silently regenerate a
                                          # different seed range than before
BOT_NAME="SakkirinaGenNeural"            # our current best agent; the ONNX
                                          # model it needs is pinned below via
                                          # EXPECT_ONNX_SHA256, not just
                                          # assumed present
EXPECT_ONNX_SHA256="71d999201b57974477f9ef1b57eb681a4ce5e54aea52293766378f93b8077fd6"
# ----------------------------------

# Refuse to run with an unedited placeholder rather than fail confusingly
# later (a non-numeric GAMES_PER_TASK, or a #SBATCH directive SLURM already
# rejected at submission time -- some placeholders can't even reach this
# point, but the plain shell variables below can).
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
