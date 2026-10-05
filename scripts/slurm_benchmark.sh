#!/usr/bin/env bash
# SLURM array template for the cluster DeepSets benchmark, one game per array
# task. The matchups, game count and --timeout are in
# experiments/configs/legacy_paper_benchmark.json: 12 matchups x 400 games = 4800
# tasks (ids 0-4799). See tools/benchmark_cluster.py for the config format and
# task-id layout, and tools/aggregate_benchmark_results.py (same --config) to
# summarise results.
#
# The other paper experiments run through scripts/slurm_experiment.sh, which
# takes any config in experiments/configs/. This script stays on the legacy
# config as the reproduction path for numbers already reported.
#
# One task = one game = one process. GameRunner reuses a bot instance across
# --runs N, and GameEndStatsCounter reports only aggregate counts, so --runs 1
# per process is what attributes each game's outcome exactly (as
# tools/benchmark_runner.py does locally). Concurrency comes from SLURM running
# array elements in parallel, bounded by the %throttle.
#
# Cluster details (RWTH CLAIX; edit only if yours differ): partition c23ms,
# 1 core/task, ~2GB/core, no --account, .NET via `source $HOME/tot/env.sh`, repo
# at $HOME/tot/deepsets-tales-of-tribute.
#
# Array size: 4800 tasks may exceed the site's MaxArraySize (check with
# `scontrol show config | grep -i MaxArraySize` on the login node). The #SBATCH
# --array line below defaults to a 0-999 chunk. If MaxArraySize allows, override
# at submission:
#   sbatch --array=0-4799%32 scripts/slurm_benchmark.sh
# Otherwise submit independent chunks sized to MaxArraySize:
#   sbatch --array=0-999%32    scripts/slurm_benchmark.sh
#   sbatch --array=1000-1999%32 scripts/slurm_benchmark.sh
#   sbatch --array=2000-2999%32 scripts/slurm_benchmark.sh
#   sbatch --array=3000-3999%32 scripts/slurm_benchmark.sh
#   sbatch --array=4000-4799%32 scripts/slurm_benchmark.sh
# If the account limits submitted jobs, use scripts/slurm_experiment_batched.sh
# instead (REPRODUCE.md section 6). %32 caps concurrently running elements; match
# it to your allocation and fair-share limits.
#
# Resumable: a task writes its result JSON only after its game completes (see
# tools/benchmark_cluster.py), and a task whose result file exists is skipped, so
# resubmitting any chunk is safe.
#
# Setup, before sbatch:
#   1. Clone into a new directory (an existing $HOME/tot/ScriptsOfTribute-Core
#      checkout is a different repository without this harness):
#      source $HOME/tot/env.sh
#        git clone -b experiments \
#            https://github.com/DorukKaraman/deepsets-tales-of-tribute.git \
#            $HOME/tot/deepsets-tales-of-tribute
#      cd $HOME/tot/deepsets-tales-of-tribute
#      dotnet build Bots/Bots.csproj -c Release
#      dotnet build GameRunner/GameRunner.csproj -c Release
#   2. Create the logs directory. SLURM does not create the directory for
#      #SBATCH --output/--error, and every task fails before the script runs if
#      it is missing:
#        mkdir -p $HOME/tot/deepsets-tales-of-tribute/logs
#   3. Check the onnx sha256 once by hand (tools/benchmark_cluster.sh checks it on
#      every task, but one failure here beats one per task). All four benchmarked
#      agents load the same file:
#        shasum -a 256 GameRunner/bin/Release/net8.0/DeepSetsValueNetwork.onnx
#      must print 86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915
#   4. Preview the plan without running anything:
#        tools/benchmark_cluster.sh --out-dir "$OUT_DIR" --seed-base "$SEED_BASE" --dry-run
#   5. Submit (see Array size above).
#
# A game takes ~60-90s. --time below is per task (one game), with a generous
# margin.

#SBATCH --job-name=deepsets_bench
#SBATCH --partition=c23ms
#SBATCH --array=0-999%32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err

# --output/--error are relative to the directory sbatch runs in, because sbatch
# does not reliably expand shell variables like $HOME in #SBATCH directives.
# Submit from $HOME/tot/deepsets-tales-of-tribute, where SETUP step 2 creates logs/.

set -euo pipefail

# --- Edit if your setup differs from the CLUSTER DETAILS above ---
REPO_ROOT="$HOME/tot/deepsets-tales-of-tribute"
OUT_DIR="$HOME/tot/benchmark_results"    # small JSON files, so $HOME is fine
SEED_BASE="20260808"                     # fixed across the whole array and every
                                          # resubmission, so a retried task replays
                                          # the same game; never time-derived
EXPECT_ONNX_SHA256="86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915"
# -------------------------------------------------------------

source "$HOME/tot/env.sh"

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID starting on $(hostname)"
echo "REPO_ROOT=$REPO_ROOT  OUT_DIR=$OUT_DIR  SEED_BASE=$SEED_BASE"

exec "$REPO_ROOT/tools/benchmark_cluster.sh" \
  --config legacy_paper_benchmark \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --out-dir "$OUT_DIR" \
  --seed-base "$SEED_BASE" \
  --allow-onnx-sha256 "$EXPECT_ONNX_SHA256" \
  --skip-build
