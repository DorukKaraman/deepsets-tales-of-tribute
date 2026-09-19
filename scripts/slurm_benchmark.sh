#!/usr/bin/env bash
# SLURM array template for the cluster-scale DeepSets benchmark: 10 matchups x
# 400 games = 4000 games, one game per array task. See
# tools/benchmark_cluster.py for the exact matchup list and task-id layout,
# and tools/aggregate_benchmark_results.py to summarize results afterward.
#
# DESIGN: one array task = one game = one process, NOT one process playing
# many games. GameRunner reuses a bot instance across --runs N, and its
# GameEndStatsCounter only reports aggregate counts with no way to tell which
# reason produced which winner across multiple games -- --runs 1 per process
# is what lets every game's outcome be attributed exactly (same reasoning
# tools/benchmark_runner.py already uses locally, and tools/generate_data.py's
# job-per-task pattern for resumability). This also means the concurrency
# here comes entirely from SLURM running multiple array elements at once
# (bounded by the %THROTTLE below), not from anything inside this script.
#
# CLUSTER DETAILS (RWTH CLAIX, filled in already -- nothing to edit here
# unless your setup differs):
#   partition c23ms, 1 core/task, ~2GB/core, no --account needed, .NET via
#   `source $HOME/tot/env.sh`, repo at $HOME/tot/ScriptsOfTribute-Core.
#
# ARRAY SIZE / CHUNKING: 4000 tasks may exceed this cluster's configured
# MaxArraySize (check with `scontrol show config | grep -i MaxArraySize` on
# the login node -- this is a site-wide setting, not something this script
# can see or work around). The #SBATCH --array line below defaults to a safe
# 0-999 chunk for that reason. If MaxArraySize >= 4000, just override it at
# submission time:
#   sbatch --array=0-3999%32 scripts/slurm_benchmark.sh
# Otherwise, submit in chunks (4 shown, adjust chunk size/count to whatever
# MaxArraySize actually allows -- each chunk is a fully independent
# submission, no coordination between them needed):
#   sbatch --array=0-999%32    scripts/slurm_benchmark.sh
#   sbatch --array=1000-1999%32 scripts/slurm_benchmark.sh
#   sbatch --array=2000-2999%32 scripts/slurm_benchmark.sh
#   sbatch --array=3000-3999%32 scripts/slurm_benchmark.sh
# The %32 throttle (max 32 concurrently running array elements) matches the
# "4000 games on 32 concurrent cores is roughly 2-3 hours" budget this
# script's --time below assumes -- raise or lower it to match your actual
# allocation/fair-share limits.
#
# RESUMABLE. Every task writes a result JSON file only after its one game
# completes successfully (see tools/benchmark_cluster.py). A task whose
# result file already exists is skipped immediately without running
# anything, so re-submitting any of the chunks above (or the whole thing) is
# always safe -- only genuinely missing/failed tasks do any work.
#
# SETUP (do this BEFORE sbatch-ing, not after):
#   1. source $HOME/tot/env.sh
#      cd $HOME/tot/ScriptsOfTribute-Core
#      git pull
#      dotnet build Bots/Bots.csproj -c Release
#      dotnet build GameRunner/GameRunner.csproj -c Release
#   2. Create the logs directory yourself -- SLURM does NOT create the
#      directory for #SBATCH --output/--error; if it doesn't already exist
#      when this is submitted, every task fails immediately before the
#      script body even runs:
#        mkdir -p $HOME/tot/ScriptsOfTribute-Core/logs
#   3. Verify the onnx sha256 by hand once (tools/benchmark_cluster.sh
#      re-verifies it on every single task anyway, but see it fail loudly
#      here first rather than 4000 times in an array log). All four
#      benchmarked agents load the same file:
#        shasum -a 256 GameRunner/bin/Release/net8.0/DeepSetsValueNetwork.onnx
#      must print 86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915
#   4. Preview the plan without running anything:
#        tools/benchmark_cluster.sh --out-dir "$OUT_DIR" --seed-base "$SEED_BASE" --dry-run
#   5. Submit (see ARRAY SIZE / CHUNKING above for the exact command).
#
# Each game runs ~60-90s (matches the local wall-clock measured for this bot
# family). 4000 games / 32 concurrent %-throttled tasks x ~75s/game is
# roughly 2.5 hours of wall clock to drain the whole array -- that figure is
# about the ARRAY as a whole, not any single task, and isn't set directly
# anywhere; --time below is a PER-TASK limit (one game), generous margin over
# the ~60-90s/game estimate.

#SBATCH --job-name=deepsets_bench
#SBATCH --partition=c23ms
#SBATCH --array=0-999%32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err

# --output/--error above are RELATIVE (to wherever `sbatch` is invoked from)
# on purpose: #SBATCH directives are parsed by sbatch itself and do not
# reliably expand shell variables like $HOME, so an absolute path would need
# a literal, pre-resolved home directory hardcoded here instead. This is why
# the SETUP steps above have you `cd $HOME/tot/ScriptsOfTribute-Core` before
# sbatch-ing -- submit from anywhere else and the logs/ directory (and its
# mkdir -p in SETUP step 2) needs to be wherever you actually ran sbatch from.

set -euo pipefail

# --- Edit if your setup differs from the CLUSTER DETAILS above ---
REPO_ROOT="$HOME/tot/ScriptsOfTribute-Core"
OUT_DIR="$HOME/tot/benchmark_results"    # small JSON files, not bulk data --
                                          # $HOME is fine here, unlike the
                                          # /hpcwork/... paths the data-gen
                                          # runs used for multi-GB shards
SEED_BASE="20260808"                     # fixed across the WHOLE array and
                                          # every resubmission/chunk above --
                                          # do not use a time-derived value,
                                          # or a retried task could silently
                                          # regenerate a different game than
                                          # originally planned. Distinct from
                                          # both data-gen runs' seed bases
                                          # (20260803, 20260807) though that's
                                          # not load-bearing here -- different
                                          # scripts, different seed spaces.
EXPECT_ONNX_SHA256="86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915"
# -------------------------------------------------------------

source "$HOME/tot/env.sh"

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID starting on $(hostname)"
echo "REPO_ROOT=$REPO_ROOT  OUT_DIR=$OUT_DIR  SEED_BASE=$SEED_BASE"

exec "$REPO_ROOT/tools/benchmark_cluster.sh" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --out-dir "$OUT_DIR" \
  --seed-base "$SEED_BASE" \
  --expect-onnx-sha256 "$EXPECT_ONNX_SHA256" \
  --skip-build
