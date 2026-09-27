#!/usr/bin/env bash
#SBATCH --job-name=deepsets_expb
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/expb_%A_%a.out
#SBATCH --error=logs/expb_%A_%a.err
# Runs several experiment tasks in sequence per array task, by calling the
# stock slurm_experiment.sh once per task id. Needed because the default
# account allows only 100 submitted jobs. Array must be 0..N-1 (contiguous).
set -uo pipefail
REPO_ROOT="$HOME/tot/deepsets-tales-of-tribute"
START_TASK="${START_TASK:-0}"
TOTAL_TASKS="${TOTAL_TASKS:?set TOTAL_TASKS via --export}"
STRIDE="$SLURM_ARRAY_TASK_COUNT"
echo "Batch $SLURM_ARRAY_TASK_ID/$STRIDE config=$SOT_EXP_CONFIG start=$START_TASK total=$TOTAL_TASKS on $(hostname)"
for (( tid=START_TASK+SLURM_ARRAY_TASK_ID; tid<TOTAL_TASKS; tid+=STRIDE )); do
  SLURM_ARRAY_TASK_ID="$tid" bash "$REPO_ROOT/scripts/slurm_experiment.sh" \
    || echo "WARN: task_id $tid exited nonzero"
done
echo "Batch $SLURM_ARRAY_TASK_ID done"
