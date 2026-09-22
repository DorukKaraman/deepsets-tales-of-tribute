#!/usr/bin/env bash
# SLURM array template: train one value network PER SEED from one shared
# dataset, export each to ONNX, and leave every artefact under its own
# per-seed directory in $HPCWORK.
#
# WHY SEEDS: a single trained model's win rate is one sample from a
# distribution, and reporting it as if it were the distribution is the most
# common way a result like this fails to replicate. Training N models that
# differ ONLY in seed -- same data, same split, same hyperparameters -- and
# benchmarking each gives the spread that belongs in the paper alongside the
# mean. training/train_local.py --seed seeds torch, numpy, python random and
# the shuffle buffer, so two tasks here differ in weight initialisation and
# data order and in nothing else.
#
# ONE TASK = ONE SEED = ONE MODEL. Tasks share the dataset read-only and write
# to disjoint directories, so there is no coordination between them and a
# failed seed is re-run by resubmitting just that array index.
#
# NOTHING GOES TO /tmp. Cluster /tmp is node-local, small, and wiped when the
# job ends -- a checkpoint written there is gone before you can fetch it, and a
# multi-GB dataset staged there can fill the node for everyone else. This
# script refuses to run if $HPCWORK is unset rather than silently falling back,
# and it points TMPDIR at $HPCWORK too so that pip, torch and matplotlib's
# scratch files land somewhere with a real quota.
#
# CPU COUNT: unlike the benchmark arrays (1 core/task, because one game is
# single-threaded by construction), training is a BLAS workload and 1 core
# would be pointlessly slow. This asks for CPUS_PER_TASK cores and tells torch
# to use exactly that many -- not more, which on a shared node means fighting
# other jobs for cores the scheduler never gave you. Adjust to your allocation.
#
# SETUP (do this BEFORE sbatch-ing, not after):
#   1. Build the Python environment once, on the login node:
#        ./scripts/setup_python_env.sh --venv-dir "$HPCWORK/tot_venv"
#      It pins torch 2.2.2, which is the version that reproduces the shipped
#      ONNX byte hash -- see that script's header.
#   2. Have a split dataset ready (tools/split_dataset.py output, i.e. a
#      directory containing train/ and val/). Set DATA_DIR below.
#   3. mkdir -p CHANGE_ME_REPO_ROOT/logs
#      SLURM does NOT create the directory for #SBATCH --output/--error.
#   4. Replace every CHANGE_ME_* placeholder below. The guard further down
#      refuses to run if any are left.
#   5. sbatch scripts/slurm_train.sh
#
# AFTERWARDS, each seed leaves:
#   $HPCWORK/tot_models/seed_NN/best_model.pth
#   $HPCWORK/tot_models/seed_NN/deepsets_value_network_epochK.pth
#   $HPCWORK/tot_models/seed_NN/training_metrics.json
#   $HPCWORK/tot_models/seed_NN/run_config.json        (seed + hyperparameters)
#   $HPCWORK/tot_models/seed_NN/DeepSetsValueNetwork_seed_NN.onnx
#   $HPCWORK/tot_models/seed_NN/SHA256SUMS
#
# Then, to use a seed's model in an experiment: add its ONNX sha256 to the
# config's allowed_onnx_sha256 and point the matchup at it with
# SOT_MODEL_PATH -- tools/benchmark_cluster.py verifies both before it will run
# a single game. To compare the seeds as MODELS rather than as agents, score
# them all on one held-out set with tools/evaluate_checkpoints.py.

#SBATCH --job-name=deepsets_train
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

# --- Fill in before submitting ---
REPO_ROOT="CHANGE_ME_REPO_ROOT"        # e.g. /home/you/tot/ScriptsOfTribute-Core
DATA_DIR="CHANGE_ME_SPLIT_DATA_DIR"    # tools/split_dataset.py output: contains train/ and val/

# --- Have real defaults; edit if you want different ones ---
VENV_DIR="${SOT_VENV_DIR:-$HPCWORK/tot_venv}"
OUT_ROOT="$HPCWORK/tot_models"
EPOCHS=3            # what the shipped model used
BATCH_SIZE=256      # what the shipped model used
LR=5e-4             # what the shipped model used
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
# DataLoader workers. One fewer than the allocation, so the main process (which
# does the actual optimizer step) is not competing with its own workers for the
# last core.
NUM_WORKERS=$((CPUS_PER_TASK > 1 ? CPUS_PER_TASK - 1 : 0))
# -------------------------------------------------------------

for name in REPO_ROOT DATA_DIR; do
  value="${!name}"
  if [[ "$value" == CHANGE_ME_* ]]; then
    echo "ERROR: $name was never edited from its placeholder value ($value)." >&2
    echo "       Edit every CHANGE_ME_* placeholder in this script before submitting." >&2
    exit 1
  fi
done

# $HPCWORK is the only place outputs may go. No fallback on purpose: a fallback
# is how training artefacts end up on node-local scratch and disappear.
if [ -z "${HPCWORK:-}" ]; then
  echo "ERROR: \$HPCWORK is not set." >&2
  echo "       Every output of this script goes under \$HPCWORK. There is deliberately no" >&2
  echo "       fallback -- /tmp on a compute node is node-local and is wiped when the job" >&2
  echo "       ends, so checkpoints written there would be gone before you could fetch them." >&2
  echo "       Set HPCWORK to a directory with a real quota and resubmit." >&2
  exit 1
fi

SEED="$SLURM_ARRAY_TASK_ID"
SEED_TAG="$(printf 'seed_%02d' "$SEED")"
OUT_DIR="$OUT_ROOT/$SEED_TAG"
mkdir -p "$OUT_DIR"

# Keep every scratch file off /tmp too -- pip, torch extensions and matplotlib
# all write there by default, and on a shared node that is somebody else's
# problem as much as yours.
export TMPDIR="$HPCWORK/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/mpl"

# Match the thread count to the allocation. Left unset, OpenMP sizes itself
# from the machine's total core count, not the cgroup's -- so an 8-core
# allocation on a 96-core node spawns 96 threads that then thrash.
export OMP_NUM_THREADS="$CPUS_PER_TASK"
export MKL_NUM_THREADS="$CPUS_PER_TASK"

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID starting on $(hostname)"
echo "SEED=$SEED  OUT_DIR=$OUT_DIR  DATA_DIR=$DATA_DIR"
echo "CPUS_PER_TASK=$CPUS_PER_TASK  NUM_WORKERS=$NUM_WORKERS  TMPDIR=$TMPDIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: no Python environment at $VENV_DIR." >&2
  echo "       Build it once on the login node first:" >&2
  echo "         ./scripts/setup_python_env.sh --venv-dir \"$VENV_DIR\"" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

for sub in train val; do
  if [ ! -d "$DATA_DIR/$sub" ]; then
    echo "ERROR: $DATA_DIR/$sub does not exist." >&2
    echo "       DATA_DIR must be tools/split_dataset.py's output directory, which contains" >&2
    echo "       train/ and val/ subdirectories. Pointing this at raw generated shards would" >&2
    echo "       train and validate on the same games." >&2
    exit 1
  fi
done

echo
echo "=== Training (seed $SEED) ==="
python "$REPO_ROOT/training/train_local.py" \
  --train-dir "$DATA_DIR/train" \
  --val-dir "$DATA_DIR/val" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"

BEST_MODEL="$OUT_DIR/best_model.pth"
if [ ! -f "$BEST_MODEL" ]; then
  echo "ERROR: training finished but $BEST_MODEL does not exist." >&2
  exit 1
fi

ONNX_OUT="$OUT_DIR/DeepSetsValueNetwork_${SEED_TAG}.onnx"
echo
echo "=== Exporting to ONNX ==="
# export_to_onnx.py verifies the exported graph against the PyTorch model across
# a range of node counts and refuses to write one whose outputs differ by more
# than 1e-5, so a silent export bug cannot reach the benchmark.
( cd "$REPO_ROOT/training" && python export_to_onnx.py --checkpoint "$BEST_MODEL" --out "$ONNX_OUT" )

echo
echo "=== Recording hashes ==="
( cd "$OUT_DIR" && sha256sum ./*.pth ./*.onnx > SHA256SUMS && cat SHA256SUMS )

cat <<EOF

=== Seed $SEED done ===
Artefacts: $OUT_DIR

To benchmark this seed's model, add its ONNX sha256 (above) to the
allowed_onnx_sha256 list in the experiment config, and point the matchup at it:

  "env": {"SOT_MODEL_PATH": "$ONNX_OUT"}

tools/benchmark_cluster.py checks that the file exists AND that its hash is on
the allowed list before it runs a game -- a bot whose model fails to load does
not crash, it silently falls back to a heuristic evaluator.

To compare the seeds as models rather than as agents, score them all on one
held-out set:

  python $REPO_ROOT/tools/evaluate_checkpoints.py \\
      $OUT_ROOT/seed_*/best_model.pth \\
      --data-dir <held-out data dir>
EOF
