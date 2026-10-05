#!/usr/bin/env bash
# SLURM array template: train one value network per seed from one shared
# dataset, export each to ONNX, and keep every artefact in its own per-seed
# directory under $HPCWORK.
#
# One model's win rate is a single sample, so the paper reports the spread over
# models that differ only in seed. training/train_local.py --seed seeds torch,
# numpy, Python's random and the shuffle buffer, so tasks differ only in weight
# initialisation and data order.
#
# One task = one seed = one model. Tasks read the dataset and write to disjoint
# directories; a failed seed is rerun by resubmitting its array index.
#
# Nothing goes to /tmp, which on the cluster is node-local, small and wiped when
# the job ends. The script refuses to run without $HPCWORK and points TMPDIR
# there too.
#
# Training is a BLAS workload, so it asks for CPUS_PER_TASK cores (unlike the
# 1-core benchmark arrays) and tells torch to use exactly that many. Adjust to
# your allocation.
#
# Setup, before sbatch:
#   1. Build the Python environment once, on the login node:
#        ./scripts/setup_python_env.sh --venv-dir "$HPCWORK/tot_venv"
#      It pins torch 2.2.2, the version that reproduces the shipped ONNX byte
#      hash; see that script's header.
#   2. Have a split dataset ready (tools/split_dataset.py output, a directory
#      containing train/ and val/). Set DATA_DIR below.
#   3. mkdir -p CHANGE_ME_REPO_ROOT/logs
#      SLURM does not create the directory for #SBATCH --output/--error.
#   4. Replace every CHANGE_ME_* placeholder below; the guard further down
#      refuses to run if any are left.
#   5. sbatch scripts/slurm_train.sh
#
# Network and output location are set by two environment variables:
#
#   ARCH=deepsets       (default) train_local.py + export_to_onnx.py
#   ARCH=matched                  train_flat.py --arch matched        + export_flat_to_onnx.py
#   ARCH=wide                     train_flat.py --arch wide           + export_flat_to_onnx.py
#   ARCH=matched_sorted           train_flat.py --arch matched_sorted + export_flat_to_onnx.py
#   OUT_ROOT=...        (default $HPCWORK/tot_models)
#
# The flat arms are the flat-MLP ablation (REPRODUCE.md section 8). Give them
# their own OUT_ROOT: the output path depends only on the array index, so an
# ablation run at --seed 0 would otherwise land on the paper's seed_00:
#
#   ARCH=matched        OUT_ROOT="$HPCWORK/tot_ablation/matched"        sbatch scripts/slurm_train.sh
#   ARCH=wide           OUT_ROOT="$HPCWORK/tot_ablation/wide"           sbatch scripts/slurm_train.sh
#   ARCH=matched_sorted OUT_ROOT="$HPCWORK/tot_ablation/matched_sorted" sbatch scripts/slurm_train.sh
#
# matched_sorted is matched with the node rows put in a canonical order before
# flattening (same widths, same 72,549 parameters). The unsorted flat models are
# not permutation-invariant while the search reshuffles hidden piles on every
# determinisation, so their game results mix evaluator quality with the lack of
# invariance.
#
# The script refuses to start if the seed directory already holds a
# best_model.pth, unless FORCE=1.
#
# Each seed leaves (paths for the default OUT_ROOT):
#   $HPCWORK/tot_models/seed_NN/best_model.pth
#   $HPCWORK/tot_models/seed_NN/deepsets_value_network_epochK.pth
#     (flat arms: flat_matched_value_network_epochK.pth / flat_wide_...)
#   $HPCWORK/tot_models/seed_NN/training_metrics.json
#   $HPCWORK/tot_models/seed_NN/run_config.json        (seed + hyperparameters,
#     plus arch/widths/input dim on the flat arms)
#   $HPCWORK/tot_models/seed_NN/DeepSetsValueNetwork_seed_NN.onnx
#     (flat arms: FlatValueNetwork_<arch>_seed_NN.onnx)
#   $HPCWORK/tot_models/seed_NN/SHA256SUMS
#
# To use a seed's model in an experiment, add its ONNX sha256 to the config's
# allowed_onnx_sha256 and point the matchup at it with SOT_MODEL_PATH;
# tools/benchmark_cluster.py checks both before running a game. To compare seeds
# as models rather than agents, score them on one held-out set with
# tools/evaluate_checkpoints.py.

#SBATCH --job-name=deepsets_train
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
# 24G: the shuffle buffer (training/stream_dataset.py,
# DEFAULT_SHUFFLE_BUFFER_SIZE = 100,000 parsed graphs) dominates memory, so this
# scales with that constant, not with dataset size. 16G ran at its ceiling.
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

# --- Fill in before submitting ---
REPO_ROOT="CHANGE_ME_REPO_ROOT"        # e.g. /home/you/tot/deepsets-tales-of-tribute
DATA_DIR="CHANGE_ME_SPLIT_DATA_DIR"    # tools/split_dataset.py output: contains train/ and val/

# --- Have real defaults; edit if you want different ones ---
VENV_DIR="${SOT_VENV_DIR:-$HPCWORK/tot_venv}"
# ARCH selects the network this array trains (default deepsets):
#   deepsets  training/train_local.py    + training/export_to_onnx.py
#   matched   training/train_flat.py     + training/export_flat_to_onnx.py  (72,549 params)
#   wide      training/train_flat.py     + training/export_flat_to_onnx.py  (1,649,409 params)
# matched/wide are the flat-MLP ablation; see REPRODUCE.md section 8.
ARCH="${ARCH:-deepsets}"
# OUT_ROOT keeps the ablation out of the directory holding the paper's per-seed
# models:
#   ARCH=matched OUT_ROOT="$HPCWORK/tot_ablation/matched" sbatch scripts/slurm_train.sh
OUT_ROOT="${OUT_ROOT:-$HPCWORK/tot_models}"
EPOCHS=3            # what the shipped model used
BATCH_SIZE=256      # what the shipped model used
LR=5e-4             # what the shipped model used
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
# DataLoader workers: one fewer than the allocation, leaving a core for the main
# process, which runs the optimizer step.
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

# Outputs go only to $HPCWORK. There is no fallback, since a fallback is how
# artefacts end up on node-local scratch and disappear.
if [ -z "${HPCWORK:-}" ]; then
  echo "ERROR: \$HPCWORK is not set." >&2
  echo "       Every output of this script goes under \$HPCWORK. There is deliberately no" >&2
  echo "       fallback -- /tmp on a compute node is node-local and is wiped when the job" >&2
  echo "       ends, so checkpoints written there would be gone before you could fetch them." >&2
  echo "       Set HPCWORK to a directory with a real quota and resubmit." >&2
  exit 1
fi

case "$ARCH" in
  deepsets|matched|wide|matched_sorted) ;;
  *)
    echo "ERROR: ARCH=$ARCH is not one of: deepsets, matched, wide, matched_sorted." >&2
    exit 1
    ;;
esac

SEED="$SLURM_ARRAY_TASK_ID"
SEED_TAG="$(printf 'seed_%02d' "$SEED")"
OUT_DIR="$OUT_ROOT/$SEED_TAG"

# Refuse to overwrite an existing trained model. The output path depends only on
# the array index, and the paper's per-seed models live at
# $HPCWORK/tot_models/seed_00..04. A rerun at the same index would replace
# best_model.pth without warning, and the directory would no longer match the
# hashes in experiments/configs/seed_benchmark.json. FORCE=1 overrides.
if [ -f "$OUT_DIR/best_model.pth" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "ERROR: $OUT_DIR/best_model.pth already exists." >&2
  echo "       This directory holds a trained model. Overwriting it would replace a" >&2
  echo "       checkpoint that something may depend on -- the per-seed models under" >&2
  echo "       \$HPCWORK/tot_models are pinned by hash in" >&2
  echo "       experiments/configs/seed_benchmark.json." >&2
  echo >&2
  echo "       Either point OUT_ROOT somewhere else:" >&2
  echo "         ARCH=$ARCH OUT_ROOT=\"\$HPCWORK/tot_ablation/$ARCH\" sbatch \$0" >&2
  echo "       or, if you really mean to replace it, set FORCE=1." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Keep scratch files off /tmp too; pip, torch extensions and matplotlib write
# there by default.
export TMPDIR="$HPCWORK/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/mpl"

# Match the thread count to the allocation. Unset, OpenMP sizes itself from the
# machine's core count, not the cgroup's, so an 8-core allocation on a 96-core
# node spawns 96 threads.
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
echo "=== Training (arch $ARCH, seed $SEED) ==="
if [ "$ARCH" = "deepsets" ]; then
  python "$REPO_ROOT/training/train_local.py" \
    --train-dir "$DATA_DIR/train" \
    --val-dir "$DATA_DIR/val" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"
else
  python "$REPO_ROOT/training/train_flat.py" \
    --arch "$ARCH" \
    --train-dir "$DATA_DIR/train" \
    --val-dir "$DATA_DIR/val" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"
fi

BEST_MODEL="$OUT_DIR/best_model.pth"
if [ ! -f "$BEST_MODEL" ]; then
  echo "ERROR: training finished but $BEST_MODEL does not exist." >&2
  exit 1
fi

echo
echo "=== Exporting to ONNX ==="
# Both exporters verify the graph against the PyTorch model across a range of
# node counts (atol+rtol) and write to a temp file renamed into place only once
# it passes, so a failed export leaves no unverified .onnx behind.
if [ "$ARCH" = "deepsets" ]; then
  ONNX_OUT="$OUT_DIR/DeepSetsValueNetwork_${SEED_TAG}.onnx"
  ( cd "$REPO_ROOT/training" && python export_to_onnx.py --checkpoint "$BEST_MODEL" --out "$ONNX_OUT" )
else
  ONNX_OUT="$OUT_DIR/FlatValueNetwork_${ARCH}_${SEED_TAG}.onnx"
  ( cd "$REPO_ROOT/training" && python export_flat_to_onnx.py --arch "$ARCH" --checkpoint "$BEST_MODEL" --out "$ONNX_OUT" )
fi

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
