#!/usr/bin/env bash
# Create the CPU-only Python environment the training/ and tools/ scripts need,
# with every version pinned.
#
# WHY PINNED, AND WHY THIS PARTICULAR TORCH
# An ONNX file embeds producer_version, so exporting the same checkpoint on a
# different PyTorch version produces a byte-different file with a different
# SHA-256, even though the operator graph, the initializer names and the weights
# are all identical (verified directly -- see REPRODUCE.md, "Reproducing the
# model file byte for byte").
#
#   torch 2.2.2 is the version that reproduces the shipped
#   models/DeepSetsValueNetwork.onnx byte hash
#   86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915.
#
# That hash is what the benchmark harnesses pin against, so an environment on
# any other torch will fail their integrity check with a perfectly good model.
# On a different version the model still reproduces exactly; only the file hash
# does not, and the fix is to add the new hash to the config's
# allowed_onnx_sha256 rather than to chase the old one.
#
# CPU ONLY, deliberately: the cluster partition these experiments run on has no
# GPUs, and the CUDA wheels are several GB of download that would never be used.
# The CPU index URL below is what keeps pip from pulling them in.
#
# Idempotent: safe to re-run. It will not overwrite an existing venv unless
# --force is given.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Pinned versions -------------------------------------------------------
# torch: see the header. Do not bump without re-reading REPRODUCE.md.
TORCH_VERSION="2.2.2"
# torch_geometric 2.5.x is the line built against torch 2.2. Only the pure
# Python package is needed (Data containers, DataLoader, global_mean_pool) --
# NOT the compiled torch-scatter/torch-sparse extensions, which torch_geometric
# has not required for these operations since 2.3.
PYG_VERSION="2.5.3"
# numpy < 2 is a hard requirement of torch 2.2.x, not a preference: torch 2.2
# was built against the NumPy 1.x ABI and fails at import against 2.x.
NUMPY_VERSION="1.26.4"
ONNXRUNTIME_VERSION="1.17.3"
SKLEARN_VERSION="1.4.2"
# Only tools/diagnose_value_net.py needs this, and only to write two PNGs.
MATPLOTLIB_VERSION="3.8.4"

# onnx (the format library) is SEPARATE from onnxruntime (the inference engine)
# and is not pulled in by it. torch.onnx.export imports onnx at call time, so
# without it training/export_to_onnx.py dies with ModuleNotFoundError partway
# through -- after training has finished, which on the cluster means a seed's
# checkpoint is on disk but its .onnx is not. That is exactly what happened
# during the per-seed training run and had to be patched by hand.
#
# Version-gated by Python, because onnx >= 1.20 requires Python >= 3.10 and the
# supported range here is 3.9-3.11 (see the PY_VER check below), so BOTH
# branches are reachable: the cluster's Python is 3.9, a local macOS install is
# typically 3.10/3.11.
ONNX_VERSION_PY39="1.19.1"
ONNX_VERSION_PY310_PLUS="1.21.0"

TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
# ---------------------------------------------------------------------------

VENV_DIR="${SOT_VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE=0
SKIP_MATPLOTLIB=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --venv-dir <path>   Where to create the venv (default: $REPO_ROOT/.venv,
                       or \$SOT_VENV_DIR). On a cluster, put this somewhere with
                       a real quota -- \$HPCWORK, not \$HOME.
  --python <path>     Interpreter to build the venv from (default: python3).
                       Must be 3.9, 3.10 or 3.11: torch $TORCH_VERSION publishes no
                       wheels for 3.12+.
  --skip-matplotlib   Skip matplotlib (only tools/diagnose_value_net.py uses it)
  --force             Recreate the venv even if it already exists
  -h, --help          Show this help

After it finishes:
  source $VENV_DIR/bin/activate
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --venv-dir) VENV_DIR="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --skip-matplotlib) SKIP_MATPLOTLIB=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: interpreter not found: $PYTHON_BIN" >&2
  echo "       On a cluster you probably need to load a module first, e.g." >&2
  echo "         module load Python/3.11.3" >&2
  exit 1
}

PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VER" in
  3.9)
    ONNX_VERSION="$ONNX_VERSION_PY39"
    ;;
  3.10|3.11)
    ONNX_VERSION="$ONNX_VERSION_PY310_PLUS"
    ;;
  *)
    echo "ERROR: Python $PY_VER is not supported here." >&2
    echo "       torch $TORCH_VERSION publishes wheels for 3.8-3.11 only, and $TORCH_VERSION is pinned" >&2
    echo "       because it is what reproduces the shipped ONNX byte hash (see this" >&2
    echo "       script's header). Pass --python /path/to/python3.11." >&2
    exit 1
    ;;
esac
echo "Python $PY_VER -> onnx $ONNX_VERSION"

if [ -d "$VENV_DIR" ] && [ "$FORCE" = "0" ]; then
  echo "venv already exists at $VENV_DIR -- nothing to do (--force to recreate)."
  echo
  echo "Activate it with:"
  echo "  source $VENV_DIR/bin/activate"
  exit 0
fi

if [ -d "$VENV_DIR" ]; then
  echo "=== --force: removing the existing venv at $VENV_DIR ==="
  rm -rf "$VENV_DIR"
fi

echo "=== Creating venv at $VENV_DIR (Python $PY_VER) ==="
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "=== Upgrading pip ==="
python -m pip install --upgrade pip

echo
echo "=== Installing CPU-only torch $TORCH_VERSION ==="
# --index-url, not --extra-index-url: with --extra, pip is free to resolve torch
# from PyPI instead, which on Linux is the CUDA build. The whole point of this
# line is that it cannot.
python -m pip install --index-url "$TORCH_CPU_INDEX" "torch==$TORCH_VERSION"

echo
echo "=== Installing the rest (PyPI) ==="
PACKAGES=(
  "numpy==$NUMPY_VERSION"
  "torch_geometric==$PYG_VERSION"
  "onnx==$ONNX_VERSION"
  "onnxruntime==$ONNXRUNTIME_VERSION"
  "scikit-learn==$SKLEARN_VERSION"
)
# onnx 1.19.1 pulls ml_dtypes as a transitive dependency (0.5.4 at time of
# writing). It is deliberately NOT pinned: every pin in this script is a direct
# dependency of our own code, and the transitive closure -- scipy and joblib
# under scikit-learn, filelock and sympy under torch -- is left to pip
# throughout. Pinning one transitive package and not the others would imply a
# guarantee this script does not make.
if [ "$SKIP_MATPLOTLIB" = "0" ]; then
  PACKAGES+=("matplotlib==$MATPLOTLIB_VERSION")
fi
python -m pip install "${PACKAGES[@]}"

echo
echo "=== Verifying ==="
EXPECTED_ONNX="$ONNX_VERSION" python - <<'PYEOF'
import os, sys
import numpy, torch, torch_geometric, onnx, onnxruntime, sklearn

print(f"  python           {sys.version.split()[0]}")
print(f"  numpy            {numpy.__version__}")
print(f"  torch            {torch.__version__}")
print(f"  torch_geometric  {torch_geometric.__version__}")
print(f"  onnx             {onnx.__version__}")
print(f"  onnxruntime      {onnxruntime.__version__}")
print(f"  scikit-learn     {sklearn.__version__}")

# The gate is by Python minor version, so a mismatch here means the resolution
# above disagrees with what pip actually installed -- worth failing on, since
# the symptom otherwise appears much later, in export_to_onnx.py.
expected_onnx = os.environ["EXPECTED_ONNX"]
if onnx.__version__ != expected_onnx:
    sys.exit(f"  ERROR: expected onnx {expected_onnx} for Python "
             f"{sys.version_info[0]}.{sys.version_info[1]}, got {onnx.__version__}")
try:
    import matplotlib
    print(f"  matplotlib       {matplotlib.__version__}")
except ImportError:
    print("  matplotlib       (not installed)")

if torch.version.cuda is not None:
    print()
    print(f"  WARNING: this torch build carries CUDA {torch.version.cuda}. The CPU-only index was")
    print( "           meant to prevent that -- it will still work, but it is a much larger")
    print( "           install than intended and not what was pinned.")

# The imports the pipeline actually performs, not just the top-level packages:
# a torch/torch_geometric version mismatch usually surfaces here, not above.
from torch_geometric.data import Data           # noqa: F401  training/StateParser.py
from torch_geometric.loader import DataLoader   # noqa: F401  training/train_local.py
from torch_geometric.nn import global_mean_pool # noqa: F401  training/ValueNetwork.py

# Exercise the export path end to end on a toy model. torch.onnx.export imports
# onnx lazily, at call time, so a missing or incompatible onnx does not show up
# on `import torch` -- it shows up in training/export_to_onnx.py, after training
# has already finished. Catching it here costs a fraction of a second.
import io
_buf = io.BytesIO()
torch.onnx.export(torch.nn.Linear(4, 1), (torch.zeros(1, 4),), _buf, opset_version=14)
onnx.load_from_string(_buf.getvalue())
print()
print("  torch.onnx.export + onnx.load round-trip OK (the path export_to_onnx.py uses).")
print("  torch_geometric Data / DataLoader / global_mean_pool all import cleanly.")
PYEOF

EXPECTED_ONNX_SHA="86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915"
INSTALLED_TORCH="$(python -c 'import torch; print(torch.__version__)')"
echo
if [ "$INSTALLED_TORCH" = "$TORCH_VERSION" ]; then
  echo "  torch $TORCH_VERSION: re-exporting models/deepsets_value_network.pth should reproduce"
  echo "  the shipped ONNX byte hash $EXPECTED_ONNX_SHA."
else
  echo "  NOTE: torch resolved to $INSTALLED_TORCH, not the pinned $TORCH_VERSION. Exports from this"
  echo "  environment will have a DIFFERENT ONNX sha256 than $EXPECTED_ONNX_SHA,"
  echo "  even though the model itself is identical. That is expected and is not an error --"
  echo "  add the new hash to the experiment config's allowed_onnx_sha256."
fi

cat <<EOF

Done.

Activate with:
  source $VENV_DIR/bin/activate

On the cluster, put that line in the env script the SLURM templates source
(scripts/slurm_train.sh expects the venv at \$SOT_VENV_DIR or the path it sets).
EOF
