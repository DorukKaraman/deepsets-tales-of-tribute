"""
Read-only verification that the C# inference path (FeatureExtractor +
ValueNetworkEvaluator) matches the PyTorch reference on REAL in-game inputs,
not synthetic/random ones.

Loads full_*.jsonl dumps (experiments/bots/FeatureDumper.cs, SOT_DUMP_DIR) -- each row is
the exact node-feature matrix and global vector NeuralEvaluate fed the ONNX
model for one real evaluation -- replays them through the PyTorch checkpoint,
applies sigmoid, and compares against the stored csharp_prob.

This is the decisive check for the avgNeural-vs-histogram contradiction: if it
FAILS, there is still a C#-side feature-extraction or inference bug and that
outranks everything else. If it PASSES, C# inference is verified end-to-end and
every remaining discrepancy is a data-distribution problem, not a code bug.

Requirements: same as tools/diagnose_value_net.py (torch, torch_geometric,
onnxruntime, scikit-learn) -- reuses load_pytorch_model/sigmoid from there.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
sys.path.insert(0, SCRIPT_DIR)
# diagnose_value_net lives in tools/; it puts training/ on the path itself.
sys.path.insert(0, TOOLS_DIR)

from diagnose_value_net import load_pytorch_model, sigmoid  # noqa: E402
import torch  # noqa: E402

AGREEMENT_THRESHOLD = 1e-4


class Shim:
    """Minimal stand-in for a PyG Data/Batch object -- TributeValueNetwork.forward
    only reads .x, .u, and (optionally) .batch."""
    def __init__(self, x, u):
        self.x = x
        self.u = u
        self.batch = torch.zeros(x.size(0), dtype=torch.long)


def load_full_dumps(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "full_*.jsonl")))
    rows = []
    for path in paths:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    print(f"  [WARN] skipping malformed line {i} in {path}: {e}")
    return paths, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=str, required=True,
                         help="Directory containing full_*.jsonl dumps (i.e. SOT_DUMP_DIR)")
    args = parser.parse_args()

    if not os.path.isdir(args.dump_dir):
        sys.exit(f"ERROR: dump dir not found: {args.dump_dir}")

    paths, rows = load_full_dumps(args.dump_dir)
    if not rows:
        sys.exit(f"No rows found in full_*.jsonl under {args.dump_dir} -- aborting.")

    print(f"Loaded {len(rows)} full-matrix rows from {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    model = load_pytorch_model()

    results = []
    with torch.no_grad():
        for row in rows:
            x = torch.tensor(row["nodes"], dtype=torch.float32)
            u = torch.tensor([row["global"]], dtype=torch.float32)
            shim = Shim(x, u)
            logit = model(shim).item()
            pytorch_prob = sigmoid(logit)
            diff = abs(pytorch_prob - row["csharp_prob"])
            results.append({
                "game_id": row.get("game_id"),
                "turn": row.get("turn"),
                "is_terminal": row.get("is_terminal"),
                "num_nodes": row.get("num_nodes"),
                "csharp_prob": row["csharp_prob"],
                "pytorch_prob": pytorch_prob,
                "diff": diff,
            })

    diffs = np.array([r["diff"] for r in results], dtype=np.float64)
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())
    verdict = "PASS" if max_diff <= AGREEMENT_THRESHOLD else "FAIL"

    print()
    print("=" * 78)
    print("C# vs PyTorch inference agreement on REAL in-game inputs")
    print("=" * 78)
    print(f"  n rows        : {len(results)}")
    print(f"  max abs diff  : {max_diff:.6e}")
    print(f"  mean abs diff : {mean_diff:.6e}")
    print(f"  threshold     : {AGREEMENT_THRESHOLD:.1e}")
    print(f"  VERDICT       : {verdict}")

    if verdict == "FAIL":
        worst = sorted(results, key=lambda r: -r["diff"])[:5]
        print()
        print("Worst 5 rows:")
        header = f"{'game_id':>8}{'turn':>6}{'is_terminal':>13}{'num_nodes':>11}{'csharp_prob':>13}{'pytorch_prob':>14}{'diff':>12}"
        print(header)
        print("-" * len(header))
        for r in worst:
            print(f"{r['game_id']:>8}{r['turn']:>6}{str(r['is_terminal']):>13}{r['num_nodes']:>11}"
                  f"{r['csharp_prob']:>13.6f}{r['pytorch_prob']:>14.6f}{r['diff']:>12.3e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
