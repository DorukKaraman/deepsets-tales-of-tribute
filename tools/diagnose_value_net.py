"""
Read-only diagnostic for the Sakkirina value network.

Streams real validation samples, runs them through both the exported ONNX
model and the PyTorch checkpoint it's supposed to match, and reports
agreement, calibration, and accuracy stats (overall and bucketed by the
prestige clock). Saves two histogram PNGs. Does not modify any existing
file and does not train anything.

Requirements (beyond what training/ already needs -- torch, torch_geometric):
    pip install onnxruntime scikit-learn matplotlib
"""
import argparse
import gzip
import json
import os
import random
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
OUT_DIR = os.path.join(SCRIPT_DIR, "out")

DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "GameRunner", "Val_Sakkirina.jsonl.gz")
PTH_PATH = os.path.join(REPO_ROOT, "models", "deepsets_value_network.pth")
ONNX_PATH = os.path.join(REPO_ROOT, "models", "DeepSetsValueNetwork.onnx")

sys.path.insert(0, TRAINING_DIR)

import torch  # noqa: E402
from StateParser import json_to_pyg_graph, NODE_DIM, GLOBAL_DIM  # noqa: E402
from ValueNetwork import TributeValueNetwork  # noqa: E402
import onnxruntime as ort  # noqa: E402
from sklearn.metrics import roc_auc_score, brier_score_loss  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Order matches StateParser.extract_global_context exactly (indices 0-18).
# TREASURY excluded from the patron favour block -- it has no favour
# mechanic (confirmed empirically: NO_PLAYER_SELECTED in 1753/1753 sampled
# records), unlike the node vector's Deck one-hot, which still includes it.
GLOBAL_FEATURE_NAMES = [
    "CurrentPlayer.Coins",
    "CurrentPlayer.Power",
    "CurrentPlayer.Prestige",
    "CurrentPlayer.PatronCalls",
    "EnemyPlayer.Coins",
    "EnemyPlayer.Power",
    "EnemyPlayer.Prestige",
    "Patron.ANSEI",
    "Patron.DUKE_OF_CROWS",
    "Patron.RAJHIN",
    "Patron.ORGNUM",
    "Patron.PELIN",
    "Patron.SAINT_ALESSIA",
    "PrestigeClock",
    "PrestigeDifferential",
    "MyDeckSize",
    "EnemyKnownDeckSize",
    "MyAgentCount",
    "EnemyAgentCount",
]

# Global feature index 13 is the prestige clock (see
# StateParser.extract_global_context / FeatureExtractor.EncodeGlobalContext).
# A named constant, not a bare literal, specifically because this index
# already drifted once: it moved 16 -> 13 when TREASURY was dropped from the
# patron favour block, and this file's own bucketing logic kept reading
# column 16 (silently the wrong column, not a crash) until a full pipeline
# dry run caught the mismatched bucket counts against train_local.py's own
# (correctly-indexed) bucketing on the same data.
PRESTIGE_CLOCK_GLOBAL_INDEX = 13
PRESTIGE_BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, float("inf"))]
AGREEMENT_THRESHOLD = 1e-4
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)

# (lo, hi, label) -- half-open [lo, hi), used to localize the ONNX export bug
# by node count (the dummy trace shape in export_to_onnx.py is 15 nodes).
NUM_NODE_BUCKETS = [
    (0, 10, "<10"),
    (10, 15, "10-14"),
    (15, 16, "15"),
    (16, 21, "16-20"),
    (21, 31, "21-30"),
    (31, float("inf"), ">30"),
]


def require_file(path, description):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {description} not found at: {path}")


def bucket_label(lo, hi):
    return f"[{lo:.2f}, inf)" if hi == float("inf") else f"[{lo:.2f}, {hi:.2f})"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_pytorch_model():
    model = TributeValueNetwork(node_in_dim=NODE_DIM, global_in_dim=GLOBAL_DIM)
    model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
    model.eval()
    return model


def run_onnx(session, x_np, u_np):
    out = session.run(None, {
        "node_features": x_np.astype(np.float32),
        "global_features": u_np.astype(np.float32),
    })[0]
    logit = float(np.asarray(out).reshape(-1)[0])
    return sigmoid(logit)


def iter_samples(data_path, limit):
    opener = gzip.open if data_path.endswith(".gz") else open
    with opener(data_path, "rt") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                yield int(row["outcome"]), row["data"]["state"]
            except Exception as e:
                print(f"  [WARN] skipping malformed line {i}: {e}")


def compute_stats(labels, probs):
    """All step-3 style metrics for one slice of samples. labels/probs are 1D np arrays."""
    stats = {
        "n": len(labels),
        "base_rate": float(np.mean(labels)),
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "frac_gt_95": float(np.mean(probs > 0.95)),
        "frac_lt_05": float(np.mean(probs < 0.05)),
        "acc_50": float(np.mean((probs >= 0.5).astype(np.float64) == labels)),
        "brier": float(brier_score_loss(labels, probs)),
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(probs, p))
    stats["auc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    return stats


def print_stats_block(stats, baseline_acc=None):
    print(f"  label base rate (P[label=1])   : {stats['base_rate']:.4f}")
    print(f"  mean predicted prob            : {stats['mean']:.4f}")
    print(f"  std  predicted prob            : {stats['std']:.4f}")
    print(f"  min  predicted prob            : {stats['min']:.4f}")
    print(f"  max  predicted prob            : {stats['max']:.4f}")
    pct_str = ", ".join(f"p{p}={stats[f'p{p}']:.4f}" for p in PERCENTILES)
    print(f"  percentiles                    : {pct_str}")
    print(f"  fraction > 0.95                : {stats['frac_gt_95']:.4f}")
    print(f"  fraction < 0.05                : {stats['frac_lt_05']:.4f}")
    print(f"  accuracy @ 0.5 (network)       : {stats['acc_50']:.4f}")
    if baseline_acc is not None:
        print(f"  accuracy @ 0.5 (baseline)      : {baseline_acc:.4f}")
    auc_str = f"{stats['auc']:.4f}" if not np.isnan(stats["auc"]) else "n/a (single class)"
    print(f"  ROC AUC                        : {auc_str}")
    print(f"  Brier score                    : {stats['brier']:.4f}")


def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only ONNX-vs-PyTorch diagnostic for the Sakkirina value network.")
    parser.add_argument("--limit", type=int, default=10000,
                         help="Number of validation samples to read (default: 10000)")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                         help=f"Path to validation .jsonl.gz (default: {DEFAULT_DATA_PATH})")
    args = parser.parse_args()

    require_file(PTH_PATH, "PyTorch checkpoint")
    require_file(ONNX_PATH, "ONNX model")
    require_file(args.data_path, "Validation data")

    print(f"PyTorch checkpoint : {PTH_PATH}")
    print(f"ONNX model         : {ONNX_PATH}")
    print(f"Validation data    : {args.data_path}")
    print(f"Sample limit       : {args.limit}")

    model = load_pytorch_model()
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    labels, pt_probs, onnx_probs = [], [], []
    prestige_clocks, cur_prestiges, enemy_prestiges = [], [], []
    u_vectors = []
    num_nodes_list = []

    n_read = 0
    with torch.no_grad():
        for label, state in iter_samples(args.data_path, args.limit):
            graph = json_to_pyg_graph(state)
            pt_prob = sigmoid(model(graph).item())

            x_np = graph.x.numpy()
            u_np = graph.u.numpy()
            onnx_prob = run_onnx(session, x_np, u_np)

            current = state.get("CurrentPlayer", {})
            enemy = state.get("EnemyPlayer", {})

            labels.append(label)
            pt_probs.append(pt_prob)
            onnx_probs.append(onnx_prob)
            prestige_clocks.append(float(u_np[0, PRESTIGE_CLOCK_GLOBAL_INDEX]))
            cur_prestiges.append(float(current.get("Prestige", 0)))
            enemy_prestiges.append(float(enemy.get("Prestige", 0)))
            u_vectors.append(u_np[0].tolist())
            num_nodes_list.append(int(graph.x.shape[0]))

            n_read += 1
            if n_read % 2000 == 0:
                print(f"  ... {n_read} samples processed")

    if n_read == 0:
        sys.exit("No samples were read -- aborting.")

    labels = np.array(labels, dtype=np.float64)
    pt_probs = np.array(pt_probs, dtype=np.float64)
    onnx_probs = np.array(onnx_probs, dtype=np.float64)
    prestige_clocks = np.array(prestige_clocks, dtype=np.float64)
    cur_prestiges = np.array(cur_prestiges, dtype=np.float64)
    enemy_prestiges = np.array(enemy_prestiges, dtype=np.float64)
    num_nodes_arr = np.array(num_nodes_list, dtype=np.int64)

    print_header(f"Loaded {n_read} samples from {args.data_path}")

    # ---- 2. ONNX vs PyTorch agreement ----
    abs_diff = np.abs(pt_probs - onnx_probs)
    max_diff = float(np.max(abs_diff))
    verdict = "PASS" if max_diff <= AGREEMENT_THRESHOLD else "FAIL"

    print_header("ONNX vs PyTorch agreement")
    print(f"  max abs diff (prob)  : {max_diff:.6e}")
    print(f"  mean abs diff (prob) : {float(np.mean(abs_diff)):.6e}")
    print(f"  threshold            : {AGREEMENT_THRESHOLD:.1e}")
    print(f"  VERDICT              : {verdict}")

    print_header("ONNX vs PyTorch agreement bucketed by num_nodes")
    node_header = f"{'num_nodes':<12}{'n':>7}{'mean|diff|':>13}{'max|diff|':>12}"
    print(node_header)
    print("-" * len(node_header))
    for lo, hi, label_txt in NUM_NODE_BUCKETS:
        mask = (num_nodes_arr >= lo) & (num_nodes_arr < hi)
        n = int(np.sum(mask))
        if n == 0:
            print(f"{label_txt:<12}{n:>7}{'--':>13}{'--':>12}")
            continue
        bucket_diff = abs_diff[mask]
        print(f"{label_txt:<12}{n:>7}{float(np.mean(bucket_diff)):>13.6f}{float(np.max(bucket_diff)):>12.6f}")

    # ---- 3. Overall PyTorch prediction stats ----
    print_header("PyTorch prediction statistics (overall)")
    overall_stats = compute_stats(labels, pt_probs)
    print_stats_block(overall_stats)

    # ---- 4. Bucketed by prestige clock ----
    print_header(f"Bucketed by prestige clock (global feature index {PRESTIGE_CLOCK_GLOBAL_INDEX})")
    header = f"{'bucket':<14}{'n':>7}{'base_rate':>11}{'acc@0.5':>9}{'baseline':>10}{'auc':>8}{'brier':>8}"
    print(header)
    print("-" * len(header))

    bucket_results = []
    for lo, hi in PRESTIGE_BUCKETS:
        mask = (prestige_clocks >= lo) & (prestige_clocks < hi)
        n = int(np.sum(mask))
        label_txt = bucket_label(lo, hi)
        if n == 0:
            print(f"{label_txt:<14}{n:>7}{'--':>11}{'--':>9}{'--':>10}{'--':>8}{'--':>8}")
            bucket_results.append((label_txt, n, None, None))
            continue

        b_labels = labels[mask]
        b_probs = pt_probs[mask]
        b_cur = cur_prestiges[mask]
        b_enemy = enemy_prestiges[mask]

        b_stats = compute_stats(b_labels, b_probs)
        baseline_pred = (b_cur > b_enemy).astype(np.float64)
        baseline_acc = float(np.mean(baseline_pred == b_labels))
        auc_txt = f"{b_stats['auc']:.4f}" if not np.isnan(b_stats["auc"]) else "n/a"

        print(f"{label_txt:<14}{n:>7}{b_stats['base_rate']:>11.4f}{b_stats['acc_50']:>9.4f}"
              f"{baseline_acc:>10.4f}{auc_txt:>8}{b_stats['brier']:>8.4f}")
        bucket_results.append((label_txt, n, b_stats, baseline_acc))

    for label_txt, n, b_stats, baseline_acc in bucket_results:
        print_header(f"Bucket {label_txt}  (n={n})")
        if b_stats is None:
            print("  no samples in this bucket")
            continue
        print_stats_block(b_stats, baseline_acc=baseline_acc)

    # ---- 5. Histograms ----
    os.makedirs(OUT_DIR, exist_ok=True)

    hist_path = os.path.join(OUT_DIR, "val_pred_hist.png")
    plt.figure(figsize=(7, 5))
    plt.hist(pt_probs, bins=50, range=(0, 1), color="#4C72B0", edgecolor="black", linewidth=0.3)
    plt.xlabel("Predicted P(win)")
    plt.ylabel("Count")
    plt.title(f"Predicted probability distribution (n={n_read})")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    facet_path = os.path.join(OUT_DIR, "val_pred_hist_by_prestige.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (lo, hi) in zip(axes.flat, PRESTIGE_BUCKETS):
        mask = (prestige_clocks >= lo) & (prestige_clocks < hi)
        vals = pt_probs[mask]
        ax.hist(vals, bins=30, range=(0, 1), color="#55A868", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{bucket_label(lo, hi)}  (n={int(np.sum(mask))})")
        ax.set_xlabel("Predicted P(win)")
        ax.set_ylabel("Count")
    fig.suptitle("Predicted probability distribution by prestige-clock bucket")
    fig.tight_layout()
    fig.savefig(facet_path, dpi=150)
    plt.close(fig)

    print_header("Saved plots")
    print(f"  {hist_path}")
    print(f"  {facet_path}")

    # ---- 6. Random sample u-vector eyeball check ----
    print_header("Random sample global feature vectors (eyeball check)")
    for idx in random.sample(range(n_read), k=min(3, n_read)):
        print(f"\n  Sample #{idx}  (label={int(labels[idx])}, pt_prob={pt_probs[idx]:.4f}):")
        for name, val in zip(GLOBAL_FEATURE_NAMES, u_vectors[idx]):
            print(f"    {name:<24}: {val:.4f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
