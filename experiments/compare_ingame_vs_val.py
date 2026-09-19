"""
Read-only comparison of in-game NeuralEvaluate feature dumps (produced by
experiments/bots/FeatureDumper.cs via SOT_DUMP_DIR) against the validation feature
distribution. Does not modify any file and does not train anything.

--dump-path accepts either a single .jsonl file (old behavior) or a directory,
in which case both evals_*.jsonl (column-mean rows) and full_*.jsonl
(full-matrix rows, mean-reduced here to column means) are pooled together
across every game_id found.

Requirements: same as tools/diagnose_value_net.py (torch, torch_geometric,
onnxruntime, scikit-learn, matplotlib) -- this script imports
GLOBAL_FEATURE_NAMES/load_pytorch_model/sigmoid from that module rather than
duplicating them, which pulls in its full dependency set even though this
script itself only uses torch/matplotlib directly.
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
OUT_DIR = os.path.join(SCRIPT_DIR, "out")
DEFAULT_VAL_PATH = os.path.join(REPO_ROOT, "GameRunner", "Val_Sakkirina.jsonl.gz")

TOOLS_DIR = os.path.join(REPO_ROOT, "tools")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, TOOLS_DIR)  # diagnose_value_net lives in tools/

from diagnose_value_net import GLOBAL_FEATURE_NAMES, load_pytorch_model, sigmoid  # noqa: E402
from StateParser import json_to_pyg_graph, NODE_DIM  # noqa: E402
import torch  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (lo, hi, block name) -- half-open [lo, hi) over the NODE_DIM node-feature
# columns, matching StateParser.encode_card's layout exactly.
NODE_FEATURE_BLOCKS = [
    (0, 7, "deck"),
    (7, 8, "cost"),
    (8, 12, "type"),
    (12, 13, "hp"),
    (13, 14, "taunt"),
    (14, 90, "effect"),
    (90, 99, "location"),
]


def node_feature_label(i):
    for lo, hi, name in NODE_FEATURE_BLOCKS:
        if lo <= i < hi:
            return f"{name}[{i}]"
    return f"unknown[{i}]"


NODE_FEATURE_NAMES = [node_feature_label(i) for i in range(NODE_DIM)]

PROB_PERCENTILES = (1, 25, 50, 75, 99)


def require_exists(path, description):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {description} not found at: {path}")


def print_header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def resolve_dump_paths(dump_path):
    """--dump-path may be a single file (old behavior) or a directory, in which
    case pool every evals_*.jsonl and full_*.jsonl under it."""
    if os.path.isdir(dump_path):
        paths = (sorted(glob.glob(os.path.join(dump_path, "evals_*.jsonl"))) +
                  sorted(glob.glob(os.path.join(dump_path, "full_*.jsonl"))))
        if not paths:
            sys.exit(f"ERROR: no evals_*.jsonl or full_*.jsonl found under {dump_path}")
        return paths
    return [dump_path]


def load_dump(paths):
    game_id, turn, is_terminal, num_nodes, csharp_prob = [], [], [], [], []
    global_rows, node_mean_rows = [], []

    for path in paths:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "node_means" in row:
                        node_means = row["node_means"]
                    elif "nodes" in row:
                        node_means = np.array(row["nodes"], dtype=np.float64).mean(axis=0).tolist()
                    else:
                        raise KeyError("row has neither 'node_means' nor 'nodes'")

                    game_id.append(row.get("game_id", -1))
                    turn.append(row["turn"])
                    is_terminal.append(bool(row["is_terminal"]))
                    num_nodes.append(row["num_nodes"])
                    csharp_prob.append(row["csharp_prob"])
                    global_rows.append(row["global"])
                    node_mean_rows.append(node_means)
                except Exception as e:
                    print(f"  [WARN] skipping malformed line {i} in {path}: {e}")

    return {
        "game_id": np.array(game_id, dtype=np.int64),
        "turn": np.array(turn, dtype=np.int64),
        "is_terminal": np.array(is_terminal, dtype=bool),
        "num_nodes": np.array(num_nodes, dtype=np.int64),
        "csharp_prob": np.array(csharp_prob, dtype=np.float64),
        "global": np.array(global_rows, dtype=np.float64),
        "node_means": np.array(node_mean_rows, dtype=np.float64),
    }


def load_validation(val_path, limit, model):
    global_rows, node_mean_rows, probs = [], [], []
    opener = gzip.open if val_path.endswith(".gz") else open
    with opener(val_path, "rt") as f:
        with torch.no_grad():
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    state = row["data"]["state"]
                except Exception as e:
                    print(f"  [WARN] skipping malformed validation line {i}: {e}")
                    continue

                graph = json_to_pyg_graph(state)
                pt_prob = sigmoid(model(graph).item())

                u = graph.u.numpy()[0]
                node_mean = graph.x.numpy().mean(axis=0)

                global_rows.append(u.tolist())
                node_mean_rows.append(node_mean.tolist())
                probs.append(pt_prob)

    return {
        "global": np.array(global_rows, dtype=np.float64),
        "node_means": np.array(node_mean_rows, dtype=np.float64),
        "prob": np.array(probs, dtype=np.float64),
    }


# ----------------------------------------------------------------------------
# Step 3: csharp_prob breakdown tables
# ----------------------------------------------------------------------------

def prob_stats(probs):
    if len(probs) == 0:
        return None
    stats = {"n": len(probs), "mean": float(np.mean(probs)), "std": float(np.std(probs))}
    for p in PROB_PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(probs, p))
    return stats


def print_prob_table(rows):
    header = (f"{'label':<20}{'n':>7}{'mean':>9}{'std':>9}" +
              "".join(f"{'p' + str(p):>8}" for p in PROB_PERCENTILES))
    print(header)
    print("-" * len(header))
    for label, s in rows:
        if s is None:
            print(f"{label:<20}{0:>7}" + f"{'--':>9}" * (2 + len(PROB_PERCENTILES)))
            continue
        line = f"{label:<20}{s['n']:>7}{s['mean']:>9.4f}{s['std']:>9.4f}"
        line += "".join(f"{s[f'p{p}']:>8.4f}" for p in PROB_PERCENTILES)
        print(line)


# ----------------------------------------------------------------------------
# Step 4: effect-size table (pooled std + absolute-difference floor)
# ----------------------------------------------------------------------------

ABS_DIFF_FLOOR = 0.01


def print_effect_size_table(labels, a, b, top_n=None):
    mean_a, std_a = a.mean(axis=0), a.std(axis=0)
    mean_b, std_b = b.mean(axis=0), b.std(axis=0)
    raw_diff = np.abs(mean_a - mean_b)
    pooled = np.sqrt(0.5 * (std_a ** 2 + std_b ** 2)) + 1e-3
    d = raw_diff / pooled

    order = [idx for idx in np.argsort(-d) if raw_diff[idx] > ABS_DIFF_FLOOR]
    if top_n is not None:
        order = order[:top_n]

    header = f"{'column':<28}{'ingame_mean':>13}{'ingame_std':>12}{'val_mean':>11}{'val_std':>10}{'raw_diff':>10}{'d':>8}"
    print(header)
    print("-" * len(header))
    if not order:
        print(f"  (no columns exceed the {ABS_DIFF_FLOOR} absolute-difference floor)")
        return
    for idx in order:
        print(f"{labels[idx]:<28}{mean_a[idx]:>13.4f}{std_a[idx]:>12.4f}"
              f"{mean_b[idx]:>11.4f}{std_b[idx]:>10.4f}{raw_diff[idx]:>10.4f}{d[idx]:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-path", type=str, required=True,
                         help="A single dump .jsonl file, OR a directory containing evals_*.jsonl / full_*.jsonl")
    parser.add_argument("--val-path", type=str, default=DEFAULT_VAL_PATH,
                         help=f"Validation .jsonl.gz path (default: {DEFAULT_VAL_PATH})")
    parser.add_argument("--val-limit", type=int, default=5000,
                         help="Number of validation samples to parse for comparison (default: 5000)")
    args = parser.parse_args()

    require_exists(args.dump_path, "Feature dump path")
    require_exists(args.val_path, "Validation data")

    dump_paths = resolve_dump_paths(args.dump_path)
    print(f"In-game dump  : {len(dump_paths)} file(s)")
    for p in dump_paths:
        print(f"  {p}")
    print(f"Validation    : {args.val_path} (limit={args.val_limit})")

    dump = load_dump(dump_paths)
    n_dump = len(dump["turn"])
    if n_dump == 0:
        sys.exit("No rows read from the dump -- aborting.")
    n_terminal = int(dump["is_terminal"].sum())
    n_games = len(np.unique(dump["game_id"]))
    print(f"Loaded {n_dump} dumped eval rows from {n_games} distinct game_id(s) "
          f"({n_terminal} terminal, {n_dump - n_terminal} non-terminal)")

    model = load_pytorch_model()
    val = load_validation(args.val_path, args.val_limit, model)
    print(f"Loaded {len(val['global'])} validation samples")

    # ---- Step 3: csharp_prob breakdown ----
    print_header("(i) csharp_prob: overall")
    print_prob_table([("overall", prob_stats(dump["csharp_prob"]))])

    print_header("(ii) csharp_prob: split by is_terminal")
    rows = [(f"is_terminal={flag}", prob_stats(dump["csharp_prob"][dump["is_terminal"] == flag]))
            for flag in (False, True)]
    print_prob_table(rows)

    print_header("(iii) csharp_prob: split by turn")
    turns_sorted = sorted(set(dump["turn"].tolist()))
    rows = [(f"turn={t}", prob_stats(dump["csharp_prob"][dump["turn"] == t])) for t in turns_sorted]
    print_prob_table(rows)

    # ---- Step 4 applied to (a)/(b)/(c) ----
    print_header("(a) Global feature columns: in-game (all) vs validation, sorted by effect size d")
    print_effect_size_table(GLOBAL_FEATURE_NAMES, dump["global"], val["global"])

    print_header("(b) Node-mean columns: in-game (all) vs validation, top 15 by effect size d")
    print_effect_size_table(NODE_FEATURE_NAMES, dump["node_means"], val["node_means"], top_n=15)

    for flag, label in [(True, "is_terminal=true"), (False, "is_terminal=false")]:
        mask = dump["is_terminal"] == flag
        n = int(mask.sum())
        print_header(f"(c) Global feature columns: in-game ({label}, n={n}) vs validation")
        if n == 0:
            print("  no dumped rows with this flag -- skipping")
            continue
        print_effect_size_table(GLOBAL_FEATURE_NAMES, dump["global"][mask], val["global"])

    # ---- Step 5: re-plot as 1x2, log-y, split by is_terminal ----
    os.makedirs(OUT_DIR, exist_ok=True)
    hist_path = os.path.join(OUT_DIR, "ingame_vs_val_probs.png")
    bins = np.linspace(0, 1, 51)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [(axes[0], False, "is_terminal=false"), (axes[1], True, "is_terminal=true")]
    for ax, flag, title in panels:
        mask = dump["is_terminal"] == flag
        ax.hist(val["prob"], bins=bins, density=True, alpha=0.5,
                label=f"validation (n={len(val['prob'])})", color="#4C72B0")
        ax.hist(dump["csharp_prob"][mask], bins=bins, density=True, alpha=0.5,
                label=f"in-game (n={int(mask.sum())})", color="#DD8452")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted P(win)")
        ax.set_title(title)
    axes[0].set_ylabel("Density (log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    print_header("Saved plot")
    print(f"  {hist_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
