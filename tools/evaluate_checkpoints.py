"""
Score .pth checkpoints on one common dataset directory.

Reports BCE loss, accuracy, ROC AUC and Brier score per checkpoint, overall and
by prestige-clock bucket, each beside that slice's majority-class baseline,
then a side-by-side table. The dataset is streamed once and every checkpoint
scores each batch, so all of them see the same samples in the same order.

DeepSets and flat-MLP checkpoints can be mixed. DeepSets checkpoints are
recognised by their keys. A flat checkpoint's arch cannot be read off its
weights (matched and matched_sorted have identical shapes), so it comes from
the run_config.json beside the checkpoint's real path, or from --arch.

Checkpoints are scored through the exported forward pass, which takes a plain
per-graph mean over nodes rather than global_mean_pool (see export_to_onnx.py).

    python tools/evaluate_checkpoints.py A.pth B.pth --data-dir <dir> \\
        [--per-state-out scores.csv.gz] [--json-out scores.json]

Read-only. Needs torch, torch_geometric, scikit-learn and numpy.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import OrderedDict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
sys.path.insert(0, TRAINING_DIR)

import torch  # noqa: E402

try:
    from StateParser import json_to_pyg_graph, NODE_DIM, GLOBAL_DIM  # noqa: E402
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(f"ERROR: could not import training/StateParser.py ({e}).\n"
             f"       It needs torch_geometric. Create the environment with "
             f"scripts/setup_python_env.sh and activate it first.")

from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

# Index of the prestige clock in the global vector (see StateParser).
PRESTIGE_CLOCK_GLOBAL_INDEX = 13
# Same buckets as train_local.py, so scores here match its validation output.
PRESTIGE_BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, float("inf"))]

EPS = 1e-7


def bucket_index(prestige_clock):
    for i, (lo, hi) in enumerate(PRESTIGE_BUCKETS):
        if lo <= prestige_clock < hi:
            return i
    return len(PRESTIGE_BUCKETS) - 1


def bucket_label(i):
    lo, hi = PRESTIGE_BUCKETS[i]
    return f"[{lo:.2f}, inf)" if hi == float("inf") else f"[{lo:.2f}, {hi:.2f})"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class DeployedValueNetwork(torch.nn.Module):
    """TributeValueNetwork's three MLPs, wired as in the exported ONNX graph
    (a plain per-graph mean in place of global_mean_pool). Built from the
    checkpoint's tensor shapes, so a mismatched input dimension fails here."""

    def __init__(self, state_dict):
        super().__init__()
        self.node_encoder = self._build(state_dict, "node_encoder")
        self.global_encoder = self._build(state_dict, "global_encoder")
        self.evaluator = self._build(state_dict, "evaluator")

    @staticmethod
    def _build(state_dict, prefix):
        # Layer indices are the nn.Sequential positions in ValueNetwork.py;
        # ReLU carries no parameters, so only Linear layers appear in the
        # state_dict and the gaps in the index sequence are where the ReLUs go.
        indices = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith(prefix + ".")})
        if not indices:
            raise ValueError(f"checkpoint has no '{prefix}.*' tensors")
        layers = []
        for pos, idx in enumerate(indices):
            weight = state_dict[f"{prefix}.{idx}.weight"]
            out_features, in_features = weight.shape
            layers.append(torch.nn.Linear(in_features, out_features))
            if pos < len(indices) - 1 or prefix != "evaluator":
                layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)

    def forward(self, x, batch_index, num_graphs, u):
        h = self.node_encoder(x)
        pooled = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype)
        pooled.index_add_(0, batch_index, h)
        counts = torch.zeros(num_graphs, dtype=h.dtype)
        counts.index_add_(0, batch_index, torch.ones(batch_index.shape[0], dtype=h.dtype))
        pooled = pooled / counts.unsqueeze(1)
        g = self.global_encoder(u)
        return self.evaluator(torch.cat([pooled, g], dim=1)).view(-1)


class DeployedFlatNetwork(torch.nn.Module):
    """The flat-MLP network behind DeployedValueNetwork's forward signature,
    so both kinds can be scored in one pass. Padding goes through StateParserFlat.

    arch must be given: matched and matched_sorted have identical shapes and
    differ only in sorting, which the state_dict does not record.
    """

    def __init__(self, state_dict, arch):
        super().__init__()
        from StateParserFlat import FLAT_DIM
        from ValueNetworkFlat import SORTED_ARCHS
        indices = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("mlp.")})
        layers = []
        for pos, idx in enumerate(indices):
            out_features, in_features = state_dict[f"mlp.{idx}.weight"].shape
            layers.append(torch.nn.Linear(in_features, out_features))
            if pos < len(indices) - 1:
                layers.append(torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(*layers)
        self.flat_dim = FLAT_DIM
        self.arch = arch
        self.sort = arch in SORTED_ARCHS

    def forward(self, x, batch_index, num_graphs, u):
        from StateParserFlat import batch_pad_and_flatten
        z = batch_pad_and_flatten(x, batch_index, u, num_graphs=num_graphs,
                                  sort=self.sort)
        return self.mlp(z).view(-1)


def resolve_flat_arch(path, overrides):
    """(arch, source) for a flat checkpoint, or (None, reason).

    run_config.json is read from beside the checkpoint's real path, since
    checkpoints are often scored through symlinks.
    """
    for key in (path, os.path.realpath(path)):
        if key in overrides:
            return overrides[key], "--arch override"
    cfg = os.path.join(os.path.dirname(os.path.realpath(path)), "run_config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                arch = json.load(f).get("arch")
        except (OSError, ValueError) as e:
            return None, f"{cfg} could not be read ({e})"
        if arch:
            return arch, f"run_config.json ({cfg})"
        return None, f"{cfg} has no 'arch' key"
    return None, f"no run_config.json beside {os.path.realpath(path)}"


def load_checkpoint(path, overrides=None):
    """Returns (model, arch, arch_source). arch is "deepsets" for the set model."""
    overrides = overrides or {}
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"expected a state_dict, got {type(state_dict).__name__}")
    # Unwrap a checkpoint that nests its weights under a "state_dict" key.
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # train_flat.py saves a FlatGraphAdapter, whose keys carry a "flat." prefix.
    if any(k.startswith("flat.mlp.") for k in state_dict):
        state_dict = {k[len("flat."):]: v for k, v in state_dict.items()
                      if k.startswith("flat.")}

    # A flat checkpoint has mlp.* keys and no node_encoder.*.
    if any(k.startswith("mlp.") for k in state_dict):
        from StateParserFlat import FLAT_DIM
        from ValueNetworkFlat import FLAT_CONFIGS

        arch, source = resolve_flat_arch(path, overrides)
        if arch is None:
            raise ValueError(
                f"this is a flat-MLP checkpoint and its arch could not be determined: "
                f"{source}.\n"
                f"  The arch is NOT recoverable from the weights. 'matched' and "
                f"'matched_sorted' have identical widths and identical parameter counts; "
                f"they differ only in whether the node rows are sorted before "
                f"flattening, which is not stored in the state_dict.\n"
                f"  Scoring the wrong one produces a plausible, wrong number rather than "
                f"an error, so this refuses to guess.\n"
                f"  Fix it by keeping run_config.json beside the checkpoint (train_flat.py "
                f"writes one), or pass --arch {path}=<arch>.\n"
                f"  Known archs: {', '.join(sorted(FLAT_CONFIGS))}")
        if arch not in FLAT_CONFIGS:
            raise ValueError(
                f"arch {arch!r} (from {source}) is not a known flat arch. "
                f"Known: {', '.join(sorted(FLAT_CONFIGS))}")

        model = DeployedFlatNetwork(state_dict, arch)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(f"state_dict does not match the flat network "
                             f"(missing={list(missing)}, unexpected={list(unexpected)})")
        got = model.mlp[0].in_features
        if got != FLAT_DIM:
            raise ValueError(
                f"flat checkpoint expects a {got}-dim input but the current flat schema "
                f"(training/StateParserFlat.py, MAX_NODES={FLAT_DIM // NODE_DIM}) produces "
                f"{FLAT_DIM}. This checkpoint was trained against a different MAX_NODES -- "
                f"the numbers would be meaningless.")
        model.eval()
        return model, arch, source

    model = DeployedValueNetwork(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"state_dict does not match the network "
                         f"(missing={list(missing)}, unexpected={list(unexpected)})")
    first = model.node_encoder[0]
    if first.in_features != NODE_DIM:
        raise ValueError(f"checkpoint expects {first.in_features}-dim node features but the current "
                         f"schema (training/StateParser.py) produces {NODE_DIM}. This checkpoint "
                         f"predates the schema it would be scored on -- the numbers would be "
                         f"meaningless.")
    if model.global_encoder[0].in_features != GLOBAL_DIM:
        raise ValueError(f"checkpoint expects {model.global_encoder[0].in_features}-dim global "
                         f"features but the current schema produces {GLOBAL_DIM}.")
    model.eval()
    return model, "deepsets", "state_dict keys (node_encoder.*)"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def iter_samples(data_dir, limit):
    """Streams (outcome, game_id, graph) from every *.jsonl.gz under data_dir,
    in sorted shard order, so repeated runs see the same order. game_id is kept
    for the game-clustered tests in clustered_significance.py."""
    shards = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl.gz"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"No *.jsonl.gz shards found under {data_dir}")
    print(f"Dataset        : {data_dir}")
    print(f"Shards         : {len(shards)}")

    n = 0
    skipped = 0
    first_error = None
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                if limit and n >= limit:
                    if skipped:
                        print(f"  [WARN] skipped {skipped} malformed record(s); first: {first_error}")
                    return
                try:
                    row = json.loads(line)
                    graph = json_to_pyg_graph(row["data"]["state"])
                    yield int(row["outcome"]), row.get("game_id", ""), graph
                    n += 1
                except Exception as e:
                    skipped += 1
                    if first_error is None:
                        first_error = f"{shard}: {e!r}"
    if skipped:
        print(f"  [WARN] skipped {skipped} malformed record(s); first: {first_error}")


def collate(batch):
    xs = [g.x for _, _, g in batch]
    us = [g.u for _, _, g in batch]
    batch_index = torch.cat([torch.full((x.shape[0],), i, dtype=torch.long)
                             for i, x in enumerate(xs)])
    return torch.cat(xs, dim=0), batch_index, len(batch), torch.cat(us, dim=0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(labels, probs):
    """labels/probs: 1D float arrays over one slice. Returns None for an empty
    slice rather than raising, so an unpopulated bucket prints as '--'."""
    if len(labels) == 0:
        return None
    p = np.clip(probs, EPS, 1 - EPS)
    loss = float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())
    acc = float(((probs >= 0.5).astype(np.float64) == labels).mean())
    majority = 1.0 if labels.mean() >= 0.5 else 0.0
    baseline = float((labels == majority).mean())
    brier = float(brier_score_loss(labels, probs))
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    return {"n": int(len(labels)), "loss": loss, "acc": acc, "auc": auc,
            "brier": brier, "baseline": baseline, "base_rate": float(labels.mean())}


def fmt_row(label, m):
    if m is None:
        return f"  {label:<14}{0:>8}{'--':>10}{'--':>9}{'--':>9}{'--':>9}{'--':>10}"
    auc = f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "n/a"
    return (f"  {label:<14}{m['n']:>8}{m['loss']:>10.4f}{m['acc'] * 100:>8.2f}%"
            f"{auc:>9}{m['brier']:>9.4f}{m['baseline'] * 100:>9.2f}%")


def print_model_report(name, metrics_overall, metrics_by_bucket):
    print()
    print("-" * 78)
    print(f"{name}")
    print("-" * 78)
    print(f"  {'slice':<14}{'n':>8}{'loss':>10}{'acc':>9}{'auc':>9}{'brier':>9}{'baseline':>10}")
    print(fmt_row("overall", metrics_overall))
    for i in range(len(PRESTIGE_BUCKETS)):
        print(fmt_row(bucket_label(i), metrics_by_bucket.get(i)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+",
                        help="One or more .pth checkpoints. Scored on the same samples, in one pass.")
    parser.add_argument("--data-dir", required=True,
                        help="Directory of *.jsonl.gz shards (searched recursively) -- the common "
                             "evaluation set")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to score (0 = all). Applied before any model runs, so every "
                             "model still sees the same samples.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--arch", action="append", default=[], metavar="PATH=ARCH",
                        help="Force a flat checkpoint's architecture, overriding its "
                             "run_config.json. Repeatable. PATH may be the path as given "
                             "or its realpath. Needed only when run_config.json is absent "
                             "-- the arch cannot be read off the weights, because "
                             "'matched' and 'matched_sorted' have identical shapes and "
                             "differ only in whether rows are sorted before flattening.")
    parser.add_argument("--json-out", default=None, help="Also write the metrics to this JSON file")
    parser.add_argument("--per-state-out", default=None, metavar="PATH.csv.gz",
                        help="Also write one gzipped CSV row per scored state: game_id, "
                             "target, prestige_clock, bucket, and each checkpoint's "
                             "predicted probability. Written streaming, so memory does not "
                             "grow with the dataset. This is the input to "
                             "tools/clustered_significance.py, which needs game_id to "
                             "cluster on -- the aggregate metrics above cannot support a "
                             "significance claim on their own.")
    args = parser.parse_args()

    arch_overrides = {}
    for spec in args.arch:
        if "=" not in spec:
            sys.exit(f"ERROR: --arch wants PATH=ARCH, got {spec!r}")
        ap, av = spec.rsplit("=", 1)
        arch_overrides[ap] = av
        arch_overrides[os.path.realpath(ap)] = av

    if args.per_state_out and not args.per_state_out.endswith(".gz"):
        sys.exit("ERROR: --per-state-out must end in .gz (the file is written gzipped; "
                 "one row per state is large).")

    torch.set_grad_enabled(False)
    # Metrics must not depend on how many cores happen to be free.
    torch.set_num_threads(1)

    models = OrderedDict()
    for path in args.checkpoints:
        # Use as many trailing path components as it takes to make the name unique;
        # <arch>/seed_00/best_model.pth collides on the last two.
        parts = os.path.splitext(os.path.normpath(os.path.abspath(path)))[0].split(os.sep)
        name = parts[-1]
        for depth in range(2, len(parts) + 1):
            if name not in models:
                break
            name = os.path.join(*parts[-depth:])
        if name in models:
            sys.exit(f"ERROR: two checkpoints resolve to the same report name {name!r}. "
                     f"Pass them from distinct paths.")
        if not os.path.isfile(path):
            sys.exit(f"ERROR: checkpoint not found: {path}")
        try:
            model, arch, arch_source = load_checkpoint(path, arch_overrides)
            models[name] = {"path": path, "model": model, "arch": arch,
                            "arch_source": arch_source, "logits": []}
        except Exception as e:
            sys.exit(f"ERROR: could not load {path}: {e}")

    print("=" * 78)
    print("CHECKPOINT EVALUATION")
    print("=" * 78)
    for name, entry in models.items():
        print(f"  {name:<28} {entry['path']}")
        # A flat arch is not recoverable from the weights, so show where it came from.
        print(f"  {'':<28} arch: {entry['arch']}  (from {entry['arch_source']})")
    print()

    labels, clocks = [], []
    batch = []
    n_scored = 0

    # Per-state rows are streamed as each batch is scored. Ten significant digits
    # keep every float32 bit, so clustered_significance.py reproduces these losses.
    per_state = None
    if args.per_state_out:
        per_state = gzip.open(args.per_state_out, "wt", newline="")
        # A '#' preamble records each column's arch, which the column name does not;
        # clustered_significance.py skips these lines.
        for n, e in models.items():
            per_state.write(f"# {n}\tarch={e['arch']}\tsource={e['arch_source']}"
                            f"\tpath={e['path']}\n")
        per_state.write(",".join(
            ["game_id", "target", "prestige_clock", "bucket"]
            + [f"p_{n}" for n in models]) + "\n")

    def flush(batch):
        nonlocal n_scored
        if not batch:
            return
        x, batch_index, num_graphs, u = collate(batch)
        batch_logits = {}
        for name, entry in models.items():
            lg = entry["model"](x, batch_index, num_graphs, u).numpy()
            entry["logits"].append(lg)
            batch_logits[name] = lg
        for i, (outcome, game_id, graph) in enumerate(batch):
            clock = float(graph.u[0, PRESTIGE_CLOCK_GLOBAL_INDEX])
            labels.append(float(outcome))
            clocks.append(clock)
            if per_state is not None:
                probs = [1.0 / (1.0 + np.exp(-float(batch_logits[n][i])))
                         for n in models]
                per_state.write(
                    f"{game_id},{int(outcome)},{clock:.10g},"
                    f"{bucket_index(clock)},"
                    + ",".join(f"{p:.10g}" for p in probs) + "\n")
        n_scored += len(batch)
        if n_scored % (args.batch_size * 20) == 0:
            print(f"  ... {n_scored} samples scored")

    try:
        for outcome, game_id, graph in iter_samples(args.data_dir, args.limit):
            batch.append((outcome, game_id, graph))
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
        flush(batch)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    finally:
        if per_state is not None:
            per_state.close()

    if n_scored == 0:
        sys.exit("ERROR: no samples were scored -- is --data-dir the right directory?")

    labels = np.array(labels, dtype=np.float64)
    clocks = np.array(clocks, dtype=np.float64)
    bucket_of = np.array([bucket_index(c) for c in clocks], dtype=np.int64)

    print()
    print(f"Samples scored : {n_scored}")
    print(f"Label base rate: {labels.mean():.4f}  (P[the logging player won])")
    print(f"Bucket sizes   : " + ", ".join(
        f"{bucket_label(i)}={int((bucket_of == i).sum())}" for i in range(len(PRESTIGE_BUCKETS))))

    report = {"data_dir": os.path.abspath(args.data_dir), "n": n_scored,
              "base_rate": float(labels.mean()), "models": {}}

    for name, entry in models.items():
        logits = np.concatenate(entry["logits"])
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        overall = compute_metrics(labels, probs)
        by_bucket = {}
        for i in range(len(PRESTIGE_BUCKETS)):
            mask = bucket_of == i
            by_bucket[i] = compute_metrics(labels[mask], probs[mask])
        print_model_report(name, overall, by_bucket)
        report["models"][name] = {
            "path": os.path.abspath(entry["path"]),
            "real_path": os.path.realpath(entry["path"]),
            "arch": entry["arch"],
            "arch_source": entry["arch_source"],
            "overall": overall,
            "by_prestige_bucket": {bucket_label(i): by_bucket[i] for i in by_bucket},
        }

    print()
    print("=" * 78)
    print("SIDE BY SIDE (overall)")
    print("=" * 78)
    header = f"  {'checkpoint':<30}{'loss':>10}{'acc':>9}{'auc':>9}{'brier':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, data in report["models"].items():
        m = data["overall"]
        auc = f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "n/a"
        print(f"  {name:<30}{m['loss']:>10.4f}{m['acc'] * 100:>8.2f}%{auc:>9}{m['brier']:>9.4f}")
    majority_baseline = max(labels.mean(), 1 - labels.mean())
    print(f"  {'(majority-class baseline)':<30}{'--':>10}{majority_baseline * 100:>8.2f}%{'--':>9}{'--':>9}")
    print()
    print("  A checkpoint whose accuracy is not clearly above the majority-class baseline has not")
    print("  learned anything usable from this set, whatever its loss looks like.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
        print()
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
