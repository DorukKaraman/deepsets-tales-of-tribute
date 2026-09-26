"""
Score any number of .pth checkpoints on ONE common dataset directory.

Reports, per checkpoint: BCE loss, accuracy, ROC AUC and Brier score, overall
and bucketed by prestige clock, each alongside that slice's majority-class
baseline accuracy. Ends with a side-by-side table so the checkpoints can be
ranked at a glance.

WHAT THIS IS FOR: comparing per-seed models (scripts/slurm_train.sh) against
each other and against the shipped one, on data none of them were trained on.
Generate that data with

    tools/generate_data.sh --games N --out-dir <dir> \\
        --bot-a DeepSetsBotExp --bot-b SakkirinaSolo --seed-base <n>

Neither shipped model was trained on games between those two agents, so the
resulting set is genuinely held out -- which a fresh self-play set from the
same generator would not be, however new its games are.

ONE PASS, ALL MODELS. The dataset is streamed once and every checkpoint is
evaluated on each batch as it goes, rather than re-read per checkpoint. That is
not (only) about speed: it is what guarantees all checkpoints are scored on
byte-identical samples in identical order, which is the entire point of a
common evaluation set. Memory stays bounded by --batch-size, so a
multi-gigabyte directory does not have to fit in RAM.

BOTH ARCHITECTURES, ONE TABLE. DeepSets checkpoints (training/ValueNetwork.py)
and flat-MLP ablation checkpoints (training/ValueNetworkFlat.py) can be passed
in the same invocation; the architecture is detected from the checkpoint's own
keys, not from a flag, so the two cannot be scored through each other's forward
pass by mistake. Mixing them is the intended use: the ablation's question is
whether the set structure contributes, and the only honest way to ask it is to
score both on byte-identical samples rather than to compare two separate
validation passes.

THE FORWARD PASS HERE IS THE EXPORTED ONE, not TributeValueNetwork.forward.
The two differ in exactly one place: this takes a plain per-graph mean over
nodes where the training model calls torch_geometric's global_mean_pool. They
are arithmetically identical for a single graph, and the plain mean is what
training/export_to_onnx.py puts in the ONNX file -- i.e. what the agent
actually runs. Scoring a checkpoint through the path it will be deployed on is
the useful measurement; see export_to_onnx.py's ONNXWrapper for why the export
bypasses global_mean_pool in the first place (opset 14 lowers scatter-reduce
without a reduction attribute, silently giving replace semantics instead of an
average).

Read-only. Does not modify any checkpoint or data file.

Requirements: torch, torch_geometric (for training/StateParser.py's Data
container), scikit-learn, numpy. See scripts/setup_python_env.sh.
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

# Global feature index 13 is the prestige clock (see
# StateParser.extract_global_context). A named constant, not a bare literal,
# because this index already drifted once: it moved 16 -> 13 when TREASURY was
# dropped from the patron favour block, and tools/diagnose_value_net.py kept
# reading column 16 -- silently the wrong column, not a crash -- until a full
# dry run caught the mismatched bucket counts.
PRESTIGE_CLOCK_GLOBAL_INDEX = 13
# Identical to training/train_local.py's, on purpose: a checkpoint's score here
# has to be comparable to the validation numbers printed while it trained.
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
    """TributeValueNetwork's three MLPs, wired the way the exported ONNX graph
    wires them (plain per-graph mean instead of global_mean_pool). Built from
    the checkpoint's own tensor shapes rather than hardcoded dimensions, so a
    checkpoint trained at a different hidden size still loads -- and one whose
    input dimension does not match the current feature schema fails loudly here
    instead of scoring garbage."""

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
    """The flat-MLP ablation (training/ValueNetworkFlat.py), behind the same
    forward signature as DeployedValueNetwork so both can be scored side by
    side on one pass of one dataset.

    That side-by-side is the whole point: a flat model and a DeepSets model
    compared from their own separate validation passes are comparable only as
    far as the two passes happened to align, whereas here every model sees
    byte-identical samples in identical order. Since the flat architecture is
    the thing under test, that distinction is not a technicality.

    Padding and flattening go through StateParserFlat.pad_and_flatten rather
    than being rebuilt here, so this scores the layout the model was actually
    trained on.
    """

    def __init__(self, state_dict):
        super().__init__()
        from StateParserFlat import FLAT_DIM
        indices = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("mlp.")})
        layers = []
        for pos, idx in enumerate(indices):
            out_features, in_features = state_dict[f"mlp.{idx}.weight"].shape
            layers.append(torch.nn.Linear(in_features, out_features))
            if pos < len(indices) - 1:
                layers.append(torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(*layers)
        self.flat_dim = FLAT_DIM

    def forward(self, x, batch_index, num_graphs, u):
        from StateParserFlat import batch_pad_and_flatten
        z = batch_pad_and_flatten(x, batch_index, u, num_graphs=num_graphs)
        return self.mlp(z).view(-1)


def load_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"expected a state_dict, got {type(state_dict).__name__}")
    # A full-object checkpoint (torch.save(model)) or a training checkpoint with
    # the weights nested under a key -- unwrap the common shapes rather than
    # failing on them.
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # train_flat.py trains a FlatGraphAdapter wrapping the MLP, so its keys are
    # prefixed "flat." -- strip it, as export_flat_to_onnx.py does.
    if any(k.startswith("flat.mlp.") for k in state_dict):
        state_dict = {k[len("flat."):]: v for k, v in state_dict.items()
                      if k.startswith("flat.")}

    # Dispatch on the checkpoint's own keys. A flat checkpoint has mlp.* and no
    # node_encoder.*; guessing wrong here would score one architecture through
    # the other's forward pass and report the result as an ablation.
    if any(k.startswith("mlp.") for k in state_dict):
        from StateParserFlat import FLAT_DIM
        model = DeployedFlatNetwork(state_dict)
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
        return model

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
    return model


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def iter_samples(data_dir, limit):
    """Streams (outcome, game_id, graph) from every *.jsonl.gz under data_dir,
    in sorted shard order. Deterministic and unshuffled on purpose: every metric
    here is order-independent, and a fixed order is what makes two invocations
    of this script directly comparable.

    game_id rides along because states within one game are highly correlated
    and any honest significance test has to cluster on it -- see
    tools/clustered_significance.py, which consumes --per-state-out."""
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
    if args.per_state_out and not args.per_state_out.endswith(".gz"):
        sys.exit("ERROR: --per-state-out must end in .gz (the file is written gzipped; "
                 "one row per state is large).")

    torch.set_grad_enabled(False)
    # Metrics must not depend on how many cores happen to be free.
    torch.set_num_threads(1)

    models = OrderedDict()
    for path in args.checkpoints:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in models:
            # Two checkpoints with the same basename from different directories
            # would otherwise silently overwrite each other in the report.
            name = os.path.join(os.path.basename(os.path.dirname(path)), name)
        if not os.path.isfile(path):
            sys.exit(f"ERROR: checkpoint not found: {path}")
        try:
            models[name] = {"path": path, "model": load_checkpoint(path), "logits": []}
        except Exception as e:
            sys.exit(f"ERROR: could not load {path}: {e}")

    print("=" * 78)
    print("CHECKPOINT EVALUATION")
    print("=" * 78)
    for name, entry in models.items():
        print(f"  {name:<28} {entry['path']}")
    print()

    labels, clocks = [], []
    batch = []
    n_scored = 0

    # Per-state rows are written as each batch is scored and never accumulated,
    # so memory stays flat whatever the dataset size. Probabilities go out at
    # 10 significant digits: float32 carries ~7, so nothing is lost, and
    # tools/clustered_significance.py recomputes losses from these numbers and
    # must land on the same p-values as scoring in-process.
    per_state = None
    if args.per_state_out:
        per_state = gzip.open(args.per_state_out, "wt", newline="")
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
