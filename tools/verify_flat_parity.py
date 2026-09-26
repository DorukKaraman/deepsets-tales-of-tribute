"""
Parity test for the flat encoder's ASSEMBLY step, on real logged states.

WHAT IT DOES AND DOES NOT COVER. There is no C# side here yet and no
reimplemented feature code to check: training/StateParserFlat.py takes its node
matrix straight from StateParser.json_to_pyg_graph and its globals from the same
object, so the two encoders agree card-for-card by construction. What is new,
and therefore what can be wrong, is the assembly -- pad to MAX_NODES rows,
flatten, concatenate the 19 globals -- and the three separate places that
assembly happens:

  1. pad_and_flatten          one state, the reference definition
  2. batch_pad_and_flatten    a PyG batch, the training path (to_dense_batch)
  3. FlatONNXWrapper          the exported graph, in traced ONNX ops

All three must produce the same vector or the model is trained on one layout
and served on another -- a failure that would show up as an inexplicably weak
baseline and would be read as evidence about architecture. #3 is checked
against #1 inside export_flat_to_onnx.verify_flat_export on every export, so it
cannot reach disk unverified. This script checks #1 against the underlying
StateParser output, and #2 against #1, on real states rather than random
tensors -- random tensors have no zero rows, and the padding is precisely where
zeros matter.

Checks per state:
  - every node row lands at its own index, in json_to_pyg_graph's emission order
  - rows past the node count are exactly zero
  - the 19 globals occupy the final slots, in order
  - nothing else is nonzero
  - the batched path is bit-identical to the per-state path
  - batches mixing different node counts pad each member independently

When this script is run after a C# FlatFeatureExtractor exists, it should be
extended to diff against it the way tools/verify_parity.py does -- by invoking
the real C# method through a console tool, not by reimplementing it here.

Read-only. Usage:
    python tools/verify_flat_parity.py --data-dir /path/to/data --num-samples 300
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))

try:
    import torch  # noqa: E402
    from torch_geometric.data import Batch  # noqa: E402
    from StateParser import NODE_DIM, GLOBAL_DIM, json_to_pyg_graph  # noqa: E402
    from StateParserFlat import (  # noqa: E402
        FLAT_DIM, FLAT_NODE_DIM, MAX_NODES,
        batch_pad_and_flatten, json_to_flat_vector, pad_and_flatten,
    )
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(f"ERROR: could not import the training modules ({e}).\n"
             f"       They need torch and torch_geometric. Activate the venv built by "
             f"scripts/setup_python_env.sh first.")


def sample_states(data_dir, num_samples, seed=0):
    """Spread the sample across shards AND across each shard.

    Consecutive records come from one game and share a board, so the first N
    records of one shard would exercise one deck and a narrow band of node
    counts -- which is the one thing this test must not do, since the padding
    region is exactly what varies with the node count.

    Reservoir sampling per shard rather than a fixed stride: shards here range
    from a few hundred records to 25,000, and any fixed stride is either too
    coarse for the small ones (returning almost nothing) or too fine for the
    large ones (returning only the head). A reservoir spreads over the whole
    shard at any length, in one pass, holding at most per_shard states.
    """
    shards = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl.gz"), recursive=True))
    if not shards:
        sys.exit(f"ERROR: no *.jsonl.gz shards found under {data_dir}")

    per_shard = max(1, -(-num_samples // len(shards)))
    rng = random.Random(seed)
    states = []
    for shard in shards:
        if len(states) >= num_samples:
            break
        reservoir = []
        with gzip.open(shard, "rt") as f:
            for i, line in enumerate(f):
                if i < per_shard:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, i)
                    if j < per_shard:
                        reservoir[j] = line
        for line in reservoir:
            if len(states) >= num_samples:
                break
            states.append(json.loads(line)["data"]["state"])
    return states


def check_layout(state):
    """Returns a list of failure strings for one state (empty == passed)."""
    graph = json_to_pyg_graph(state)
    x, u = graph.x, graph.u
    n = x.shape[0]
    flat = json_to_flat_vector(state)
    problems = []

    if flat.shape != (FLAT_DIM,):
        return [f"wrong shape {tuple(flat.shape)}, expected ({FLAT_DIM},)"]
    if flat.dtype != torch.float32:
        problems.append(f"dtype {flat.dtype}, expected torch.float32")

    kept = min(n, MAX_NODES)

    # Each node row at its own index, in emission order.
    node_part = flat[:FLAT_NODE_DIM].reshape(MAX_NODES, NODE_DIM)
    if not torch.equal(node_part[:kept], x[:kept]):
        bad = (node_part[:kept] != x[:kept]).any(dim=1).nonzero().flatten().tolist()
        problems.append(f"node rows differ at indices {bad[:5]} (n={n})")

    # Everything past the node count is padding and must be exactly zero --
    # not merely small. A nonzero there would be read by the model as a card.
    if kept < MAX_NODES and node_part[kept:].abs().sum().item() != 0.0:
        nz = int((node_part[kept:] != 0).sum().item())
        problems.append(f"{nz} nonzero values in the padding region (n={n})")

    # Globals in the final GLOBAL_DIM slots, in order.
    if not torch.equal(flat[FLAT_NODE_DIM:], u.reshape(-1)):
        problems.append(f"global block differs (n={n})")

    # Node count sanity against the corpus the cap was measured on.
    if n > MAX_NODES:
        problems.append(f"NOTE n={n} exceeds MAX_NODES={MAX_NODES} and was truncated")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="A generate_data.py or split_dataset.py output directory")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the per-shard reservoir, so a failure is reproducible.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print(f"MAX_NODES={MAX_NODES}  NODE_DIM={NODE_DIM}  GLOBAL_DIM={GLOBAL_DIM}  "
          f"FLAT_DIM={FLAT_DIM}")
    states = sample_states(args.data_dir, args.num_samples, args.seed)
    print(f"Sampled {len(states)} state(s) from {args.data_dir}")
    if not states:
        sys.exit("ERROR: no states sampled")

    failures = 0
    node_counts = []
    for i, state in enumerate(states):
        node_counts.append(json_to_pyg_graph(state).x.shape[0])
        for problem in check_layout(state):
            failures += 1
            if failures <= 10:
                print(f"  FAIL state {i}: {problem}")

    print(f"\n1. Layout vs StateParser, {len(states)} states: "
          f"{'PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    print(f"   node counts seen: min {min(node_counts)}, median "
          f"{sorted(node_counts)[len(node_counts) // 2]}, max {max(node_counts)}")

    # 2. The training path against the reference, on batches that deliberately
    # mix node counts -- a padding bug that pads every member to the batch's
    # own maximum instead of MAX_NODES is invisible on a uniform batch.
    graphs = [json_to_pyg_graph(s) for s in states]
    for g in graphs:
        g.y = torch.zeros(1, 1)
    batch_failures = 0
    checked = 0
    for start in range(0, len(graphs), args.batch_size):
        chunk = graphs[start:start + args.batch_size]
        if not chunk:
            continue
        batch = Batch.from_data_list(chunk)
        got = batch_pad_and_flatten(batch.x, batch.batch, batch.u,
                                    num_graphs=len(chunk))
        want = torch.stack([pad_and_flatten(g.x, g.u) for g in chunk])
        if got.shape != want.shape:
            batch_failures += 1
            print(f"  FAIL batch at {start}: shape {tuple(got.shape)} != {tuple(want.shape)}")
            continue
        if not torch.equal(got, want):
            batch_failures += 1
            rows = (got != want).any(dim=1).nonzero().flatten().tolist()
            worst = (got - want).abs().max().item()
            if batch_failures <= 5:
                print(f"  FAIL batch at {start}: members {rows[:5]} differ, "
                      f"max abs {worst:.3e}")
        checked += len(chunk)

    spread = len({g.x.shape[0] for g in graphs[:args.batch_size]})
    print(f"2. Batched path vs per-state path, {checked} states in batches of "
          f"{args.batch_size}: {'PASS' if batch_failures == 0 else f'{batch_failures} FAILURE(S)'}")
    print(f"   (first batch spans {spread} distinct node count(s) -- a batch of one "
          f"count would not test padding)")

    total = failures + batch_failures
    print()
    if total:
        sys.exit(f"FAILED: {total} problem(s). The flat encoder does not agree with "
                 f"StateParser; do not train on it.")
    print("PASSED: the flat encoding agrees with StateParser on every sampled state, "
          "and the\n        training path is bit-identical to the reference.")
    print("        The exported ONNX graph is checked against the reference separately, "
          "on every\n        export, by export_flat_to_onnx.verify_flat_export.")


if __name__ == "__main__":
    main()
