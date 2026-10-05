"""
Check the flat encoder's assembly on real logged states.

StateParserFlat takes its node matrix from StateParser.json_to_pyg_graph, so the
card features cannot differ; what can go wrong is the assembly (pad to
MAX_NODES, flatten, append the globals), which has three implementations:

  1. pad_and_flatten          one state, the reference
  2. batch_pad_and_flatten    a PyG batch, the training path
  3. FlatONNXWrapper          the exported graph

This checks 1 against StateParser and 2 against 1; export_flat_to_onnx.py checks
3 against 1 on every export. Real states are used because random tensors have
no zero rows, and padding is where zeros matter.

For the sorted arm it also checks that the sort key separates every distinct
row (otherwise argsort's tie-breaking decides, and PyTorch and onnxruntime may
disagree), that location stays the primary key, that the result does not
depend on row order, and that the batched path matches.

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
        FLAT_DIM, FLAT_NODE_DIM, MAX_NODES, SORT_KEY_LOC_SCALE,
        batch_pad_and_flatten, json_to_flat_vector, pad_and_flatten, row_sort_key,
    )
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(f"ERROR: could not import the training modules ({e}).\n"
             f"       They need torch and torch_geometric. Activate the venv built by "
             f"scripts/setup_python_env.sh first.")


def sample_states(data_dir, num_samples, seed=0):
    """Sample across shards and across each shard, so the sample covers many
    node counts; consecutive records share a game and a board. Reservoir sampling
    per shard works at any shard length, in one pass.
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

    # Padding must be exactly zero; a nonzero value would read as a card.
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

    # 2. The training path against the reference, on batches with mixed node
    # counts, which expose padding to the batch's own maximum instead of MAX_NODES.
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

    # 3. The sorted arm: the key must separate distinct rows, the result must not
    #    depend on row order, and the batched path must match the reference.
    sort_failures = 0

    distinct = {tuple(r.tolist()) for g in graphs for r in g.x}
    R = torch.tensor(sorted(distinct), dtype=torch.float32)
    keys = row_sort_key(R)
    n_unique = len(torch.unique(keys))
    if n_unique != len(R):
        sort_failures += 1
        print(f"  FAIL sort key collides: {len(R) - n_unique} of {len(R)} distinct rows "
              f"share a key with another distinct row.")
        print(f"       Distinct rows with equal keys are ordered by argsort's internal "
              f"tie-breaking, which PyTorch and onnxruntime need not resolve the same "
              f"way -- expect a torch/ONNX mismatch on some states and not others.")
        print(f"       Raise SORT_KEY_LOC_SCALE, or reseed SORT_KEY_SEED, in "
              f"training/StateParserFlat.py.")

    # Location must stay the primary key, or the vector loses its block structure.
    loc_ids = (R[:, NODE_DIM - 9:].to(torch.float64) @ torch.arange(9, dtype=torch.float64))
    bands = torch.floor(keys / SORT_KEY_LOC_SCALE)
    if not torch.equal(bands, loc_ids):
        sort_failures += 1
        print(f"  FAIL location is no longer the primary sort key -- "
              f"SORT_KEY_LOC_SCALE is too small for the projection's range.")

    perm_failures = 0
    rng = random.Random(args.seed)
    for g in graphs:
        ref = pad_and_flatten(g.x, g.u, sort=True)
        for _ in range(4):
            order = list(range(g.x.shape[0]))
            rng.shuffle(order)
            got = pad_and_flatten(g.x[order], g.u, sort=True)
            if not torch.equal(got, ref):
                perm_failures += 1
                break
    if perm_failures:
        sort_failures += perm_failures
        print(f"  FAIL {perm_failures} state(s) give a different flattened vector under "
              f"row permutation, so the sorted arm is not permutation-invariant.")

    sorted_batch_failures = 0
    for start in range(0, len(graphs), args.batch_size):
        chunk = graphs[start:start + args.batch_size]
        if not chunk:
            continue
        batch = Batch.from_data_list(chunk)
        got = batch_pad_and_flatten(batch.x, batch.batch, batch.u,
                                    num_graphs=len(chunk), sort=True)
        want = torch.stack([pad_and_flatten(g.x, g.u, sort=True) for g in chunk])
        if got.shape != want.shape or not torch.equal(got, want):
            sorted_batch_failures += 1
    if sorted_batch_failures:
        sort_failures += sorted_batch_failures
        print(f"  FAIL {sorted_batch_failures} sorted batch(es) differ from the "
              f"per-state reference.")

    print(f"3. Canonical order (matched_sorted): "
          f"{'PASS' if sort_failures == 0 else f'{sort_failures} FAILURE(S)'}")
    print(f"   sort key separates {n_unique}/{len(R)} distinct rows; "
          f"{len(graphs)} states x 4 permutations invariant; "
          f"batched path matches per-state")

    total = failures + batch_failures + sort_failures
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
