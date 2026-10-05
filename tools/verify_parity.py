"""
Cross-language parity test for the value network feature schema.

Runs the production code on both sides: StateParser.json_to_pyg_graph in
Python, and Bots.FeatureExtractor.ParseState in C# through tools/ParityCheck,
which rebuilds a ScriptsOfTribute.Serializers.GameState from the logged JSON.

Samples states from a generate_data.py output directory (default /tmp/gen50),
spread evenly over four prestige-clock phases, and compares:
  - node matrix row count
  - node matrix contents, each side's rows sorted first
  - node matrix row order, unsorted, reported separately; the DeepSets model
    mean-pools and cannot see order, but the flat-MLP ablation
    (REPRODUCE.md section 8) is not permutation-invariant
  - global vector, element by element

Prints the max abs delta per column, labelled by feature block, whether or not
the test passes. Exits nonzero if any column exceeds DELTA_THRESHOLD.

Read-only against --data-dir. Builds tools/ParityCheck (Release) and writes only
to a temp directory.

Usage:
    python tools/verify_parity.py
    python tools/verify_parity.py --data-dir /path/to/gen50 --num-samples 50
"""
import argparse
import glob
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
PARITYCHECK_DIR = os.path.join(SCRIPT_DIR, "ParityCheck")
PARITYCHECK_CSPROJ = os.path.join(PARITYCHECK_DIR, "ParityCheck.csproj")

sys.path.insert(0, TRAINING_DIR)

DELTA_THRESHOLD = 1e-6

# NODE_DIM layout of FeatureExtractor in Bots/src/DeepSetsCore.cs.
NODE_FEATURE_BLOCKS = [
    (0, 7, "deck"),
    (7, 8, "cost"),
    (8, 12, "type"),
    (12, 13, "hp"),
    (13, 14, "taunt"),
    (14, 90, "effect"),
    (90, 99, "location"),
]

# Layout of StateParser.extract_global_context / FeatureExtractor.EncodeGlobalContext.
GLOBAL_FEATURE_NAMES = [
    "CurrentPlayer.Coins", "CurrentPlayer.Power", "CurrentPlayer.Prestige", "CurrentPlayer.PatronCalls",
    "EnemyPlayer.Coins", "EnemyPlayer.Power", "EnemyPlayer.Prestige",
    "Patron.ANSEI", "Patron.DUKE_OF_CROWS", "Patron.RAJHIN", "Patron.ORGNUM", "Patron.PELIN", "Patron.SAINT_ALESSIA",
    "PrestigeClock", "PrestigeDifferential", "MyDeckSize", "EnemyKnownDeckSize", "MyAgentCount", "EnemyAgentCount",
]

# The prestige-clock buckets tools/diagnose_value_net.py uses.
BUCKET_EDGES = [0.25, 0.5, 0.75, 1.2001]


def node_feature_label(i):
    for lo, hi, name in NODE_FEATURE_BLOCKS:
        if lo <= i < hi:
            return f"{name}[{i}]"
    return f"unknown[{i}]"


def build_parity_check(configuration):
    print(f"Building ParityCheck ({configuration})...", file=sys.stderr)
    subprocess.run(["dotnet", "build", PARITYCHECK_CSPROJ, "-c", configuration],
                    check=True, cwd=PARITYCHECK_DIR, stdout=subprocess.DEVNULL)
    dll = os.path.join(PARITYCHECK_DIR, "bin", configuration, "net8.0", "ParityCheck.dll")
    if not os.path.isfile(dll):
        sys.exit(f"ERROR: expected built ParityCheck.dll not found at {dll}")
    return dll


def prestige_clock_of(state):
    cp = state.get("CurrentPlayer", {})
    ep = state.get("EnemyPlayer", {})
    my_p = float(cp.get("Prestige", 0))
    en_p = float(ep.get("Prestige", 0))
    return min(max(my_p, en_p) / 40.0, 1.2)


def sample_states(data_dir, num_samples, seed):
    """Stream every record under data_dir, bucket by prestige clock, then
    sample evenly across buckets."""
    shards = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl.gz"), recursive=True))
    if not shards:
        sys.exit(f"ERROR: no shards found under {data_dir}")

    buckets = defaultdict(list)
    total = 0
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                rec = json.loads(line)
                state = rec["data"]["state"]
                pc = prestige_clock_of(state)
                bucket = next(i for i, edge in enumerate(BUCKET_EDGES) if pc < edge)
                buckets[bucket].append((rec["game_id"], rec["player"], state))
                total += 1

    print(f"Streamed {total} records from {len(shards)} shard(s); "
          f"bucket sizes: {[len(buckets.get(i, [])) for i in range(len(BUCKET_EDGES))]}")

    rng = random.Random(seed)
    per_bucket = max(1, num_samples // len(BUCKET_EDGES))
    sampled = []
    for i in range(len(BUCKET_EDGES)):
        pool = list(buckets.get(i, []))
        rng.shuffle(pool)
        sampled.extend(pool[:per_bucket])

    rng.shuffle(sampled)
    return sampled[:num_samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="/tmp/gen50")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--configuration", default="Release")
    args = parser.parse_args()

    from StateParser import json_to_pyg_graph, NODE_DIM, GLOBAL_DIM

    print(f"Sampling ~{args.num_samples} states from {args.data_dir}, spread across prestige-clock phases...")
    samples = sample_states(args.data_dir, args.num_samples, args.seed)
    print(f"Sampled {len(samples)} states.\n")

    cases = [{"id": f"{game_id}_{player}_{idx}", "state": state}
              for idx, (game_id, player, state) in enumerate(samples)]

    dll = build_parity_check(args.configuration)

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "input.json")
        output_path = os.path.join(tmp, "output.json")
        with open(input_path, "w") as f:
            json.dump(cases, f)

        print("Running Bots.FeatureExtractor.ParseState via ParityCheck...", file=sys.stderr)
        subprocess.run(["dotnet", dll, "dump", input_path, output_path], check=True)

        with open(output_path) as f:
            cs_results = {r["id"]: r for r in json.load(f)}

    node_deltas = defaultdict(float)
    global_deltas = [0.0] * GLOBAL_DIM
    row_count_mismatches = []
    order_identical = 0
    order_mismatches = []

    for case in cases:
        cid = case["id"]
        state = case["state"]
        cs = cs_results[cid]

        data = json_to_pyg_graph(state)
        py_node = data.x.tolist()
        py_global = data.u.squeeze(0).tolist()

        cs_node = cs["node_matrix"]
        cs_global = cs["global_vector"]

        if len(py_node) != len(cs_node):
            row_count_mismatches.append((cid, len(py_node), len(cs_node)))
        else:
            py_sorted = sorted(py_node, key=lambda row: tuple(row))
            cs_sorted = sorted(cs_node, key=lambda row: tuple(row))
            for row_py, row_cs in zip(py_sorted, cs_sorted):
                for j in range(NODE_DIM):
                    d = abs(row_py[j] - row_cs[j])
                    if d > node_deltas[j]:
                        node_deltas[j] = d

            # Row order, unsorted. Matters only for a model that is not
            # permutation-invariant, such as the flat-MLP ablation.
            # Compared at float32: both sides produce float32, and the JSON
            # round-trip perturbs the float64 view of the same value.
            order_ok = all(
                np.float32(row_py[j]) == np.float32(row_cs[j])
                for row_py, row_cs in zip(py_node, cs_node)
                for j in range(NODE_DIM))
            if order_ok:
                order_identical += 1
            else:
                order_mismatches.append(cid)

        for j in range(GLOBAL_DIM):
            d = abs(py_global[j] - cs_global[j])
            if d > global_deltas[j]:
                global_deltas[j] = d

    print()
    print("=" * 78)
    print("NODE ROW ORDER (unsorted)")
    print("=" * 78)
    n_cmp = len(cases) - len(row_count_mismatches)
    print(f"  identical row order: {order_identical}/{n_cmp}")
    if order_mismatches:
        print(f"  DIFFERENT row order: {len(order_mismatches)} state(s), "
              f"e.g. {', '.join(order_mismatches[:3])}")
        print(f"  The DeepSets model mean-pools and is unaffected. Any model that is")
        print(f"  NOT permutation-invariant -- the flat-MLP ablation, REPRODUCE.md")
        print(f"  section 9 -- would be trained on one order and served another.")
    else:
        print(f"  Both implementations emit the nine location blocks in the same order")
        print(f"  and the same order within each block, so a non-permutation-invariant")
        print(f"  model sees the same vector in training and in a game.")

    print()
    print("=" * 78)
    print("NODE FEATURE MAX ABS DELTA PER COLUMN")
    print("=" * 78)
    node_fail_cols = []
    for j in range(NODE_DIM):
        d = node_deltas.get(j, 0.0)
        flag = "" if d <= DELTA_THRESHOLD else "  <-- FAIL"
        if flag:
            node_fail_cols.append(j)
        print(f"  [{j:2d}] {node_feature_label(j):16s} max_abs_delta={d:.3e}{flag}")

    print()
    print("=" * 78)
    print("GLOBAL FEATURE MAX ABS DELTA PER COLUMN")
    print("=" * 78)
    global_fail_cols = []
    for j in range(GLOBAL_DIM):
        d = global_deltas[j]
        flag = "" if d <= DELTA_THRESHOLD else "  <-- FAIL"
        if flag:
            global_fail_cols.append(j)
        print(f"  [{j:2d}] {GLOBAL_FEATURE_NAMES[j]:24s} max_abs_delta={d:.3e}{flag}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  states compared: {len(cases)}")
    print(f"  row count mismatches: {len(row_count_mismatches)}")
    for cid, py_n, cs_n in row_count_mismatches:
        print(f"    {cid}: python={py_n} rows, csharp={cs_n} rows")
    print(f"  node columns exceeding {DELTA_THRESHOLD}: {len(node_fail_cols)} {node_fail_cols if node_fail_cols else ''}")
    print(f"  global columns exceeding {DELTA_THRESHOLD}: {len(global_fail_cols)} {global_fail_cols if global_fail_cols else ''}")

    ok = not row_count_mismatches and not node_fail_cols and not global_fail_cols
    print()
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
