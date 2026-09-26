"""
Does node ORDER matter to a value network, and does the agent's search change it?

WHY THIS EXISTS. The flat-MLP ablation (REPRODUCE.md section 9) lost by 1.9
accuracy points offline and by 31 win-rate points in games. That gap needed
explaining, and the explanation is here: the agent searches over DETERMINISED
states, in which the hidden piles are reshuffled, and a flat MLP reads its
input slot by slot. DeepSets is permutation-invariant and cannot notice; a flat
model can, and does.

Three analyses, all on real logged states:

  sensitivity  How far does reshuffling the hidden piles move each model's
               output? Reported against two scales: the spread across unrelated
               states, and -- the one that matters for play -- the change
               between two states one move apart, which is what a search has to
               resolve.

  accuracy     Offline accuracy on the logged order, on a uniform reshuffle,
               and on a reshuffle that preserves duplicate clustering. The
               first tells you how optimistic the reported offline figures are;
               the third tests whether the clustering regularity in the logged
               data is what the model was using.

  leak         Does the logged pile order reflect the TRUE upcoming draw order
               -- information no player has? If it did, the flat models'
               offline numbers would be inflated by a leak rather than by
               distribution fit, and the reshuffle penalty would be the
               removal of that leak.

Read-only. Usage:

    python tools/analyze_order_sensitivity.py all --data-dir "$SPLIT/val" \\
        --onnx deepsets=.../DeepSetsValueNetwork_seed_00.onnx \\
        --onnx matched=.../FlatValueNetwork_matched_seed_00.onnx \\
        --onnx wide=.../FlatValueNetwork_wide_seed_00.onnx

`leak` needs no models. Every analysis is seeded, so reruns agree.
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))

try:
    import numpy as np  # noqa: E402
    import onnxruntime as ort  # noqa: E402
    from StateParser import json_to_pyg_graph  # noqa: E402
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(f"ERROR: could not import dependencies ({e}).\n"
             f"       Activate the venv built by scripts/setup_python_env.sh first.")

# MY_DRAW and ENEMY_UNSEEN: the two location blocks a determiniser reshuffles.
# The others are public information and are not re-randomised.
HIDDEN_LOCS = (4, 8)
LOC_BASE = 90        # location one-hot occupies node-feature indices 90..98


def loc_of(row):
    nz = np.nonzero(row[LOC_BASE:LOC_BASE + 9])[0]
    return int(nz[0]) if len(nz) == 1 else -1


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def infer(sess, x, u):
    return float(np.asarray(sess.run(
        None, {"node_features": x, "global_features": u})[0]).reshape(-1)[0])


def reorder(x, locs, rng, mode):
    """Re-order rows WITHIN the hidden piles only.

    uniform  -- a uniformly random permutation, which is what a determiniser
                produces.
    grouped  -- a random permutation that keeps identical cards adjacent, so it
                preserves the duplicate clustering the logged order has while
                changing everything else about the arrangement.
    """
    out = x.copy()
    for b in HIDDEN_LOCS:
        idx = np.nonzero(locs == b)[0]
        if len(idx) < 2:
            continue
        rows = [tuple(x[i]) for i in idx]
        if mode == "grouped":
            uniq = list(dict.fromkeys(rows))
            rng.shuffle(uniq)
            new = [r for u_ in uniq for r in rows if r == u_]
        else:
            new = [rows[j] for j in rng.permutation(len(rows))]
        for k, i in enumerate(idx):
            out[i] = np.array(new[k], dtype=np.float32)
    return out


def iter_records(data_dir, stride, limit):
    shards = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl.gz"), recursive=True))
    if not shards:
        sys.exit(f"ERROR: no *.jsonl.gz shards found under {data_dir}")
    n = 0
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for i, line in enumerate(f):
                if stride > 1 and i % stride:
                    continue
                yield shard, json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def load_states(data_dir, stride, limit):
    out = []
    for _, rec in iter_records(data_dir, stride, limit):
        g = json_to_pyg_graph(rec["data"]["state"])
        out.append((g.x.numpy().astype(np.float32),
                    g.u.numpy().astype(np.float32),
                    float(rec["outcome"])))
    return out


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------

def run_sensitivity(models, data_dir, n_states, k, seed):
    states = load_states(data_dir, 53, n_states)
    # Consecutive states from a few games: the one-move scale.
    games = defaultdict(list)
    for _, rec in iter_records(data_dir, 1, 20000):
        g = games[rec["game_id"]]
        if len(g) < 40:
            g.append(json_to_pyg_graph(rec["data"]["state"]))
        if len(games) >= 6 and all(len(v) >= 20 for v in games.values()):
            break
    games = {kk: v for kk, v in list(games.items())[:6] if len(v) >= 20}

    print(f"  {len(states)} states x {k} reshuffles; one-move scale from "
          f"{len(games)} games\n")
    hidden_frac = float(np.mean([
        np.isin([loc_of(r) for r in x], HIDDEN_LOCS).mean() for x, _, _ in states]))
    print(f"  hidden piles are {100*hidden_frac:.0f}% of nodes in the median state\n")

    for name, path in models.items():
        sess = session(path)
        base_logits, sds, flips = [], [], 0
        for x, u, _ in states:
            locs = np.array([loc_of(r) for r in x])
            base = infer(sess, x, u)
            base_logits.append(base)
            rng = np.random.default_rng(seed)
            v = np.array([infer(sess, reorder(x, locs, rng, "uniform"), u)
                          for _ in range(k)])
            sds.append(v.std())
            if ((v >= 0) != (base >= 0)).any():
                flips += 1
        between = float(np.std(base_logits))
        sds = np.array(sds)

        step_gaps, step_noise = [], []
        for graphs in games.values():
            seq = []
            for g in graphs:
                x = g.x.numpy().astype(np.float32)
                u = g.u.numpy().astype(np.float32)
                seq.append(infer(sess, x, u))
                locs = np.array([loc_of(r) for r in x])
                rng = np.random.default_rng(seed + 1)
                step_noise.append(np.std([infer(sess, reorder(x, locs, rng, "uniform"), u)
                                          for _ in range(k)]))
            step_gaps.extend(np.abs(np.diff(seq)).tolist())
        sg = np.array(step_gaps)
        ns = np.array(step_noise)
        m = min(len(sg), len(ns))

        print(f"  --- {name} ---")
        print(f"    between-state sd of the logit (SIGNAL)  : {between:8.4f}")
        print(f"    reshuffle sd on one state (NOISE)       : {sds.mean():8.4f}"
              f"  (median {np.median(sds):.4f}, max {sds.max():.4f})")
        print(f"    noise / between-state signal            : {sds.mean()/between:8.1%}")
        print(f"    |logit change| one move apart           : {sg.mean():8.4f}"
              f"  (median {np.median(sg):.4f})")
        print(f"    noise / ONE-MOVE signal                 : "
              f"{ns.mean()/sg.mean():8.2f}x")
        print(f"    reshuffle sd exceeds the one-move gap in: "
              f"{100*np.mean(ns[:m] > sg[:m]):7.0f}% of steps")
        print(f"    states where a reshuffle flips a winner : "
              f"{flips}/{len(states)}  ({100*flips/len(states):.1f}%)")
        print()


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------

def run_accuracy(models, data_dir, n_states, seed):
    states = load_states(data_dir, 7, n_states)
    print(f"  {len(states)} states\n")
    print(f"  {'model':<12}{'logged':>10}{'clustered':>12}{'uniform':>10}"
          f"{'logged-uniform':>17}{'recovered':>12}")
    eps = 1e-7
    for name, path in models.items():
        sess = session(path)
        acc = {"logged": [], "grouped": [], "uniform": []}
        loss = {"logged": [], "uniform": []}
        rng = np.random.default_rng(seed)
        for x, u, t in states:
            locs = np.array([loc_of(r) for r in x])
            variants = {"logged": x,
                        "grouped": reorder(x, locs, rng, "grouped"),
                        "uniform": reorder(x, locs, rng, "uniform")}
            for kk, xx in variants.items():
                lg = infer(sess, xx, u)
                acc[kk].append((lg >= 0) == (t == 1.0))
                if kk in loss:
                    p = min(max(1 / (1 + np.exp(-lg)), eps), 1 - eps)
                    loss[kk].append(-(t * np.log(p) + (1 - t) * np.log(1 - p)))
        lg_, gr, un = [100 * np.mean(acc[kk]) for kk in ("logged", "grouped", "uniform")]
        rec = (gr - un) / (lg_ - un) * 100 if lg_ != un else float("nan")
        print(f"  {name:<12}{lg_:>9.2f}%{gr:>11.2f}%{un:>9.2f}%"
              f"{lg_-un:>+16.2f}{rec:>11.0f}%")
    print()
    print("  'logged-uniform' is how optimistic the reported offline figures are:")
    print("  they were measured on the logged order, and the agent never sees it.")
    print("  'recovered' is how much of that gap a duplicate-clustered reshuffle")
    print("  gets back -- near 0 means the model was not using the clustering.")


# --------------------------------------------------------------------------
# leak
# --------------------------------------------------------------------------

def run_leak(data_dir, limit):
    print("  A pile logged in true draw order would lose cards from its FRONT.\n")
    for path, label in (("CurrentPlayer.DrawPile", "CurrentPlayer.DrawPile"),
                        ("EnemyPlayer.HandAndDraw", "EnemyPlayer.HandAndDraw"),
                        ("CurrentPlayer.CooldownPile", "CurrentPlayer.CooldownPile (control)")):
        seqs = defaultdict(list)
        for shard, rec in iter_records(data_dir, 1, limit):
            obj = rec["data"]["state"]
            for part in path.split("."):
                obj = obj.get(part, {}) if isinstance(obj, dict) else []
            cards = obj if isinstance(obj, list) else []
            seqs[(shard, rec["game_id"], rec.get("player", 0))].append(
                [c.get("CommonId") for c in cards])

        front = tot = k1 = k1_hit = 0
        k1_chance = 0.0
        for series in seqs.values():
            for a, b in zip(series, series[1:]):
                if not (0 < len(b) < len(a)):
                    continue
                k = len(a) - len(b)
                removed = Counter(a) - Counter(b)
                if sum(removed.values()) != k:
                    continue          # the pile was added to as well
                tot += 1
                if Counter(a[:k]) == removed:
                    front += 1
                if k == 1:
                    k1 += 1
                    card = next(iter(removed))
                    if a[0] == card:
                        k1_hit += 1
                    k1_chance += Counter(a)[card] / len(a)
        if not tot:
            print(f"  {label}: no clean shrink steps")
            continue
        print(f"  {label}")
        print(f"    clean shrink steps             : {tot}")
        print(f"    removed cards were the front k : {100*front/tot:.2f}%")
        if k1:
            print(f"    k=1 steps                      : {k1}")
            print(f"    front card was the one removed : {100*k1_hit/k1:.2f}%"
                  f"   (chance, given duplicates: {100*k1_chance/k1:.2f}%)")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis", choices=["sensitivity", "accuracy", "leak", "all"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--onnx", action="append", default=[], metavar="NAME=PATH",
                   help="Repeatable. Not needed for 'leak'.")
    p.add_argument("--states", type=int, default=150,
                   help="States for 'sensitivity' (default 150)")
    p.add_argument("--accuracy-states", type=int, default=3000)
    p.add_argument("--leak-records", type=int, default=60000)
    p.add_argument("--reshuffles", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    models = {}
    for spec in args.onnx:
        if "=" not in spec:
            sys.exit(f"ERROR: --onnx wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.isfile(path):
            sys.exit(f"ERROR: no such ONNX file: {path}")
        models[name] = path
    if args.analysis in ("sensitivity", "accuracy", "all") and not models:
        sys.exit(f"ERROR: '{args.analysis}' needs at least one --onnx NAME=PATH")

    if args.analysis in ("sensitivity", "all"):
        print("=" * 78); print("SENSITIVITY TO RESHUFFLING THE HIDDEN PILES"); print("=" * 78)
        run_sensitivity(models, args.data_dir, args.states, args.reshuffles, args.seed)
    if args.analysis in ("accuracy", "all"):
        print("=" * 78); print("OFFLINE ACCURACY: LOGGED vs RESHUFFLED ORDER"); print("=" * 78)
        run_accuracy(models, args.data_dir, args.accuracy_states, args.seed)
        print()
    if args.analysis in ("leak", "all"):
        print("=" * 78); print("DOES THE LOGGED ORDER LEAK THE TRUE DRAW ORDER?"); print("=" * 78)
        run_leak(args.data_dir, args.leak_records)


if __name__ == "__main__":
    main()
