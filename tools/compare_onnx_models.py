"""
Compare two ONNX models on real game states: do they disagree, and how fast is
each?

WHY THIS EXISTS. The five per-seed models each carry 14,000-19,000 subnormal
weights; the shipped model carries none. The shipped model was trained on Apple
Silicon, which flushes denormals to zero; the seeds were trained on x86, which
does not. Subnormal arithmetic falls off onnxruntime's fast path, and the cost
is not subtle: seed 0 measured 1,016 us per inference against the shipped
model's 26.8 us, a 38x slowdown on graphs that are otherwise identical. In a
benchmark where the six models were supposed to differ only in training seed,
that silently gave the seeds ~1,400 evaluations per turn against the shipped
model's ~6,400 -- so they were not being compared at equal search at all.

The fix is to zero weights below 1e-30 before export. A 10-input spot check
showed no output change, but that is far too thin to justify re-exporting five
models on: these weights are near-dead precisely because almost nothing
activates them, so most random inputs will never touch them and a clean spot
check is close to uninformative. This script runs the check on thousands of real
states instead, and measures the timing gap directly rather than inferring it
from evaluations per turn.

Typical use -- confirm a de-subnormalised re-export changes nothing:

    python tools/compare_onnx_models.py \\
        --onnx-a $HPCWORK/tot_models/seed_00/DeepSetsValueNetwork_seed_00.onnx \\
        --onnx-b $HPCWORK/tot_models/seed_00/DeepSetsValueNetwork_seed_00_fixed.onnx \\
        --data-dir $HPCWORK/heldout

Exits nonzero if any predicted winner changes, so it can gate a re-export
script.

WHAT "PREDICTED WINNER" MEANS HERE, AND WHY IT IS THE WEAKER TEST.
The graph's output is a raw logit -- the tensor is named "win_probability", but
ValueNetworkEvaluator.EvaluateBoardState in Bots/src/DeepSetsCore.cs applies the
sigmoid itself before returning. So this script thresholds at sigmoid(logit) >=
0.5, equivalently logit >= 0, which is the convention train_local.py,
evaluate_checkpoints.py and diagnose_value_net.py all use for accuracy.

But the bot itself never thresholds anything. It feeds the win probability
straight into the search as a continuous value and ranks moves with
`score > bestScore` in BestChild()/BanditChild(). The only 0.5 comparisons
anywhere in the agents are an unrelated UCB prior and some policy-logit bumps.

That means a zero prediction-flip count does NOT establish that two models play
identically: ANY nonzero difference can reorder two close moves and change the
game. The count that actually bounds behavioural equivalence is "states with any
nonzero difference", reported alongside. Read that one first; treat the
prediction-flip count as the coarse gate it is.

RUN THE TIMING HALF ON THE TARGET ARCHITECTURE. The subnormal penalty is not
universal -- it is what NATIVE x86 does. A host that flushes denormals to zero
pays nothing for them, so this script will report a ~1.0x timing ratio there
even between a clean model and one stuffed with subnormals, and that number
means "wrong machine", not "no problem".

Measured directly while building this: on an Apple M1, with the pinned x86_64
Python running under Rosetta, a float32 matmul against an all-subnormal operand
was 0.87x the time of a normal one -- no penalty at all -- and this script
reported 1.00x between the shipped model and a copy with 22% of its weights
forced subnormal. The same property is why the shipped model has no subnormals
in the first place: it was trained on that machine, and they never survived
training. The seeds were trained natively on x86, where they did.

So the agreement half of this script is architecture-independent and can be run
anywhere; the timing half only means something on the cluster. The script warns
when it detects this situation rather than letting a 1.0x ratio be read as
reassurance.

Requirements: onnxruntime, numpy, torch, torch_geometric -- all already pinned
in scripts/setup_python_env.sh. No new dependencies.
"""
import argparse
import os
import random
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
sys.path.insert(0, TRAINING_DIR)

import onnxruntime as ort  # noqa: E402

try:
    # Features come from the training code path, unmodified. Reimplementing
    # feature extraction here would mean this script could pass while the real
    # one disagrees -- the exact failure mode tools/verify_parity.py exists to
    # catch between the C# and Python extractors.
    from stream_dataset import SakkirinaStreamDataset  # noqa: E402
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(f"ERROR: could not import training/stream_dataset.py ({e}).\n"
             f"       It needs torch and torch_geometric. Activate the venv built by "
             f"scripts/setup_python_env.sh first.")

# The dataset's own default is 100,000 parsed graphs, sized for training, where
# mixing across games matters and 24 GB is allocated for it. That buffer has to
# FILL before it yields anything, so using it here would cost minutes and many
# GB before the first comparison. A small buffer is enough: shard order is
# already shuffled, and a few thousand records spans several hundred games
# because each game contributes only a dozen or so.
DEFAULT_SHUFFLE_BUFFER = 1000

# Subnormal float32: nonzero and below FLT_MIN. These are what fall off
# onnxruntime's fast path.
FLT_MIN = float(np.finfo(np.float32).tiny)


def make_session(path):
    """Single-threaded, matching tools/ParityCheck, export_to_onnx.py's
    verify_export and the bot's own ValueNetworkEvaluator. Threading would buy
    nothing on one small graph at a time, and here it would also add scheduler
    noise to the very timings this script exists to measure."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: no such ONNX file: {path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def count_subnormals(path):
    """Subnormal and total weight counts, straight from the initializers.
    Reported because it is what ties the timing gap to a cause rather than
    leaving it as an unexplained difference between two files."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return None
    try:
        model = onnx.load(path)
    except Exception:
        return None
    subnormal = 0
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32:
            continue
        a = np.abs(arr)
        total += a.size
        subnormal += int(np.count_nonzero((a > 0) & (a < FLT_MIN)))
    return subnormal, total


def run_one(sess, x_np, u_np):
    """Returns (raw_logit, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = sess.run(None, {"node_features": x_np, "global_features": u_np})
    dt = time.perf_counter() - t0
    return float(np.asarray(out[0]).reshape(-1)[0]), dt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-a", required=True, help="First model")
    parser.add_argument("--onnx-b", required=True, help="Second model")
    parser.add_argument("--data-dir", required=True,
                        help="Directory of *.jsonl.gz shards, searched recursively")
    parser.add_argument("--limit", type=int, default=5000, help="States to compare (default: 5000)")
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER,
                        help=f"Shuffle-buffer depth (default: {DEFAULT_SHUFFLE_BUFFER}; the "
                             f"dataset's own 100,000 default is sized for training and would "
                             f"cost GBs here)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds shard order and the shuffle buffer so the sampled states "
                             "are reproducible run to run (default: 0)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Untimed inferences per model before measuring (default: 20)")
    args = parser.parse_args()

    # stream_dataset draws from the `random` module. Seeding it here makes the
    # sample reproducible, which matters because this script is meant to gate a
    # re-export: a flaky sample would make the gate flaky. Safe to touch the
    # global RNG from a CLI entry point; nothing else in this process uses it.
    random.seed(args.seed)

    print("=" * 78)
    print("ONNX MODEL COMPARISON")
    print("=" * 78)
    print(f"  A         : {args.onnx_a}")
    print(f"  B         : {args.onnx_b}")
    print(f"  data      : {args.data_dir}")
    print(f"  limit     : {args.limit} states   (seed {args.seed}, shuffle buffer "
          f"{args.shuffle_buffer})")

    subnormal_counts = {}
    for label, path in (("A", args.onnx_a), ("B", args.onnx_b)):
        counts = count_subnormals(path)
        if counts is None:
            continue
        n_sub, total = counts
        subnormal_counts[label] = n_sub
        pct = (100.0 * n_sub / total) if total else 0.0
        print(f"  {label} weights : {total:,} float32, of which {n_sub:,} subnormal ({pct:.3f}%)")
    print()

    sess_a = make_session(args.onnx_a)
    sess_b = make_session(args.onnx_b)

    try:
        dataset = SakkirinaStreamDataset(args.data_dir, shuffle_buffer_size=args.shuffle_buffer)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")

    n = 0
    warmed = False
    abs_diffs = []
    times_a = []
    times_b = []
    nonzero_diff = 0
    flipped = 0
    flip_examples = []

    for graph in dataset:
        if n >= args.limit:
            break
        x_np = graph.x.numpy().astype(np.float32)
        u_np = graph.u.numpy().astype(np.float32)

        if not warmed:
            # First inferences pay one-off allocation and page-in costs. Timing
            # them would flatter whichever model happens to run second.
            for _ in range(max(0, args.warmup)):
                run_one(sess_a, x_np, u_np)
                run_one(sess_b, x_np, u_np)
            warmed = True

        # Alternate which model goes first so any ordering effect (cache
        # residency of the input arrays, in particular) lands on both equally.
        if n % 2 == 0:
            logit_a, dt_a = run_one(sess_a, x_np, u_np)
            logit_b, dt_b = run_one(sess_b, x_np, u_np)
        else:
            logit_b, dt_b = run_one(sess_b, x_np, u_np)
            logit_a, dt_a = run_one(sess_a, x_np, u_np)

        times_a.append(dt_a)
        times_b.append(dt_b)

        d = abs(logit_a - logit_b)
        abs_diffs.append(d)
        if d != 0.0:
            nonzero_diff += 1
        # sigmoid(logit) >= 0.5  <=>  logit >= 0. See the module docstring for
        # why this is the coarse test and not the decisive one.
        if (logit_a >= 0.0) != (logit_b >= 0.0):
            flipped += 1
            if len(flip_examples) < 5:
                flip_examples.append((n, logit_a, logit_b))

        n += 1
        if n % 1000 == 0:
            print(f"  ... {n} states compared")

    if n == 0:
        sys.exit("ERROR: no states were read -- is --data-dir the right directory?")

    abs_diffs = np.array(abs_diffs, dtype=np.float64)
    ta = np.array(times_a, dtype=np.float64) * 1e6
    tb = np.array(times_b, dtype=np.float64) * 1e6

    print()
    print("=" * 78)
    print("AGREEMENT")
    print("=" * 78)
    print(f"  states compared              : {n}")
    print(f"  max  |A - B| (raw logit)     : {abs_diffs.max():.6e}")
    print(f"  mean |A - B| (raw logit)     : {abs_diffs.mean():.6e}")
    print(f"  states with ANY difference   : {nonzero_diff}  ({100.0 * nonzero_diff / n:.2f}%)")
    print(f"  states where the winner flips: {flipped}  ({100.0 * flipped / n:.2f}%)")
    print(f"      threshold: sigmoid(logit) >= 0.5, i.e. logit >= 0 -- the convention")
    print(f"      train_local.py and evaluate_checkpoints.py use. The BOT does not")
    print(f"      threshold at all; it ranks moves on the continuous value, so a flip")
    print(f"      count of 0 does not prove identical play. See 'states with ANY")
    print(f"      difference' above for that.")
    for idx, la, lb in flip_examples:
        print(f"        state {idx}: A={la:+.6f}  B={lb:+.6f}")

    print()
    print("=" * 78)
    print("INFERENCE TIME (single-threaded, per call)")
    print("=" * 78)
    print(f"  {'model':<8}{'mean us':>12}{'median us':>12}{'min us':>12}{'max us':>12}")
    print(f"  {'A':<8}{ta.mean():>12.1f}{np.median(ta):>12.1f}{ta.min():>12.1f}{ta.max():>12.1f}")
    print(f"  {'B':<8}{tb.mean():>12.1f}{np.median(tb):>12.1f}{tb.min():>12.1f}{tb.max():>12.1f}")
    ratio = tb.mean() / ta.mean() if ta.mean() > 0 else float("nan")
    if ta.mean() > 0:
        print(f"  B/A mean ratio : {ratio:.2f}x")

    # A host that flushes denormals to zero pays nothing for subnormal weights,
    # so a ~1x ratio there says nothing about how the same pair behaves on the
    # machine the benchmark runs on. Catch that rather than let the number be
    # read as reassurance -- see "RUN THE TIMING HALF ON THE TARGET
    # ARCHITECTURE" in the module docstring.
    if len(subnormal_counts) == 2:
        lo, hi = sorted(subnormal_counts.values())
        if hi >= 1000 and hi >= 10 * max(lo, 1) and 0.8 <= ratio <= 1.25:
            import platform
            print()
            print(f"  *** WARNING: one model carries {hi:,} subnormal weights and the other "
                  f"{lo:,},")
            print(f"      yet the timing ratio is {ratio:.2f}x. On native x86 that gap is large "
                  f"(38x was")
            print(f"      measured for seed 0 against the shipped model). A ratio near 1 means "
                  f"this host")
            print(f"      is not paying the subnormal penalty -- it flushes denormals to zero -- "
                  f"so the")
            print(f"      timing columns above are NOT evidence that the models cost the same on "
                  f"the")
            print(f"      cluster. Re-run the timing check there.")
            print(f"      host: {platform.machine()} / {platform.system()}")

    print()
    if flipped:
        print(f"RESULT: FAIL -- {flipped} of {n} states change the predicted winner.")
        sys.exit(1)
    if nonzero_diff:
        print(f"RESULT: no predicted winner changes, but {nonzero_diff} of {n} states "
              f"differ numerically")
        print(f"        (max {abs_diffs.max():.3e} on the raw logit). The two models are "
              f"NOT bit-identical;")
        print(f"        whether that is acceptable depends on whether a difference that "
              f"small can reorder")
        print(f"        two close moves in the search. It can, in principle.")
    else:
        print(f"RESULT: PASS -- the two models produced bit-identical outputs on all "
              f"{n} states.")


if __name__ == "__main__":
    main()
