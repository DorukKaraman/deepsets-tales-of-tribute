"""
Game-clustered paired comparison of checkpoints, from evaluate_checkpoints.py's
--per-state-out file.

WHY CLUSTERING IS NOT OPTIONAL HERE. A validation set of 20,138 states sounds
like 20,138 observations. It is not: those states come from 76 games, about 265
consecutive positions each, sharing a board and an outcome label. A test that
treats them as independent understates the standard error by roughly the square
root of the cluster size and will call almost anything significant. On the
flat-MLP ablation the same accuracy difference came out at p = 2e-14 by
sample-level McNemar and p = 0.086 clustered by game -- twelve orders of
magnitude apart, and only the second one is defensible.

So the unit of analysis is the GAME. For each pair of checkpoints this computes
one difference per game (mean loss, and mean accuracy, of A minus B over that
game's states), then a paired t-test over games. The samples are paired at the
state level too, because evaluate_checkpoints.py scores every checkpoint on
byte-identical samples in one pass -- which is what makes the per-game
differences meaningful rather than two independent estimates subtracted.

BONFERRONI. Comparing k checkpoints makes k(k-1)/2 comparisons, and each metric
is tested separately, so the adjustment is over all of them. It is applied and
shown rather than left to the reader.

    python tools/evaluate_checkpoints.py A.pth B.pth C.pth \\
        --data-dir "$SPLIT/val" --per-state-out /path/scores.csv.gz
    python tools/clustered_significance.py /path/scores.csv.gz

Streaming and memory-light: it accumulates per-game sums, not per-state arrays.

Read-only. Requires numpy; uses scipy for the t distribution when available and
falls back to a normal approximation (noted in the output) when it is not.
"""
import argparse
import gzip
import math
import os
import sys
from collections import defaultdict

import numpy as np

# Matches evaluate_checkpoints.EPS so losses recomputed here equal the ones it
# reported. A different clip would silently shift every loss.
EPS = 1e-7


def read_per_state(path):
    """Returns (names, per_game) where per_game maps game_id -> dict with
    'n', and per model 'loss_sum' / 'correct'. One pass, no per-state storage."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        if len(header) < 5 or header[:4] != ["game_id", "target", "prestige_clock", "bucket"]:
            sys.exit(f"ERROR: {path} does not look like an evaluate_checkpoints.py "
                     f"--per-state-out file.\n"
                     f"       Expected a header starting "
                     f"game_id,target,prestige_clock,bucket,...\n"
                     f"       Got: {','.join(header[:6])}")
        names = [h[2:] if h.startswith("p_") else h for h in header[4:]]
        if not names:
            sys.exit(f"ERROR: {path} has no checkpoint probability columns.")

        per_game = defaultdict(lambda: {"n": 0,
                                        "loss": np.zeros(len(names)),
                                        "correct": np.zeros(len(names))})
        n_rows = 0
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != len(header):
                continue
            gid = parts[0]
            target = float(parts[1])
            probs = np.array([float(x) for x in parts[4:]], dtype=np.float64)
            p = np.clip(probs, EPS, 1 - EPS)
            loss = -(target * np.log(p) + (1 - target) * np.log(1 - p))
            correct = ((probs >= 0.5).astype(np.float64) == target).astype(np.float64)
            g = per_game[gid]
            g["n"] += 1
            g["loss"] += loss
            g["correct"] += correct
            n_rows += 1
    return names, per_game, n_rows


def paired_t(diffs):
    """Two-sided paired t-test over the per-game differences."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan"), n
    se = float(d.std(ddof=1) / math.sqrt(n))
    if se == 0.0:
        return mean, 0.0, 0.0, 1.0, n
    t = mean / se
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    return mean, se, t, p, n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("per_state_csv",
                        help="evaluate_checkpoints.py --per-state-out file (.csv.gz)")
    parser.add_argument("--baseline", default=None,
                        help="Compare every other checkpoint against this one only, instead "
                             "of all pairs. Reduces the Bonferroni factor from k(k-1)/2 to "
                             "k-1 when there is a designated reference model.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.isfile(args.per_state_csv):
        sys.exit(f"ERROR: no such file: {args.per_state_csv}")

    try:
        from scipy import stats  # noqa: F401
        dist = "t"
    except ImportError:
        dist = "normal approximation (scipy not installed)"

    names, per_game, n_rows = read_per_state(args.per_state_csv)
    games = sorted(per_game)
    if len(games) < 2:
        sys.exit(f"ERROR: only {len(games)} game(s) in {args.per_state_csv}; a clustered "
                 f"test needs many. Was game_id populated when the file was written?")

    sizes = np.array([per_game[g]["n"] for g in games], dtype=np.float64)
    loss = np.array([per_game[g]["loss"] / per_game[g]["n"] for g in games])      # [G, K]
    acc = np.array([per_game[g]["correct"] / per_game[g]["n"] for g in games])    # [G, K]

    print("=" * 78)
    print("GAME-CLUSTERED PAIRED COMPARISON")
    print("=" * 78)
    print(f"  file            : {args.per_state_csv}")
    print(f"  states          : {n_rows:,}")
    print(f"  games (clusters): {len(games)}")
    print(f"  states per game : min {int(sizes.min())}, median "
          f"{int(np.median(sizes))}, max {int(sizes.max())}")
    print(f"  null distribution: {dist}")
    print()
    print(f"  {'checkpoint':<32}{'loss':>10}{'accuracy':>11}")
    for k, name in enumerate(names):
        # Weight per game by that game's size to recover the pooled figure
        # evaluate_checkpoints.py prints, rather than the unweighted mean of
        # per-game means used for the test.
        pooled_loss = float((loss[:, k] * sizes).sum() / sizes.sum())
        pooled_acc = float((acc[:, k] * sizes).sum() / sizes.sum())
        print(f"  {name:<32}{pooled_loss:>10.4f}{100 * pooled_acc:>10.2f}%")
    print()

    if args.baseline is not None:
        if args.baseline not in names:
            sys.exit(f"ERROR: --baseline {args.baseline!r} is not one of the checkpoints "
                     f"in the file: {names}")
        b = names.index(args.baseline)
        pairs = [(b, j) for j in range(len(names)) if j != b]
    else:
        pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    # Each pair is tested on two metrics, so both count toward the correction.
    n_tests = len(pairs) * 2
    print(f"  {n_tests} test(s): {len(pairs)} pair(s) x 2 metrics. "
          f"Bonferroni factor {n_tests}, family alpha {args.alpha}.")
    print()

    rows = []
    for i, j in pairs:
        for metric, mat, unit in (("loss", loss, ""), ("accuracy", acc, " pts")):
            d = mat[:, i] - mat[:, j]
            mean, se, t, p, n = paired_t(d)
            scale = 100.0 if metric == "accuracy" else 1.0
            better_i = int((d > 0).sum()) if metric == "accuracy" else int((d < 0).sum())
            better_j = int((d < 0).sum()) if metric == "accuracy" else int((d > 0).sum())
            rows.append((names[i], names[j], metric, mean * scale, se * scale, t, p,
                         min(1.0, p * n_tests), better_i, better_j, unit))

    def fmt_p(p):
        """Never print a p-value as 0.0000. A t of -13.5 gives something like
        1e-38, and rounding that to '0.0000' both loses the magnitude and reads
        as a computation that failed."""
        if p < 1e-9:
            return "<1e-9"
        if p < 1e-4:
            return f"{p:.1e}"
        return f"{p:.4f}"

    w = max(len(r[0]) for r in rows) + max(len(r[1]) for r in rows) + 5
    print(f"  {'comparison':<{w}}{'metric':<10}{'mean diff':>12}{'SE':>9}"
          f"{'t':>8}{'p':>10}{'p_adj':>10}{'better':>12}")
    for a, b, metric, mean, se, t, p, padj, ba, bb, unit in rows:
        star = " *" if padj < args.alpha else ""
        print(f"  {a + ' - ' + b:<{w}}{metric:<10}{mean:>+12.4f}{se:>9.4f}"
              f"{t:>8.2f}{fmt_p(p):>10}{fmt_p(padj):>10}{f'{ba}/{ba + bb}':>12}{star}")

    print()
    print("  mean diff is (first - second), averaged over per-game means; for loss, "
          "negative")
    print("  favours the first model. 'better' counts games where the first model wins, "
          "out of")
    print(f"  games where they differ. * marks p_adj < {args.alpha}.")
    print()
    print("  A difference that is large in the aggregate table and not significant here "
          "is not")
    print("  a contradiction: it means the per-game variation is wide enough that this "
          "many")
    print("  games cannot resolve it. Report the clustered p-value, not the sample-level "
          "one.")


if __name__ == "__main__":
    main()
