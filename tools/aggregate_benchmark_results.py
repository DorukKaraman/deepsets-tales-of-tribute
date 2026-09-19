"""
Reads every per-task result JSON file tools/benchmark_cluster.py writes and
prints, per matchup: games completed/failed, clean win rate (bot_a's
perspective) with a 95% Wilson interval, and the swapped/non-swapped
breakdown.

The swapped/non-swapped breakdown is not decoration -- it is the check for a
seat-swap inversion bug (see benchmark_cluster.py's top-of-file docstring). A
bug there makes the AGGREGATE win rate drift toward 50%, which looks like a
perfectly plausible result on its own. It does NOT make the two per-seat rates
agree with each other; a real, correctly-measured skill difference should
show up in both the non-swapped and swapped rows, at least roughly (some
divergence between them is expected and fine -- first-player advantage is
real, which is the whole reason games get swapped in the first place -- but
both rows moving in the SAME direction, wide of 50%, is what a healthy result
looks like. Both rows sitting close to 50% independently, or one at 90% and
the other at 10%, is what a swap bug looks like).

Read-only. Does not modify or delete any result file.
"""
import argparse
import glob
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from benchmark_cluster import MATCHUPS, GAMES_PER_MATCHUP  # noqa: E402


def wilson_interval(k, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion k/n. Same formula
    tools/benchmark_runner.py uses, kept identical on purpose."""
    if n == 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (center - margin) / denom, (center + margin) / denom


def load_matchup_results(out_dir, matchup_index, bot_a, bot_b):
    matchup_dir = os.path.join(out_dir, f"matchup_{matchup_index:02d}_{bot_a}_vs_{bot_b}")
    results = []
    malformed = 0
    for path in sorted(glob.glob(os.path.join(matchup_dir, "task_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("completed") is not True:
                malformed += 1
                continue
            results.append(data)
        except Exception as e:
            malformed += 1
            print(f"  [WARN] {path}: unreadable/malformed ({e}) -- treated as not completed", file=sys.stderr)
    return results, malformed


def summarize_subset(results):
    """results: list of result dicts, already filtered to one seat-subset (or
    all of them). Returns (wins_a, wins_b, draws, decided, win_rate, ci_low, ci_high)."""
    clean = [r for r in results if r.get("clean")]
    wins_a = sum(1 for r in clean if r["winner"] == "bot_a")
    wins_b = sum(1 for r in clean if r["winner"] == "bot_b")
    draws = sum(1 for r in clean if r["winner"] == "draw")
    decided = wins_a + wins_b
    win_rate, ci_low, ci_high = None, None, None
    if decided > 0:
        win_rate = wins_a / decided
        ci_low, ci_high = wilson_interval(wins_a, decided)
    return wins_a, wins_b, draws, decided, win_rate, ci_low, ci_high


def fmt_rate(win_rate, ci_low, ci_high, n):
    if win_rate is None:
        return f"n/a (n={n}, no clean decided games)"
    return f"{win_rate:.4f}  [95% CI {ci_low:.4f}, {ci_high:.4f}]  (n={n})"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Same --out-dir the benchmark run used")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        sys.exit(f"ERROR: --out-dir does not exist: {out_dir}")

    print("=" * 100)
    print(f"BENCHMARK RESULTS: {out_dir}")
    print("=" * 100)

    grand_completed = 0
    grand_expected = 0

    for i, (bot_a, bot_b) in enumerate(MATCHUPS):
        results, malformed = load_matchup_results(out_dir, i, bot_a, bot_b)
        completed = len(results)
        failed = GAMES_PER_MATCHUP - completed
        grand_completed += completed
        grand_expected += GAMES_PER_MATCHUP

        not_swapped = [r for r in results if not r["swapped"]]
        swapped = [r for r in results if r["swapped"]]

        not_clean = [r for r in results if not r.get("clean")]

        print()
        print(f"[{i}] {bot_a} vs {bot_b}")
        print(f"  games completed        : {completed}/{GAMES_PER_MATCHUP}"
              + (f"  ({malformed} malformed result file(s), not counted as completed)" if malformed else ""))
        print(f"  games failed/pending   : {failed}/{GAMES_PER_MATCHUP}  "
              "(no result file yet -- never ran, still running, or crashed/timed out)")
        if not_clean:
            reasons = {}
            for r in not_clean:
                reasons[r["end_reason"]] = reasons.get(r["end_reason"], 0) + 1
            reasons_str = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
            print(f"  non-clean (excluded from win rate below): {len(not_clean)}  ({reasons_str})")

        wins_a, wins_b, draws, decided, win_rate, ci_low, ci_high = summarize_subset(results)
        print(f"  clean wins {bot_a:<22}: {wins_a}")
        print(f"  clean wins {bot_b:<22}: {wins_b}")
        print(f"  clean draws            : {draws}")
        print(f"  {bot_a} win rate (clean, decided, both seats): {fmt_rate(win_rate, ci_low, ci_high, decided)}")

        _, _, _, ns_decided, ns_rate, ns_lo, ns_hi = summarize_subset(not_swapped)
        _, _, _, sw_decided, sw_rate, sw_lo, sw_hi = summarize_subset(swapped)
        print(f"    not swapped ({bot_a}=P1): {fmt_rate(ns_rate, ns_lo, ns_hi, ns_decided)}")
        print(f"    swapped     ({bot_b}=P1): {fmt_rate(sw_rate, sw_lo, sw_hi, sw_decided)}")

    print()
    print("=" * 100)
    print(f"TOTAL: {grand_completed}/{grand_expected} games completed across {len(MATCHUPS)} matchups.")
    print("=" * 100)


if __name__ == "__main__":
    main()
