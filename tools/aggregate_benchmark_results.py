"""
Reads every per-task result JSON file tools/benchmark_cluster.py writes and
prints, per matchup: games completed/failed, clean win rate (bot_a's
perspective) with a 95% Wilson interval, the swapped/non-swapped breakdown,
the non-clean games split by CATEGORY and attributed to a side, and mean
evaluations per turn per agent.

Takes the same --config the run used, so it knows what it is looking for.

THE SWAPPED/NON-SWAPPED BREAKDOWN is not decoration -- it is the check for a
seat-swap inversion bug (see benchmark_cluster.py's top-of-file docstring). A
bug there makes the AGGREGATE win rate drift toward 50%, which looks like a
perfectly plausible result on its own. It does NOT make the two per-seat rates
agree with each other; a real, correctly-measured skill difference should show
up in both rows, at least roughly (some divergence is expected and fine --
first-player advantage is real, which is the whole reason games get swapped --
but both rows moving in the SAME direction, wide of 50%, is what a healthy
result looks like. Both rows sitting close to 50% independently, or one at 90%
and the other at 10%, is what a swap bug looks like).

DISQUALIFICATIONS AND TIMEOUTS ARE REPORTED SEPARATELY, per side, because they
mean opposite things:

  timeout         The agent ran out of clock. At the short budgets
                  experiments/configs/time_scaling.json uses, a game lost this
                  way says the engine --timeout margin is too tight, NOT that
                  the agent plays worse. If this column is not near zero, the
                  row's win rate is not a measurement of play and should not be
                  reported as one -- raise the margin and re-run the row.
  disqualification  The agent made a move the engine rejected, threw, or failed
                  patron selection. A real defect, and it does not get excused
                  by a tight budget.
  turn_limit      500 turns with no result. Neither agent's fault in
                  particular; excluded from the win rate as it always was.

All three are excluded from the clean win rate, exactly as before -- the
difference is that they are now visible individually instead of pooled into
"other".

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
from benchmark_cluster import (  # noqa: E402
    ConfigError, NON_CLEAN_CATEGORIES, load_config, matchup_dir, resolve_task,
)


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


def load_matchup_results(out_dir, config, matchup):
    """Result files for one matchup. The directory name is derived through
    resolve_task/matchup_dir rather than rebuilt here, so a change to the
    layout cannot silently make this look at the wrong (or an empty) place."""
    probe = resolve_task(config, matchup["task_offset"])
    directory = matchup_dir(out_dir, probe)
    results, malformed = [], 0
    for path in sorted(glob.glob(os.path.join(directory, "task_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("completed") is not True:
                malformed += 1
                continue
            results.append(data)
        except Exception as e:
            malformed += 1
            print(f"  [WARN] {path}: unreadable/malformed ({e}) -- treated as not completed",
                  file=sys.stderr)
    return results, malformed, directory


def category_of(result):
    """Result files written before end_reason_detail existed only carry the
    coarse bucket, which cannot tell a timeout from an illegal move. Those are
    reported as 'unknown' rather than guessed at."""
    category = result.get("category")
    if category:
        return category
    return "clean" if result.get("clean") else "unknown"


def summarize_subset(results):
    """results: list of result dicts, already filtered to one seat-subset (or
    all of them). Returns (wins_a, wins_b, draws, decided, win_rate, ci_low, ci_high)."""
    clean = [r for r in results if category_of(r) == "clean"]
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


def print_category_block(results, bot_a, bot_b):
    """The DQ/timeout/turn-limit split, per side. Nothing is printed for a
    category with no games, except that a non-zero timeout count always gets
    its warning."""
    by_category = {}
    for r in results:
        by_category.setdefault(category_of(r), []).append(r)

    for category in NON_CLEAN_CATEGORIES:
        rows = by_category.get(category, [])
        if not rows:
            continue
        reasons = {}
        offenders = {"bot_a": 0, "bot_b": 0, "unknown": 0}
        for r in rows:
            detail = r.get("end_reason_detail") or r.get("end_reason") or "?"
            reasons[detail] = reasons.get(detail, 0) + 1
            offenders[r.get("offender") or "unknown"] += 1
        reasons_str = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"  {category:<16}: {len(rows):>4}  ({reasons_str})")
        if category in ("timeout", "disqualification"):
            print(f"  {'':<16}   caused by {bot_a}: {offenders['bot_a']}, "
                  f"{bot_b}: {offenders['bot_b']}, undetermined: {offenders['unknown']}")
        if category == "timeout":
            print(f"  {'':<16}   *** timeouts are a statement about the --timeout margin, not about "
                  f"play. If this is not near zero, do not report this matchup's win rate as a "
                  f"measurement of play -- raise the margin and re-run the row.")
        if category == "unknown":
            print(f"  {'':<16}   (result files from a GameRunner without the GAME_END_REASON line "
                  f"cannot separate timeouts from disqualifications -- re-run to classify them)")


def print_evals_block(results):
    per_class = {}
    ambiguous = set()
    for r in results:
        for cls, stats in (r.get("evals_per_turn") or {}).items():
            per_class.setdefault(cls, []).append(stats.get("mean_evals_per_turn", 0.0))
        for cls in r.get("evals_per_turn_ambiguous") or []:
            ambiguous.add(cls)
    if not per_class:
        return
    print("  evaluations per turn (from the per-task bot logs):")
    for cls, means in sorted(per_class.items()):
        flag = "  [AMBIGUOUS: both seats log under this class name]" if cls in ambiguous else ""
        print(f"    {cls:<24} {sum(means) / len(means):>12.1f}  (n={len(means)} games){flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Same --out-dir the benchmark run used")
    parser.add_argument("--config", default="legacy_paper_benchmark",
                        help="Same --config the benchmark run used (default: legacy_paper_benchmark)")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="Only needed if the run overrode the config's seed_base; it does not "
                             "affect the numbers, only the plan echo.")
    args = parser.parse_args()

    try:
        config = load_config(args.config, args.seed_base)
    except (ConfigError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: {e}")

    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        sys.exit(f"ERROR: --out-dir does not exist: {out_dir}")

    print("=" * 100)
    print(f"BENCHMARK RESULTS: {out_dir}")
    print(f"CONFIG           : {config['name']}  ({config['path']})")
    if config["description"]:
        print(f"                   {config['description']}")
    print("=" * 100)

    grand_completed = 0
    grand_expected = 0

    for m in config["matchups"]:
        results, malformed, directory = load_matchup_results(out_dir, config, m)
        completed = len(results)
        failed = m["games"] - completed
        grand_completed += completed
        grand_expected += m["games"]

        not_swapped = [r for r in results if not r["swapped"]]
        swapped = [r for r in results if r["swapped"]]

        print()
        print(f"[{m['index']}] {m['label']}  --  {m['bot_a']} vs {m['bot_b']}")
        env_str = " ".join(f"{k}={v}" for k, v in sorted(m["env"].items())) or "(none)"
        print(f"  env / timeout          : {env_str}  |  engine --timeout {m['timeout']}s")
        print(f"  games completed        : {completed}/{m['games']}"
              + (f"  ({malformed} malformed result file(s), not counted as completed)" if malformed else ""))
        print(f"  games failed/pending   : {failed}/{m['games']}  "
              "(no result file yet -- never ran, still running, or crashed/timed out)")
        if completed == 0:
            print(f"  (nothing found under {directory})")
            continue

        print_category_block(results, m["bot_a"], m["bot_b"])

        wins_a, wins_b, draws, decided, win_rate, ci_low, ci_high = summarize_subset(results)
        print(f"  clean wins {m['bot_a']:<22}: {wins_a}")
        print(f"  clean wins {m['bot_b']:<22}: {wins_b}")
        print(f"  clean draws            : {draws}")
        print(f"  {m['bot_a']} win rate (clean, decided, both seats): "
              f"{fmt_rate(win_rate, ci_low, ci_high, decided)}")

        _, _, _, ns_decided, ns_rate, ns_lo, ns_hi = summarize_subset(not_swapped)
        _, _, _, sw_decided, sw_rate, sw_lo, sw_hi = summarize_subset(swapped)
        print(f"    not swapped ({m['bot_a']}=P1): {fmt_rate(ns_rate, ns_lo, ns_hi, ns_decided)}")
        print(f"    swapped     ({m['bot_b']}=P1): {fmt_rate(sw_rate, sw_lo, sw_hi, sw_decided)}")

        print_evals_block(results)

    print()
    print("=" * 100)
    print(f"TOTAL: {grand_completed}/{grand_expected} games completed across "
          f"{len(config['matchups'])} matchups.")
    print("=" * 100)


if __name__ == "__main__":
    main()
