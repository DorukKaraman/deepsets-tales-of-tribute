"""
Check that experiments/bots/DeepSetsBotExp.cs, with none of its environment
variables set, plays the same moves as the submitted Bots/src/DeepSetsBlendBot.cs.

DeepSetsBotExp is a copy of DeepSetsBlendBot with three environment hooks
(SOT_ALPHA0, SOT_TIME_SCALE, SOT_MODEL_PATH) whose defaults are the
submission's own constants. If that claim is true, an unset environment must
reproduce the submission's play. This runs both agents against the same
opponent from the same seed and compares their move sequences.

READ THIS BEFORE FILING A BUG ON A DIVERGENCE. The search is WALL-CLOCK
budgeted: `while (s.Elapsed < timeForMoveComputation) TreeSearch(...)`. How
many iterations fit into 0.65 seconds depends on machine load, JIT warm-up,
CPU frequency and GC timing, none of which a seed controls. Two runs of the
SAME binary on the same seed can therefore pick different moves, and the two
sequences drift apart permanently once they do, because from that point on the
agents are playing different games.

So a divergence here is evidence of nothing on its own. What this script
reports is the INDEX at which the sequences first differ, and the honest way to
read it is comparatively:

  * Run --self-check to get the baseline: DeepSetsBlendBot against itself, same
    seed, two separate runs. That is the divergence a pure timing difference
    produces, with no code difference at all.
  * If DeepSetsBotExp diverges no earlier than that baseline does, the
    comparison is consistent with the two agents being identical. It does not
    prove it.
  * A divergence at move 0 or 1, reproducible across seeds and well before the
    self-check baseline, is a real signal worth chasing -- most likely a
    mis-defaulted environment variable.

The comparison uses GameRunner's --log-training-data wrapper, which records the
GameState and the chosen move for every Play() call. Note the wrapper only
writes its buffer for cleanly-ended games (PRESTIGE_OVER_40/80, PATRON_FAVOR);
a seed whose game ends in a turn limit or a timeout produces no data and is
reported as such rather than counted as agreement.

Usage:
    python experiments/verify_exp_bot_parity.py                       # default seed
    python experiments/verify_exp_bot_parity.py --seed 12345 --moves 10
    python experiments/verify_exp_bot_parity.py --self-check          # timing baseline
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_BINARY = os.path.join(REPO_ROOT, "GameRunner", "bin", "Release", "net8.0", "GameRunner")

PATRONS = "ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA"
DEFAULT_OPPONENT = "MaxPrestigeBot"
DEFAULT_SEED = 20260922
DEFAULT_MOVES = 10


def run_game(binary, bot, opponent, seed, timeout, data_dir):
    """One game, bot as P1, with per-turn (state, move) logging. Returns the
    list of moves P1 chose, or None if the game did not end cleanly (the
    logging wrapper discards the buffer in that case)."""
    os.makedirs(data_dir, exist_ok=True)
    cmd = [binary, bot, opponent, "--runs", "1", "--timeout", str(timeout),
           "--seed", str(seed), "--patrons", PATRONS,
           "--log-training-data", "--data-dir", data_dir]

    env = os.environ.copy()
    # The whole question is what DeepSetsBotExp does with NOTHING set. An
    # exported SOT_ALPHA0 in the calling shell would quietly answer a different
    # one.
    for key in ("SOT_ALPHA0", "SOT_TIME_SCALE", "SOT_MODEL_PATH", "SOT_LOG",
                "SOT_LOG_FILE", "SOT_DUMP_DIR"):
        env.pop(key, None)

    proc = subprocess.run(cmd, cwd=os.path.dirname(binary), env=env,
                          capture_output=True, text=True, timeout=max(1800, timeout * 180))
    if proc.returncode != 0:
        print(f"  ERROR: {bot} run exited {proc.returncode}")
        print("  " + "\n  ".join((proc.stderr or proc.stdout or "(no output)").splitlines()[:15]))
        return None, None

    reason = None
    for line in proc.stdout.splitlines():
        if line.startswith("GAME_END_REASON:"):
            reason = line.split()[1]

    shards = sorted(glob.glob(os.path.join(data_dir, "*_bot1.jsonl.gz")))
    if not shards:
        return None, reason
    moves = []
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                moves.append(json.loads(line)["data"]["action"])
    return moves, reason


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def compare(label_a, moves_a, label_b, moves_b, n):
    head_a, head_b = moves_a[:n], moves_b[:n]
    idx = first_divergence(head_a, head_b)

    print()
    print(f"  {'#':>3}  {label_a:<46}  {label_b:<46}")
    print("  " + "-" * 99)
    for i in range(max(len(head_a), len(head_b))):
        x = head_a[i] if i < len(head_a) else "(no move)"
        y = head_b[i] if i < len(head_b) else "(no move)"
        mark = "  " if x == y else "!!"
        print(f"  {i:>3}  {x[:46]:<46}  {y[:46]:<46} {mark}")

    print()
    if len(moves_a) < n or len(moves_b) < n:
        print(f"  NOTE: only {min(len(moves_a), len(moves_b))} move(s) available "
              f"({len(moves_a)} and {len(moves_b)} logged), fewer than the {n} requested.")
    if idx is None:
        print(f"  RESULT: the first {min(n, len(head_a), len(head_b))} moves are IDENTICAL.")
    else:
        print(f"  RESULT: sequences first differ at move index {idx}.")
    return idx


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--moves", type=int, default=DEFAULT_MOVES,
                        help=f"How many of P1's moves to compare (default: {DEFAULT_MOVES})")
    parser.add_argument("--opponent", default=DEFAULT_OPPONENT,
                        help=f"Opponent for both runs (default: {DEFAULT_OPPONENT}, which is "
                             f"instant and deterministic, so it contributes no timing noise of "
                             f"its own)")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--self-check", action="store_true",
                        help="Run DeepSetsBlendBot against itself in two separate processes instead. "
                             "This is the timing-noise baseline: any divergence it shows is caused by "
                             "wall-clock budgeting alone, with no code difference involved.")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary game logs")
    args = parser.parse_args()

    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        sys.exit(f"ERROR: GameRunner not found or not executable at {binary}\n"
                 f"       Build it first: dotnet build GameRunner/GameRunner.csproj -c Release")

    bot_a = "DeepSetsBlendBot"
    bot_b = "DeepSetsBlendBot" if args.self_check else "DeepSetsBotExp"

    print("=" * 101)
    print("DeepSetsBotExp vs DeepSetsBlendBot -- move-sequence comparison"
          if not args.self_check else
          "DeepSetsBlendBot vs itself -- wall-clock timing-noise baseline")
    print("=" * 101)
    print(f"  seed      : {args.seed}")
    print(f"  opponent  : {args.opponent}")
    print(f"  timeout   : {args.timeout}s/turn")
    print(f"  comparing : the first {args.moves} moves P1 chose")
    print(f"  env       : SOT_ALPHA0 / SOT_TIME_SCALE / SOT_MODEL_PATH explicitly unset")

    work_dir = tempfile.mkdtemp(prefix="sot_parity_")
    try:
        print()
        print(f"  running {bot_a} ...", flush=True)
        moves_a, reason_a = run_game(binary, bot_a, args.opponent, args.seed, args.timeout,
                                     os.path.join(work_dir, "run_a"))
        print(f"  running {bot_b} ...", flush=True)
        moves_b, reason_b = run_game(binary, bot_b, args.opponent, args.seed, args.timeout,
                                     os.path.join(work_dir, "run_b"))

        print(f"  end reasons: {bot_a}={reason_a}, {bot_b}={reason_b}")
        if moves_a is None or moves_b is None:
            print()
            print("  INCONCLUSIVE: at least one game produced no move log. GameRunner's")
            print("  --log-training-data wrapper only writes cleanly-ended games")
            print("  (PRESTIGE_OVER_40_NOT_MATCHED, PRESTIGE_OVER_80, PATRON_FAVOR); a turn")
            print("  limit or a timeout discards the buffer. Try another --seed.")
            sys.exit(2)

        label_b = f"{bot_b} (run 2)" if args.self_check else bot_b
        compare(f"{bot_a} (run 1)" if args.self_check else bot_a, moves_a, label_b, moves_b,
                args.moves)

        if not args.self_check:
            print()
            print("  Reminder: the search is wall-clock budgeted, so a divergence here is not by")
            print("  itself a defect. Run --self-check to see what divergence pure timing noise")
            print("  produces between two runs of the SAME agent, and compare.")
    finally:
        if args.keep:
            print(f"\n  logs kept in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
