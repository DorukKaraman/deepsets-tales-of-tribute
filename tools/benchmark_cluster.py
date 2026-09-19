"""
One-game-per-SLURM-array-task benchmark runner for the DeepSets bot family.
Invoked by tools/benchmark_cluster.sh -- not usually run directly (it skips
the build + Bots.dll/onnx integrity checks tools/benchmark_cluster.sh does
before exec'ing here).

DESIGN: exactly one game per OS process, exactly one process per SLURM array
task (--task-id). This is the same reasoning tools/benchmark_runner.py already
uses locally (GameRunner's GameEndStatsCounter only reports aggregate counts
per process, so --runs 1 is what lets a single game's outcome be attributed
exactly) -- just with SLURM's own array scheduler providing the concurrency
instead of a local ThreadPoolExecutor. There is no --jobs flag here: 4000
tasks running 32-wide is a property of how the array is submitted
(scripts/slurm_benchmark.sh's %32 throttle), not of this script.

TASK ID LAYOUT: 10 matchups (MATCHUPS below) x GAMES_PER_MATCHUP games each,
laid out contiguously: task_id // GAMES_PER_MATCHUP selects the matchup,
task_id % GAMES_PER_MATCHUP selects the game within it. Within each matchup,
the first half of games run bot_a as P1 (swapped=False) and the second half
run bot_b as P1 (swapped=True) -- same 50/50 split tools/benchmark_runner.py
uses locally, and for the same reason (first-player advantage is real, so an
unswapped result isn't a clean measurement of anything). seed = SEED_BASE +
task_id, unique and collision-free across the whole 4000-task space by
construction.

SEAT-SWAP INVERSION -- read this before touching winner logic: GameRunner
always reports "P1 wins" / "P2 wins", not "bot_a wins" / "bot_b wins". Task N
determines swapped from its position within its matchup (see above); the
ACTUAL processes passed to GameRunner are (bot_a, bot_b) if not swapped, or
(bot_b, bot_a) if swapped. Converting the P1/P2 result back to a bot_a/bot_b
result therefore means: if swapped, a "P1 win" IS a bot_b win, not a bot_a
win. Get this backwards and every matchup's aggregate win rate lands close to
50% (roughly: real skill gap averaged against its own mirror image) --
plausible-looking and wrong. See resolve_winner() below, and verify its output
against the printed "p1={p1}/p2={p2}" plus swapped flag directly, not just the
final "winner" label, if you ever touch this.

RESUMABILITY: a task's result file (see result_path()) is written ONLY after
a game completes and its stats are parsed successfully -- never before, never
on a process crash/timeout. A crashed/killed/still-running task therefore has
no result file and is retried by simply resubmitting the same array indices;
an already-completed task is detected (result file exists and parses with
"completed": true) and skipped immediately, without invoking GameRunner at
all. Written atomically (temp file + os.replace) so a task killed mid-write
never leaves a result file that looks valid but isn't.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

TIMEOUT_S = 10
PATRONS = "ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA"
GAMES_PER_MATCHUP = 400

# Exact order as specified -- matchup_index below is this list's index, and
# is embedded in every result file and output subdirectory name, so changing
# this order after a run has started would misattribute in-flight/resumed
# tasks. Add new matchups at the END if this ever needs to grow.
MATCHUPS = [
    ("DeepSetsBot", "SakkirinaSolo"),
    ("DeepSetsBotTrim", "SakkirinaSolo"),
    ("DeepSetsBlendBot", "SakkirinaSolo"),
    ("DeepSetsBlendBotTrim", "SakkirinaSolo"),
    ("DeepSetsBot", "MCTSBot"),
    ("DeepSetsBotTrim", "MCTSBot"),
    ("DeepSetsBlendBot", "MCTSBot"),
    ("DeepSetsBlendBotTrim", "MCTSBot"),
    ("DeepSetsBot", "DeepSetsBotTrim"),
    ("DeepSetsBlendBot", "DeepSetsBlendBotTrim"),
]

TOTAL_TASKS = len(MATCHUPS) * GAMES_PER_MATCHUP

# Must match DataLoggingWrapper.CleanEndReasons in GameRunner/Program.cs / the
# same convention tools/generate_data.py uses: these three end reasons are
# genuine decided games; TURN_LIMIT_EXCEEDED and "other factors" (exceptions,
# timeouts, illegal moves) are not a real measurement of either bot's skill.
LINE_PATTERNS = {
    "draws": re.compile(r"Final amount of draws:\s*(\d+)/(\d+)"),
    "p1_wins": re.compile(r"Final amount of P1 wins:\s*(\d+)/(\d+)"),
    "p2_wins": re.compile(r"Final amount of P2 wins:\s*(\d+)/(\d+)"),
    "prestige40": re.compile(r"Ends due to Prestige>40:\s*(\d+)/(\d+)"),
    "prestige80": re.compile(r"Ends due to Prestige>80:\s*(\d+)/(\d+)"),
    "patron_favor": re.compile(r"Ends due to Patron Favor:\s*(\d+)/(\d+)"),
    "turn_limit": re.compile(r"Ends due to Turn Limit:\s*(\d+)/(\d+)"),
    "other": re.compile(r"Ends due to other factors:\s*(\d+)/(\d+)"),
}
REQUIRED_KEYS = set(LINE_PATTERNS)
CLEAN_REASONS = {"prestige40", "prestige80", "patron_favor"}

# Safety-net subprocess watchdog. SLURM's own --time is the real per-task
# limit (see scripts/slurm_benchmark.sh); this just makes sure a hang doesn't
# also wedge whatever invoked this script directly (e.g. a local --dry-run
# session testing a single --task-id for real). Same multiplier
# tools/benchmark_runner.py / tools/generate_data.py use.
PROC_TIMEOUT_S = max(1800, TIMEOUT_S * 180)


def resolve_task(task_id, seed_base):
    if not (0 <= task_id < TOTAL_TASKS):
        raise ValueError(f"task_id {task_id} out of range [0, {TOTAL_TASKS})")

    matchup_index = task_id // GAMES_PER_MATCHUP
    game_index = task_id % GAMES_PER_MATCHUP
    bot_a, bot_b = MATCHUPS[matchup_index]
    swapped = game_index >= GAMES_PER_MATCHUP // 2
    seed = seed_base + task_id
    p1, p2 = (bot_b, bot_a) if swapped else (bot_a, bot_b)

    return {
        "task_id": task_id,
        "matchup_index": matchup_index,
        "matchup": f"{bot_a}_vs_{bot_b}",
        "bot_a": bot_a,
        "bot_b": bot_b,
        "game_index": game_index,
        "swapped": swapped,
        "seed": seed,
        "p1": p1,
        "p2": p2,
    }


def result_path(out_dir, task):
    matchup_dir = os.path.join(
        out_dir, f"matchup_{task['matchup_index']:02d}_{task['matchup']}")
    return os.path.join(matchup_dir, f"task_{task['task_id']:04d}.json")


def load_existing_result(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("completed") is True:
        return data
    return None


def write_result_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def resolve_winner(p1_wins, p2_wins, swapped):
    """Returns "bot_a", "bot_b", or "draw". See the seat-swap docstring note
    at the top of this file before changing this function."""
    if p1_wins == 1:
        return "bot_b" if swapped else "bot_a"
    if p2_wins == 1:
        return "bot_a" if swapped else "bot_b"
    return "draw"


def resolve_end_reason(parsed):
    for key in ("prestige40", "prestige80", "patron_favor", "turn_limit", "other"):
        if parsed.get(key, (0, 0))[0] == 1:
            return key
    return "unknown"


def run_task(binary, task, out_dir):
    """Runs exactly one game for `task`. Returns (ok: bool, message: str).
    Writes a result file on success only."""
    cmd = [binary, task["p1"], task["p2"], "--runs", "1", "--timeout", str(TIMEOUT_S),
           "--seed", str(task["seed"]), "--patrons", PATRONS]

    env = os.environ.copy()
    env.pop("SOT_LOG", None)
    env.pop("SOT_LOG_FILE", None)
    env.pop("SOT_DUMP_DIR", None)

    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(binary), env=env,
                               capture_output=True, text=True, timeout=PROC_TIMEOUT_S)
        stdout, stderr, returncode = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        return False, f"killed by {PROC_TIMEOUT_S}s watchdog -- no data, task is retriable"
    wall_clock_s = time.time() - start

    parsed = {}
    for key, pattern in LINE_PATTERNS.items():
        m = pattern.search(stdout)
        if m:
            parsed[key] = (int(m.group(1)), int(m.group(2)))

    if returncode != 0 or not REQUIRED_KEYS.issubset(parsed):
        stderr_head = "\n".join((stderr.strip().splitlines() or stdout.strip().splitlines() or ["(no output)"])[:20])
        return False, (f"exit code {returncode}, parsed {len(parsed)}/{len(REQUIRED_KEYS)} stats lines "
                        f"-- no data, task is retriable. stderr/stdout head:\n{stderr_head}")

    p1_wins = parsed["p1_wins"][0]
    p2_wins = parsed["p2_wins"][0]
    winner = resolve_winner(p1_wins, p2_wins, task["swapped"])
    end_reason = resolve_end_reason(parsed)

    result = {
        "task_id": task["task_id"],
        "matchup_index": task["matchup_index"],
        "matchup": task["matchup"],
        "bot_a": task["bot_a"],
        "bot_b": task["bot_b"],
        "game_index": task["game_index"],
        "swapped": task["swapped"],
        "seed": task["seed"],
        "p1": task["p1"],
        "p2": task["p2"],
        "completed": True,
        "clean": end_reason in CLEAN_REASONS,
        "winner": winner,
        "end_reason": end_reason,
        "wall_clock_s": round(wall_clock_s, 1),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_result_atomic(result_path(out_dir, task), result)
    return True, (f"winner={winner} end_reason={end_reason} clean={result['clean']} "
                   f"wall_clock={wall_clock_s:.1f}s")


def print_task_plan(task):
    print(f"task_id        : {task['task_id']} / {TOTAL_TASKS}")
    print(f"matchup        : [{task['matchup_index']}] {task['bot_a']} vs {task['bot_b']}")
    print(f"game_index     : {task['game_index']} / {GAMES_PER_MATCHUP} "
          f"({'swapped' if task['swapped'] else 'not swapped'})")
    print(f"seed           : {task['seed']}")
    print(f"GameRunner P1/P2: {task['p1']} vs {task['p2']}")


def print_full_plan(seed_base, out_dir):
    print(f"Total matchups : {len(MATCHUPS)}")
    print(f"Games/matchup  : {GAMES_PER_MATCHUP} ({GAMES_PER_MATCHUP // 2} not swapped, "
          f"{GAMES_PER_MATCHUP // 2} swapped)")
    print(f"Total tasks    : {TOTAL_TASKS}  (task_id 0..{TOTAL_TASKS - 1})")
    print(f"Seed base      : {seed_base}  (seed = seed_base + task_id, range "
          f"[{seed_base}, {seed_base + TOTAL_TASKS - 1}])")
    print(f"Timeout        : {TIMEOUT_S}s/move")
    print(f"Patrons        : {PATRONS}")
    print()
    print(f"{'idx':>3}  {'matchup':<45}  {'task_id range':<15}  {'done':>6}  {'pending':>7}")
    total_done = 0
    for i, (bot_a, bot_b) in enumerate(MATCHUPS):
        lo, hi = i * GAMES_PER_MATCHUP, (i + 1) * GAMES_PER_MATCHUP - 1
        matchup_dir = os.path.join(out_dir, f"matchup_{i:02d}_{bot_a}_vs_{bot_b}")
        done = 0
        if os.path.isdir(matchup_dir):
            for task_id in range(lo, hi + 1):
                task = resolve_task(task_id, seed_base)
                if load_existing_result(result_path(out_dir, task)) is not None:
                    done += 1
        total_done += done
        print(f"{i:>3}  {bot_a + ' vs ' + bot_b:<45}  {lo:>6}-{hi:<7}  {done:>6}  {GAMES_PER_MATCHUP - done:>7}")
    print()
    print(f"Total: {total_done}/{TOTAL_TASKS} already done, {TOTAL_TASKS - total_done} pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", help="Path to the built GameRunner binary (required unless --dry-run)")
    parser.add_argument("--task-id", type=int, default=None,
                         help="Global task id, 0..%d. Required unless --dry-run is given with no --task-id "
                              "(which prints the whole plan instead)." % (TOTAL_TASKS - 1))
    parser.add_argument("--out-dir", required=True, help="Output directory for result JSON files")
    parser.add_argument("--seed-base", type=int, required=True,
                         help="Fixed seed base (seed = seed-base + task-id). No time-derived default on "
                              "purpose -- must stay fixed across the whole array and any resubmission, or "
                              "retried tasks would silently regenerate different games than originally planned.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the plan (whole plan if --task-id is omitted, one task's plan if given); run nothing.")
    parser.add_argument("--configuration", default="unknown", help="For traceability only")
    parser.add_argument("--onnx-sha256", default=None, help="For traceability only")
    parser.add_argument("--bots-dll-sha256", default=None, help="For traceability only")
    args = parser.parse_args()

    if args.dry_run:
        if args.task_id is None:
            print_full_plan(args.seed_base, args.out_dir)
        else:
            try:
                print_task_plan(resolve_task(args.task_id, args.seed_base))
            except ValueError as e:
                sys.exit(f"ERROR: {e}")
        return

    if args.task_id is None:
        sys.exit("ERROR: --task-id is required (except for a plan-only --dry-run).")
    if not args.binary:
        sys.exit("ERROR: --binary is required.")

    args.binary = os.path.abspath(args.binary)
    if not os.path.isfile(args.binary) or not os.access(args.binary, os.X_OK):
        sys.exit(f"ERROR: binary not found or not executable: {args.binary}")

    try:
        task = resolve_task(args.task_id, args.seed_base)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    out_dir = os.path.abspath(args.out_dir)
    path = result_path(out_dir, task)

    print(f"Task {task['task_id']}/{TOTAL_TASKS}: [{task['matchup_index']}] "
          f"{task['bot_a']} vs {task['bot_b']}, game {task['game_index']}/{GAMES_PER_MATCHUP}, "
          f"swapped={task['swapped']}, seed={task['seed']}")
    if args.configuration != "unknown":
        print(f"Configuration  : {args.configuration}")
    if args.bots_dll_sha256:
        print(f"Bots.dll sha256: {args.bots_dll_sha256}")
    if args.onnx_sha256:
        print(f"Onnx sha256    : {args.onnx_sha256}")

    existing = load_existing_result(path)
    if existing is not None:
        print(f"Result already exists at {path} -- skipping (winner={existing.get('winner')}, "
              f"end_reason={existing.get('end_reason')}). Delete the file to force a re-run.")
        return

    ok, message = run_task(args.binary, task, out_dir)
    print(message)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
