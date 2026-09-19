"""
Process-parallel benchmark runner for ISMCTSBot (or any two bots) against
GameRunner. Invoked by tools/benchmark.sh -- not usually run directly.

DESIGN: one game per OS process (--runs 1 per invocation), pooled through a
bounded set of concurrent subprocesses (--jobs). This is deliberately NOT
"--jobs worker processes each playing --games/--jobs games" -- GameRunner's
GameEndStatsCounter.ToString() only reports aggregate win/draw/other-factors
counts per process, with no cross-tabulation of *which* reason produced *which*
winner. With --runs 1, exactly one reason bucket and one winner line are set
per process, so each game's outcome can be attributed exactly: a clean win/loss,
or a win/loss caused by the opponent's (or own) exception/timeout/illegal move.
That exact attribution is what lets error/timeout games be reported separately
from real losses instead of silently inflating or deflating the win rate.

Every game gets its own distinct --seed. Half the games run bot-a as P1 (the
first GameRunner argument), half run bot-b as P1 (bot order swapped) -- first-
player advantage is real in this game, so an unswapped result isn't a clean
measurement of anything.

SOT_LOG and SOT_DUMP_DIR are stripped from every worker's environment
regardless of what the caller's shell has set, so benchmark-scale runs never
accidentally write gigabytes of debug logs/dumps.

Every game is run with --enable-logs BOTH, which is a separate, always-cheap
mechanism from SOT_LOG: the engine flushes each bot's AI.Log() messages to
stdout once per completed move, tagged [PLAYER][hh:mm:ss:fff][turn][move].
That's the only way to tell, after a game gets killed by the watchdog, whether
it was still making steady turn progress (a legitimately long game) or had
stalled (a hang) -- see parse_turn_progress().
"""
import argparse
import csv
import datetime
import hashlib
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "out", "benchmark_log.csv")
DEFAULT_ONNX_PATH = os.path.join(REPO_ROOT, "models", "DeepSetsValueNetwork.onnx")
FAILURES_DIR = os.path.join(SCRIPT_DIR, "out", "failures")
STDERR_HEAD_LINES = 30

# The value network is trained exclusively on the competition patron pool --
# benchmarking against GameRunner's engine-level default (a different
# 9-patron pool including PSIJIC/HLAALU/RED_EAGLE, which the network has never
# seen and which cannot occur in competition) would not be representative.
DEFAULT_PATRONS = "ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA"

CSV_HEADER = ["timestamp", "git_commit", "bot_a", "bot_b", "games", "timeout", "patrons",
              "win_rate", "ci_low", "ci_high", "onnx_sha256", "bots_dll_sha256"]

LINE_PATTERNS = {
    "draws": re.compile(r"Final amount of draws:\s*(\d+)/(\d+)"),
    "p1_wins": re.compile(r"Final amount of P1 wins:\s*(\d+)/(\d+)"),
    "p2_wins": re.compile(r"Final amount of P2 wins:\s*(\d+)/(\d+)"),
    "other": re.compile(r"Ends due to other factors:\s*(\d+)/(\d+)"),
}

# PREPARE_TIME_EXCEEDED isn't handled by GameEndStatsCounter.Add()'s switch and
# throws, crashing the whole process with no stats block at all -- worth its
# own diagnostic tag rather than lumping it in with a generic parse failure.
PREPARE_TIME_MARKER = "PREPARE_TIME_EXCEEDED"


def sha256_of(path):
    if not os.path.isfile(path):
        return "N/A (file not found)"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git_commit(repo_root):
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
    except Exception:
        return "unknown"

    try:
        status = subprocess.run(["git", "-C", repo_root, "status", "--porcelain"],
                                 capture_output=True, text=True, check=True)
        if status.stdout.strip():
            commit += "-dirty"
    except Exception:
        pass
    return commit


def bots_dll_info(path):
    """(size, mtime_iso, sha256) of the Bots.dll actually loaded by the runner,
    or ("N/A", ...) if not given/found -- this is what makes a Debug-under-
    Release mixup (or any other stale-artifact class of bug) visible after the
    fact in tools/out/benchmark_log.csv, not just at pre-flight check time."""
    if not path or not os.path.isfile(path):
        return "N/A (file not found)", "N/A", "N/A"
    size = os.path.getsize(path)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    return size, mtime, sha256_of(path)


def ensure_csv_header(csv_path, header):
    """If csv_path already exists with a DIFFERENT stored header than `header`
    (e.g. a column was just added), rewrite it in place: keep every existing
    row, backfill any newly-added column with an explicit
    'unknown (pre-<col>-column)' marker, and write the current header once at
    the top. Without this, appending new rows under a changed CSV_HEADER
    would silently produce a file whose stored header (row 1) no longer
    matches its own data rows' column count/order -- exactly the kind of
    thing that parses "fine" with a lenient reader and silently misaligns
    columns with a strict one (pandas.read_csv, a fixed-position DictReader).
    A no-op if the file doesn't exist yet or its header already matches."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames
        if existing_header == header:
            return
        rows = list(reader)

    new_cols = [c for c in header if c not in (existing_header or [])]
    for row in rows:
        for col in new_cols:
            row[col] = f"unknown (pre-{col}-column)"
        row.pop(None, None)  # DictReader's catch-all for any unexpected extra fields

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Migrated {csv_path}: added column(s) {new_cols} to {len(rows)} existing row(s), "
          f"backfilled as 'unknown (pre-<col>-column)' so past runs stay distinguishable from future ones.")


def wilson_interval(k, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (center - margin) / denom, (center + margin) / denom


def write_failure_file(seed, swapped, returncode, stdout, stderr, note):
    """Full stdout+stderr of a failed game, for offline inspection -- the console
    summary only shows the first STDERR_HEAD_LINES, which is enough to identify
    the exception but not necessarily the whole story."""
    os.makedirs(FAILURES_DIR, exist_ok=True)
    path = os.path.join(FAILURES_DIR, f"{seed}_{swapped}.txt")
    with open(path, "w") as f:
        f.write(f"seed={seed} swapped={swapped} returncode={returncode}\n")
        f.write(f"note: {note}\n")
        f.write("\n--- STDOUT ---\n")
        f.write(stdout if stdout else "(empty)")
        f.write("\n\n--- STDERR ---\n")
        f.write(stderr if stderr else "(empty)")
        f.write("\n")
    return path


# Matches Logger.cs's per-move format: "[PLAYER1][hh:mm:ss:fff][turn][move] msg".
# Only present because run_one_game passes --enable-logs BOTH.
TURN_LOG_PATTERN = re.compile(r"^\[(?:PLAYER1|PLAYER2)\]\[(\d{2}):(\d{2}):(\d{2}):(\d{3})\]\[(\d+)\]\[(\d+)\]")


def parse_turn_progress(stdout, process_start, proc_timeout):
    """Extract per-move progress lines to determine how far a killed game's turn
    counter got, and whether it was still advancing near the kill or had stalled
    well before it. Returns None if no such line was captured at all (e.g. killed
    before the very first move completed).

    process_start/proc_timeout let us reconstruct an approximate wall-clock kill
    time to compare against the log's hh:mm:ss:fff timestamps, which carry no date."""
    events = []
    for line in stdout.splitlines():
        m = TURN_LOG_PATTERN.match(line)
        if not m:
            continue
        h, mi, s, ms, turn = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        ts = process_start.replace(hour=h, minute=mi, second=s, microsecond=ms * 1000)
        if ts < process_start:
            ts += datetime.timedelta(days=1)  # midnight rollover guard
        events.append((ts, turn))
    if not events:
        return None

    events.sort(key=lambda e: e[0])
    first_ts, _ = events[0]
    last_ts, max_turn = events[-1]
    kill_ts = process_start + datetime.timedelta(seconds=proc_timeout)

    span_s = (last_ts - first_ts).total_seconds()
    avg_move_interval_s = span_s / (len(events) - 1) if len(events) > 1 else 0.0
    trailing_gap_s = (kill_ts - last_ts).total_seconds()

    # Stalled if the gap since the last logged move is far larger than the pace of
    # every move before it -- a legitimately long game keeps advancing at a roughly
    # steady clip; a hang shows progress stop well before the kill actually lands.
    stalled = trailing_gap_s > max(60.0, 5 * avg_move_interval_s)
    return {
        "max_turn": max_turn,
        "moves_logged": len(events),
        "avg_move_interval_s": avg_move_interval_s,
        "trailing_gap_s": trailing_gap_s,
        "stalled": stalled,
    }


def plan_tasks(games, seed_base):
    """Half the games bot-a-as-P1 (swapped=False), half bot-b-as-P1 (swapped=True).
    Every task gets a distinct, non-overlapping seed."""
    n_non_swapped = (games + 1) // 2
    n_swapped = games - n_non_swapped
    tasks = []
    seed = seed_base
    for _ in range(n_non_swapped):
        tasks.append({"seed": seed, "swapped": False})
        seed += 1
    for _ in range(n_swapped):
        tasks.append({"seed": seed, "swapped": True})
        seed += 1
    return tasks


def run_one_game(binary, bot_a, bot_b, timeout_s, task, proc_timeout, patrons):
    bot1, bot2 = (bot_a, bot_b) if not task["swapped"] else (bot_b, bot_a)
    cmd = [binary, bot1, bot2, "--runs", "1", "--timeout", str(timeout_s), "--seed", str(task["seed"]),
           "--enable-logs", "BOTH", "--patrons", patrons]

    env = os.environ.copy()
    env.pop("SOT_LOG", None)
    env.pop("SOT_LOG_FILE", None)
    env.pop("SOT_DUMP_DIR", None)

    result = {"seed": task["seed"], "swapped": task["swapped"], "ok": False,
              "category": None, "kind": None, "error": None, "returncode": None,
              "stderr_head": None, "failure_file": None}

    process_start = datetime.datetime.now()
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(binary), env=env,
                               capture_output=True, text=True, timeout=proc_timeout)
        stdout, stderr, returncode = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as e:
        # e.stdout/e.stderr hold whatever was captured before the process was killed --
        # but CPython does NOT decode this partial output when the timeout fires, even
        # with text=True (decoding normally happens in Popen.communicate()'s regular
        # return path, which a timeout bypasses). Coerce before it touches anything
        # that assumes str, e.g. write_failure_file's text-mode f.write().
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr

        progress = parse_turn_progress(stdout, process_start, proc_timeout)
        if progress is None:
            kind = "timeout_hung"
            note = (f"killed by {proc_timeout}s watchdog with NO per-move progress captured at all "
                     "(died before the first move completed) -- treated as a hang.")
        elif progress["stalled"]:
            kind = "timeout_hung"
            note = (f"killed by {proc_timeout}s watchdog; turn progress STALLED at turn {progress['max_turn']} "
                     f"({progress['trailing_gap_s']:.0f}s since the last move logged, vs an average of "
                     f"~{progress['avg_move_interval_s']:.1f}s/move before that) -- looks like a genuine hang.")
        else:
            kind = "timeout_incomplete"
            note = (f"killed by {proc_timeout}s watchdog after reaching turn {progress['max_turn']} "
                     f"({progress['moves_logged']} moves logged, ~{progress['avg_move_interval_s']:.1f}s/move) -- "
                     "still progressing at a steady pace, not stalled; likely a legitimately long game "
                     "(prestige racing toward the 80 cap), not a hang.")

        result["error"] = note
        result["kind"] = kind
        result["stderr_head"] = "\n".join((stderr.strip().splitlines() or ["(no stderr captured before kill)"])[:STDERR_HEAD_LINES])
        result["failure_file"] = write_failure_file(task["seed"], task["swapped"], "TIMEOUT (killed)", stdout, stderr, note)
        return result

    parsed = {}
    for key, pattern in LINE_PATTERNS.items():
        m = pattern.search(stdout)
        if m:
            parsed[key] = (int(m.group(1)), int(m.group(2)))

    if returncode != 0 or len(parsed) != 4:
        if not stdout.strip():
            stdout_state = "stdout was completely empty"
        elif len(parsed) != 4:
            stdout_state = f"stdout was non-empty but only {len(parsed)}/4 expected stats lines were found (unparseable/incomplete)"
        else:
            stdout_state = "stdout parsed fine, but exit code was non-zero"

        note = f"exit code {returncode}; {stdout_state}"
        if PREPARE_TIME_MARKER in stdout or PREPARE_TIME_MARKER in stderr:
            note += " -- PREPARE_TIME_EXCEEDED (unhandled by GameEndStatsCounter.Add(), crashes the process)"

        # .NET stack traces put the exception type/message at the TOP, not the
        # bottom -- a tail of System.CommandLine plumbing is useless here.
        stderr_lines = stderr.strip().splitlines()
        head_source = stderr_lines if stderr_lines else stdout.strip().splitlines()
        stderr_head = head_source[:STDERR_HEAD_LINES] if head_source else ["(no output at all)"]

        result["error"] = note
        result["kind"] = "process_error"
        result["returncode"] = returncode
        result["stderr_head"] = "\n".join(stderr_head)
        result["failure_file"] = write_failure_file(task["seed"], task["swapped"], returncode, stdout, stderr, note)
        return result

    p1_wins = parsed["p1_wins"][0]
    p2_wins = parsed["p2_wins"][0]
    draws = parsed["draws"][0]
    other = parsed["other"][0]

    if p1_wins == 1:
        winner = "bot_b" if task["swapped"] else "bot_a"
    elif p2_wins == 1:
        winner = "bot_a" if task["swapped"] else "bot_b"
    else:
        winner = "draw"

    is_error_game = (other == 1)
    if is_error_game:
        category = {"bot_a": "error_win_a", "bot_b": "error_win_b", "draw": "error_draw"}[winner]
    else:
        category = winner  # "bot_a", "bot_b", or "draw"

    result["ok"] = True
    result["category"] = category
    result["kind"] = "ok"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="Path to the built GameRunner binary")
    parser.add_argument("--bot-a", required=True)
    parser.add_argument("--bot-b", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True, help="Per-move timeout in seconds")
    parser.add_argument("--jobs", type=int, required=True, help="Max concurrent OS processes")
    parser.add_argument("--seed-base", type=int, default=None,
                         help="First seed to use (default: derived from current time)")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--patrons", default=DEFAULT_PATRONS,
                         help=f"Comma-separated patron pool passed to GameRunner "
                              f"(default: {DEFAULT_PATRONS} -- the competition set)")
    parser.add_argument("--onnx-path", default=DEFAULT_ONNX_PATH)
    parser.add_argument("--bots-dll-path", default=None,
                         help="Path to the Bots.dll actually loaded by the runner (for traceability)")
    parser.add_argument("--configuration", default="unknown",
                         help="Build configuration used (Debug/Release), for traceability only")
    parser.add_argument("--repeat-seed", type=int, default=None,
                         help="Re-run exactly one game with this seed only, bypassing normal "
                              "task planning/aggregation/CSV logging -- for testing whether a "
                              "failure found during a larger run is deterministic or load-dependent.")
    parser.add_argument("--repeat-swapped", action="store_true",
                         help="With --repeat-seed, use bot-b as P1 instead of bot-a (default: bot-a as P1).")
    args = parser.parse_args()

    # Resolve to absolute up front: run_one_game passes cwd=dirname(binary) to
    # subprocess.run, which breaks cmd[0] resolution if binary is relative.
    args.binary = os.path.abspath(args.binary)

    if not os.path.isfile(args.binary) or not os.access(args.binary, os.X_OK):
        sys.exit(f"ERROR: binary not found or not executable: {args.binary}")

    # Generous watchdog: engine-level timeouts (TURN_TIMEOUT etc.) already bound
    # a single move's duration, not the whole game's -- prestige-matching drives
    # ToT games past the 40 threshold toward the 80 cap when both players are
    # strong, so a legitimately long head-to-head game can run for many minutes.
    # *60 (600s at --timeout 10) was cutting those off; *180 (1800s) gives real
    # long games room while still catching genuine hangs, which parse_turn_progress
    # distinguishes from legitimately-long games regardless of where this lands.
    proc_timeout = max(1800, args.timeout * 180)

    if args.repeat_seed is not None:
        task = {"seed": args.repeat_seed, "swapped": args.repeat_swapped}
        p1 = args.bot_b if args.repeat_swapped else args.bot_a
        p2 = args.bot_a if args.repeat_swapped else args.bot_b
        print(f"Repeating a single game in isolation: seed={args.repeat_seed} "
              f"swapped={args.repeat_swapped} ({p1} as P1, {p2} as P2)")
        print()
        r = run_one_game(args.binary, args.bot_a, args.bot_b, args.timeout, task, proc_timeout, args.patrons)
        print("=" * 78)
        if r["ok"]:
            print(f"RESULT: ok -- category={r['category']}")
        else:
            print("RESULT: FAILED")
            print(f"  returncode   : {r['returncode']}")
            print(f"  error        : {r['error']}")
            print(f"  failure file : {r['failure_file']}")
            print(f"  first {STDERR_HEAD_LINES} lines of stderr (or stdout if stderr was empty):")
            for line in (r["stderr_head"] or "").splitlines():
                print(f"    | {line}")
        print("=" * 78)
        sys.exit(0 if r["ok"] else 1)

    seed_base = args.seed_base if args.seed_base is not None else int(datetime.datetime.now().timestamp())
    tasks = plan_tasks(args.games, seed_base)
    n_non_swapped = sum(1 for t in tasks if not t["swapped"])
    n_swapped = len(tasks) - n_non_swapped

    bots_dll_size, bots_dll_mtime, bots_dll_sha = bots_dll_info(args.bots_dll_path)

    print(f"Bot A          : {args.bot_a}")
    print(f"Bot B          : {args.bot_b}")
    print(f"Games          : {len(tasks)} ({n_non_swapped} with A as P1, {n_swapped} with B as P1)")
    print(f"Timeout        : {args.timeout}s/move")
    print(f"Jobs           : {args.jobs} concurrent OS processes")
    print(f"Seed base      : {seed_base}")
    print(f"Binary         : {args.binary}")
    print(f"Configuration  : {args.configuration}")
    print(f"Bots.dll       : {args.bots_dll_path or 'N/A'} (size={bots_dll_size}, mtime={bots_dll_mtime})")
    print(f"Bots.dll sha256: {bots_dll_sha}")
    print(f"onnx sha256    : {sha256_of(args.onnx_path)}")
    print(f"Patrons        : {args.patrons}")
    print()

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one_game, args.binary, args.bot_a, args.bot_b,
                                args.timeout, task, proc_timeout, args.patrons): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                # A single bad game must never abort the whole benchmark -- record it
                # as its own failure (same shape as every other failure path) and keep
                # going, so a 150-game run still reaches the RESULTS block even if one
                # game's harness code blows up.
                results.append({
                    "seed": task["seed"], "swapped": task["swapped"], "ok": False,
                    "category": None, "kind": "harness_exception",
                    "error": f"unhandled exception in run_one_game: {e!r}",
                    "returncode": None, "stderr_head": None, "failure_file": None,
                })
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  ... {done}/{len(tasks)} games finished")

    failed = [r for r in results if not r["ok"]]
    ok = [r for r in results if r["ok"]]

    # "incomplete (too long)" is a killed game that was still advancing at a steady
    # pace when the watchdog hit -- almost certainly a legitimately long game, not
    # a bug. Everything else in `failed` (stalled/hung, process errors, unhandled
    # harness exceptions) is a real failure with no data, worth investigating.
    incomplete_too_long = [r for r in failed if r.get("kind") == "timeout_incomplete"]
    genuinely_failed = [r for r in failed if r.get("kind") != "timeout_incomplete"]

    counts = {"bot_a": 0, "bot_b": 0, "draw": 0, "error_win_a": 0, "error_win_b": 0, "error_draw": 0}
    for r in ok:
        counts[r["category"]] += 1

    clean_wins_a = counts["bot_a"]
    clean_wins_b = counts["bot_b"]
    clean_draws = counts["draw"]
    decided = clean_wins_a + clean_wins_b
    win_rate, ci_low, ci_high = None, None, None
    if decided > 0:
        win_rate = clean_wins_a / decided
        ci_low, ci_high = wilson_interval(clean_wins_a, decided)

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  games requested        : {len(tasks)}")
    print(f"  games completed        : {len(ok)}")
    print(f"  games incomplete (too long, NOT a bug): {len(incomplete_too_long)}")
    print(f"  games crashed/hung/errored (NO data, may be a bug): {len(genuinely_failed)}")
    print()
    print(f"  clean wins  bot_a      : {clean_wins_a}")
    print(f"  clean wins  bot_b      : {clean_wins_b}")
    print(f"  clean draws            : {clean_draws}")
    if win_rate is not None:
        print(f"  bot_a win rate (clean, decided games only, n={decided}): "
              f"{win_rate:.4f}  [95% CI {ci_low:.4f}, {ci_high:.4f}]")
    else:
        print("  bot_a win rate: N/A (no clean decided games)")
    print()
    print("  --- error/timeout games (NOT counted as clean wins/losses above) ---")
    print(f"  bot_a won because bot_b errored/timed out : {counts['error_win_a']}")
    print(f"  bot_b won because bot_a errored/timed out : {counts['error_win_b']}  <-- not a real loss for bot_a")
    print(f"  draw caused by an error/timeout            : {counts['error_draw']}")

    if failed:
        failed_swapped = sum(1 for r in failed if r["swapped"])
        failed_nonswapped = len(failed) - failed_swapped
        print()
        print(f"  --- all {len(failed)} non-completing game(s), by seat (seat-correlation check) ---")
        print(f"  swapped=False (bot_a as P1): {failed_nonswapped}")
        print(f"  swapped=True  (bot_b as P1): {failed_swapped}")

    if incomplete_too_long:
        print()
        print(f"  {len(incomplete_too_long)} game(s) killed by the watchdog but still progressing steadily "
              "when killed -- treated as legitimately long games, NOT a bug:")
        for r in incomplete_too_long[:5]:
            print(f"    seed={r['seed']} swapped={r['swapped']}: {r['error']}")
        if len(incomplete_too_long) > 5:
            print(f"    ... and {len(incomplete_too_long) - 5} more; see tools/out/failures/ for all of them.")

    if genuinely_failed:
        print()
        print(f"  WARNING: {len(genuinely_failed)} process(es) produced NO usable data "
              "(crash, hung/stalled search, or unparseable output).")
        print("  Results above are INCOMPLETE relative to --games requested. First few failures:")
        for r in genuinely_failed[:5]:
            print(f"    seed={r['seed']} swapped={r['swapped']} returncode={r['returncode']}")
            print(f"      {r['error']}")
            print(f"      full output: {r['failure_file']}")
            print(f"      first {STDERR_HEAD_LINES} lines of stderr (or stdout if stderr was empty):")
            for line in (r["stderr_head"] or "").splitlines():
                print(f"        | {line}")
        if len(genuinely_failed) > 5:
            print(f"    ... and {len(genuinely_failed) - 5} more; see tools/out/failures/ for all of them.")

    # ---- CSV traceability ----
    if win_rate is None:
        print()
        print("Not appending to CSV: no clean decided games to report a win rate for.")
    else:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        ensure_csv_header(args.csv, CSV_HEADER)
        write_header = not os.path.isfile(args.csv) or os.path.getsize(args.csv) == 0
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(REPO_ROOT),
            "bot_a": args.bot_a,
            "bot_b": args.bot_b,
            "games": args.games,
            "timeout": args.timeout,
            "patrons": args.patrons,
            "win_rate": f"{win_rate:.4f}",
            "ci_low": f"{ci_low:.4f}",
            "ci_high": f"{ci_high:.4f}",
            "onnx_sha256": sha256_of(args.onnx_path),
            "bots_dll_sha256": bots_dll_sha,
        }
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print()
        print(f"Appended one row to {args.csv}")

    print()
    print("Done.")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
