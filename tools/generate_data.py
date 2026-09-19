"""
Process-parallel training-data generation for <bot> vs <bot> (default
SakkirinaGenNeural, our current best agent -- see --bot) via GameRunner's
--log-training-data. Invoked by tools/generate_data.sh -- not usually run
directly (it skips the build + Bots.dll/onnx integrity checks
tools/generate_data.sh does before exec'ing here).

DESIGN: one OS process per JOB (bounded by --jobs concurrent processes), each
running `--runs N` so GameRunner's own bot-instance-reuse-across-games
amortizes process startup -- this is deliberately NOT one process per game
(see tools/benchmark_runner.py for that model; it doesn't fit here because a
10k-game generation run needs process count decoupled from game count).
Each job gets its own contiguous seed range and its own subdirectory under
--out-dir, so GameRunner's existing per-process shard naming (pid + worker id
+ matchup) can never collide across jobs even if two jobs happen to share a
PID on different machines in a cluster.

RESUMABLE AT JOB GRANULARITY, not game granularity: a job writes a marker
file (tools/out/.../.progress/job_NNNN.json) only after it exits cleanly with
a fully-parsed stats block. Relaunching the same command skips any job whose
marker matches the current plan (same seed, same game count) and reuses its
recorded stats for the summary. A job with no marker -- whether it never ran,
or died partway through --runs N -- has its output subdirectory wiped and is
run again from scratch; GameRunner has no way to resume a single process
mid-`--runs`, so partial per-job progress cannot be preserved, only whole-job
progress. This is why the marker is written strictly after success, never
before or during.
"""
import argparse
import datetime
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_BOT_NAME = "SakkirinaGenNeural"
TIMEOUT_S = 10
PATRONS = "ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA"

STDERR_HEAD_LINES = 30

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

# Must match DataLoggingWrapper.CleanEndReasons in GameRunner/Program.cs
# exactly -- these are the three GameEndReasons whose turns actually get
# written to a shard. TURN_LIMIT_EXCEEDED and "other factors" (BOT_EXCEPTION,
# TURN_TIMEOUT, INCORRECT_MOVE, etc.) are discarded by the wrapper even
# though GameRunner's own GameEndStatsCounter still counts them as completed
# games -- so "completed" and "clean/logged" are different numbers, both
# worth reporting for a data-generation run.
CLEAN_REASON_KEYS = ("prestige40", "prestige80", "patron_favor")
DISCARDED_REASON_KEYS = ("turn_limit", "other")


def plan_jobs(games, jobs, seed_base, out_dir):
    games_per_job = games // jobs
    remainder = games % jobs
    plan = []
    seed = seed_base
    for job_id in range(jobs):
        n = games_per_job + (1 if job_id < remainder else 0)
        if n == 0:
            continue  # more jobs than games requested; no work for this slot
        plan.append({
            "job_id": job_id,
            "games": n,
            "seed": seed,
            "out_dir": os.path.join(out_dir, f"job_{job_id:04d}"),
        })
        seed += n
    return plan


def build_command(binary, job, bot_name):
    return [
        binary, bot_name, bot_name,
        "--runs", str(job["games"]),
        "--timeout", str(TIMEOUT_S),
        "--seed", str(job["seed"]),
        "--patrons", PATRONS,
        "--log-training-data",
        "--data-dir", job["out_dir"],
    ]


def marker_path(progress_dir, job_id):
    return os.path.join(progress_dir, f"job_{job_id:04d}.json")


def load_marker(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_failure_file(failures_dir, job_id, returncode, stdout, stderr, note):
    os.makedirs(failures_dir, exist_ok=True)
    path = os.path.join(failures_dir, f"job_{job_id:04d}.txt")
    with open(path, "w") as f:
        f.write(f"job_id={job_id} returncode={returncode}\n")
        f.write(f"note: {note}\n")
        f.write("\n--- STDOUT ---\n")
        f.write(stdout if stdout else "(empty)")
        f.write("\n\n--- STDERR ---\n")
        f.write(stderr if stderr else "(empty)")
        f.write("\n")
    return path


def run_one_job(binary, job, proc_timeout, failures_dir, bot_name):
    cmd = build_command(binary, job, bot_name)
    result = {
        "job_id": job["job_id"], "seed": job["seed"], "games": job["games"],
        "ok": False, "error": None, "returncode": None, "stderr_head": None,
        "failure_file": None, "wall_clock_s": None, "cmd": cmd,
    }
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(binary),
                               capture_output=True, text=True, timeout=proc_timeout)
        stdout, stderr, returncode = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr = e.stdout or "", e.stderr or ""
        stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        note = f"killed by {proc_timeout}s watchdog after {job['games']} requested games"
        result["error"] = note
        result["wall_clock_s"] = time.time() - start
        result["stderr_head"] = "\n".join((stderr.strip().splitlines() or ["(no stderr captured before kill)"])[:STDERR_HEAD_LINES])
        result["failure_file"] = write_failure_file(failures_dir, job["job_id"], "TIMEOUT (killed)", stdout, stderr, note)
        return result

    result["wall_clock_s"] = time.time() - start

    parsed = {}
    for key, pattern in LINE_PATTERNS.items():
        m = pattern.search(stdout)
        if m:
            parsed[key] = (int(m.group(1)), int(m.group(2)))

    if returncode != 0 or not REQUIRED_KEYS.issubset(parsed):
        if not stdout.strip():
            stdout_state = "stdout was completely empty"
        elif not REQUIRED_KEYS.issubset(parsed):
            stdout_state = f"stdout was non-empty but only {len(parsed)}/{len(REQUIRED_KEYS)} expected stats lines were found"
        else:
            stdout_state = "stdout parsed fine, but exit code was non-zero"
        note = f"exit code {returncode}; {stdout_state}"

        stderr_lines = stderr.strip().splitlines()
        head_source = stderr_lines if stderr_lines else stdout.strip().splitlines()
        result["error"] = note
        result["returncode"] = returncode
        result["stderr_head"] = "\n".join(head_source[:STDERR_HEAD_LINES] if head_source else ["(no output at all)"])
        result["failure_file"] = write_failure_file(failures_dir, job["job_id"], returncode, stdout, stderr, note)
        return result

    total_checked = parsed["draws"][1]
    if total_checked != job["games"]:
        note = f"GameRunner reported {total_checked} games checked, expected {job['games']}"
        result["error"] = note
        result["stderr_head"] = f"(stats parsed fine, but count mismatch) {note}"
        result["failure_file"] = write_failure_file(failures_dir, job["job_id"], returncode, stdout, stderr, note)
        return result

    result["ok"] = True
    for key in LINE_PATTERNS:
        result[key] = parsed[key][0]
    result["clean_logged"] = sum(result[k] for k in CLEAN_REASON_KEYS)
    result["discarded"] = sum(result[k] for k in DISCARDED_REASON_KEYS)
    return result


def count_gzip_lines(path):
    count = 0
    with gzip.open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="Path to the built GameRunner binary")
    parser.add_argument("--bot", default=DEFAULT_BOT_NAME,
                         help=f"Bot to self-play, both sides (default: {DEFAULT_BOT_NAME})")
    parser.add_argument("--games", type=int, required=True, help="Total games across all jobs")
    parser.add_argument("--jobs", type=int, default=None,
                         help="Max concurrent OS processes (default: detected CPU count)")
    parser.add_argument("--out-dir", required=True, help="Output directory for shards + progress markers")
    parser.add_argument("--seed-base", type=int, default=None,
                         help="First seed to use (default: derived from current time)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the plan and every job's command line; run nothing")
    parser.add_argument("--configuration", default="unknown", help="Build configuration, for traceability only")
    parser.add_argument("--bots-dll-sha256", default=None, help="For traceability only")
    parser.add_argument("--onnx-sha256", default=None,
                         help="For traceability only -- printed in the header. tools/generate_data.sh "
                              "passes the sha256 it already verified during its pre-flight check.")
    args = parser.parse_args()

    jobs = args.jobs if args.jobs is not None else (os.cpu_count() or 4)
    if jobs < 1:
        sys.exit("ERROR: --jobs must be >= 1")
    if args.games < 1:
        sys.exit("ERROR: --games must be >= 1")

    seed_base = args.seed_base if args.seed_base is not None else int(datetime.datetime.now().timestamp())
    out_dir = os.path.abspath(args.out_dir)
    progress_dir = os.path.join(out_dir, ".progress")
    failures_dir = os.path.join(progress_dir, "failures")

    plan = plan_jobs(args.games, jobs, seed_base, out_dir)
    n_empty = jobs - len(plan)

    print(f"Bot            : {args.bot} vs {args.bot}")
    print(f"Patrons        : {PATRONS}")
    print(f"Timeout        : {TIMEOUT_S}s/move")
    print(f"Games          : {args.games}")
    print(f"Jobs           : {jobs} concurrent OS processes"
          + (f" ({n_empty} idle -- more jobs than games)" if n_empty else ""))
    print(f"Seed base      : {seed_base}")
    print(f"Out dir        : {out_dir}")
    print(f"Binary         : {args.binary}")
    print(f"Configuration  : {args.configuration}")
    if args.bots_dll_sha256:
        print(f"Bots.dll sha256: {args.bots_dll_sha256}")
    if args.onnx_sha256:
        print(f"Onnx sha256    : {args.onnx_sha256}")
    print()

    if args.dry_run:
        print(f"DRY RUN -- {len(plan)} job(s) planned, nothing will execute:")
        print()
        for job in plan:
            cmd = build_command(args.binary, job, args.bot)
            print(f"  job {job['job_id']:04d}: {job['games']} games, seed={job['seed']}, "
                  f"out_dir={job['out_dir']}")
            print(f"    {' '.join(cmd)}")
        print()
        print(f"Total: {len(plan)} job(s), {sum(j['games'] for j in plan)} games.")
        return

    args.binary = os.path.abspath(args.binary)
    if not os.path.isfile(args.binary) or not os.access(args.binary, os.X_OK):
        sys.exit(f"ERROR: binary not found or not executable: {args.binary}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    # Generous per-job watchdog, scaled by how many games this job will play
    # sequentially: same per-game budget tools/benchmark_runner.py uses
    # (max(1800, timeout*180)), multiplied out across the job's whole --runs N.
    per_game_watchdog = max(1800, TIMEOUT_S * 180)

    to_run = []
    resumed = []
    for job in plan:
        marker = load_marker(marker_path(progress_dir, job["job_id"]))
        if marker is not None and marker.get("seed") == job["seed"] and marker.get("games") == job["games"] and marker.get("ok"):
            resumed.append((job, marker))
        else:
            if marker is not None:
                print(f"  job {job['job_id']:04d}: marker present but does not match the current plan "
                      f"(stale from a different --games/--jobs/--seed-base) -- re-running.")
            to_run.append(job)

    print(f"Plan: {len(plan)} job(s) total -- {len(resumed)} already complete (resumed), "
          f"{len(to_run)} to run now.")
    print()

    overall_start = time.time()
    fresh_results = []
    if to_run:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(run_one_job, args.binary, job,
                            job["games"] * per_game_watchdog, failures_dir, args.bot): job
                for job in to_run
            }
            done_count = 0
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"job_id": job["job_id"], "seed": job["seed"], "games": job["games"],
                         "ok": False, "error": f"unhandled exception in run_one_job: {e!r}",
                         "returncode": None, "stderr_head": None, "failure_file": None,
                         "wall_clock_s": None, "cmd": build_command(args.binary, job, args.bot)}
                fresh_results.append(r)
                done_count += 1
                status = "ok" if r["ok"] else "FAILED"
                print(f"  ... job {job['job_id']:04d} finished ({status}, "
                      f"{done_count}/{len(to_run)} jobs done this run)")

                if r["ok"]:
                    marker = {
                        "job_id": r["job_id"], "seed": r["seed"], "games": r["games"], "ok": True,
                        "draws": r["draws"], "p1_wins": r["p1_wins"], "p2_wins": r["p2_wins"],
                        "prestige40": r["prestige40"], "prestige80": r["prestige80"],
                        "patron_favor": r["patron_favor"], "turn_limit": r["turn_limit"],
                        "other": r["other"], "clean_logged": r["clean_logged"],
                        "discarded": r["discarded"], "wall_clock_s": r["wall_clock_s"],
                        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                    with open(marker_path(progress_dir, r["job_id"]), "w") as f:
                        json.dump(marker, f, indent=2)
    wall_clock_this_run = time.time() - overall_start

    all_results = [(job, marker, False) for job, marker in resumed] + \
                  [(None, r, True) for r in fresh_results]

    games_attempted = sum(j["games"] for j in plan)
    ok_records = [r for _, r, _ in all_results if r.get("ok")]
    failed_records = [r for _, r, _ in all_results if not r.get("ok")]

    games_completed = sum(r["games"] for r in ok_records)
    games_failed = sum(r["games"] for r in failed_records)
    clean_logged_total = sum(r.get("clean_logged", 0) for r in ok_records)
    discarded_total = sum(r.get("discarded", 0) for r in ok_records)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  jobs total             : {len(plan)}  ({len(resumed)} resumed from a prior run, {len(to_run)} run now)")
    print(f"  wall clock (this run)  : {wall_clock_this_run:.1f}s")
    print()
    print(f"  games attempted        : {games_attempted}")
    print(f"  games completed        : {games_completed}  (GameRunner finished and reported a result)")
    print(f"  games failed           : {games_failed}  (job process crashed/hung/errored -- NO data, will "
          f"be redone on next relaunch)")
    print()
    print(f"  of the {games_completed} completed games:")
    print(f"    clean / logged to shards : {clean_logged_total}  (Prestige>40, Prestige>80, Patron Favor)")
    print(f"    discarded by DataLoggingWrapper : {discarded_total}  (Turn Limit, or other/error factors)")

    if failed_records:
        print()
        print(f"  WARNING: {len(failed_records)} job(s) failed, accounting for {games_failed} unplayed games.")
        print("  These jobs have no marker and will be retried on the next invocation with the same args.")
        for r in failed_records[:10]:
            print(f"    job {r['job_id']:04d} (seed={r['seed']}, {r['games']} games): {r['error']}")
            if r.get("failure_file"):
                print(f"      full output: {r['failure_file']}")
        if len(failed_records) > 10:
            print(f"    ... and {len(failed_records) - 10} more; see {failures_dir}")

    print()
    print("  shards on disk (all jobs, including resumed):")
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "job_*", "*.jsonl.gz")))
    total_records = 0
    for path in shard_paths:
        n = count_gzip_lines(path)
        total_records += n
        rel = os.path.relpath(path, out_dir)
        print(f"    {rel:60s} {n:8d} records")
    print(f"  total shards           : {len(shard_paths)}")
    print(f"  total records written  : {total_records}")

    print()
    print("Done.")

    if failed_records:
        sys.exit(1)


if __name__ == "__main__":
    main()
