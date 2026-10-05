"""
Run one benchmark game per SLURM array task, from a JSON experiment config.

Invoked by tools/benchmark_cluster.sh, which first checks the build and the
model hashes; running this directly skips those checks. The config format is
described in experiments/configs/README.md.

One game per process: GameRunner's stats counter reports only per-process
totals, so --runs 1 is what makes each game's outcome attributable.
Concurrency comes from how the array is submitted, not from this script.

Task ids: matchups are laid out contiguously in config order, each taking its
`games` tasks, so adding a matchup anywhere but the end renumbers every task
after it. Within a matchup the first half runs bot_a as P1 and the second half
swaps seats. seed = seed_base + task_id.

Seat swap: GameRunner reports P1 and P2 wins. When swapped, a P1 win is a
bot_b win (see resolve_winner); getting this backwards pulls every win rate
towards 50%.

End reasons: GameRunner prints a GAME_END_REASON line per game, which this
classifies as clean, turn_limit, timeout or disqualification, attributing a
non-clean end to the side that caused it.

Evaluations per turn: DeepSetsBotExp and SakkirinaScaled log an EvalsPerTurn
line through BotLog, which this enables per task (SOT_LOG=1) and parses.

Model checks, in three layers. benchmark_cluster.sh checks GameRunner's own
model against builtin_onnx_sha256. check_task_model() requires a row's
SOT_MODEL_PATH to be in allowed_onnx_sha256. After the game,
verify_loaded_models() requires the hash each bot logged to match that row's
model; a bot that fails to load its model falls back silently, so this is the
only check that confirms a row ran its own model.

Resumability: a result file is written atomically, and only after a game
completes and parses. A task without one is retried on resubmission; a task
with one is skipped.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = os.path.join(REPO_ROOT, "experiments", "configs")

DEFAULT_PATRONS = "ANSEI,DUKE_OF_CROWS,RAJHIN,ORGNUM,PELIN,SAINT_ALESSIA"
DEFAULT_GAMES = 400
DEFAULT_TIMEOUT_S = 10
DEFAULT_CALIBRATION_GAMES = 20

# Default allowed list. A config's allowed_onnx_sha256 replaces it rather than
# extending it.
SHIPPED_ONNX_SHA256 = "86e0f9a8891915bf5f151afc43c3ef98b50334d9967d79eac0ddc0b14706a915"

# A literal "TBD" marks a value not yet calibrated. Matchups containing one still
# load, so --dry-run and --calibrate work, but refuse to run a game. Checked
# case-insensitively, on env values and scalar fields.
TBD_SENTINEL = "TBD"

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

# One line per game from GameRunner/Program.cs, with the exact end reason, which
# the aggregate stats block cannot give.
GAME_END_REASON_PATTERN = re.compile(r"^GAME_END_REASON:\s*(\w+)\s+WINNER:\s*(\w+)\s*$", re.MULTILINE)

# Emitted at game end by experiments/bots/DeepSetsBotExp.cs and by
# SakkirinaScaled (scripts/sakkirina_scaled.patch), via BotLog.
EVALS_PER_TURN_PATTERN = re.compile(
    r"(\w+)\.EvalsPerTurn:\s*totalEvalCalls=(\d+),\s*turns=(\d+),\s*meanEvalsPerTurn=([0-9.]+)")

# Logged in PregamePrepare by each bot that loads an ONNX model: the path it
# resolved and the file's sha256.
LOADED_MODEL_PATTERN = re.compile(
    r"(\w+)\.PregamePrepare:\s*ONNX model loaded OK from '([^']*)',\s*"
    r"size=(\d+) bytes,\s*sha256=([0-9a-fA-F]{64})")

# Bots that load an ONNX model and log the line above. A missing line fails the
# task: either the load failed and the bot fell back to its heuristic, or the log
# was lost.
MODEL_LOADING_BOTS = {
    "DeepSetsBot", "DeepSetsBlendBot",
    "DeepSetsBotTrim", "DeepSetsBlendBotTrim",
    "DeepSetsBotExp",
}

# Of those, the bots that honour SOT_MODEL_PATH. The others always load
# GameRunner's own copy.
SOT_MODEL_PATH_BOTS = {"DeepSetsBotExp"}

# Exhaustive over ScriptsOfTribute.Board.GameEndReason. An unlisted reason is
# classified "unknown" rather than folded into one of these.
REASON_CATEGORY = {
    "PRESTIGE_OVER_40_NOT_MATCHED": "clean",
    "PRESTIGE_OVER_80": "clean",
    "PATRON_FAVOR": "clean",
    "TURN_LIMIT_EXCEEDED": "turn_limit",
    # The clock ran out: a budget effect, not a playing one.
    "TURN_TIMEOUT": "timeout",
    "PATRON_SELECTION_TIMEOUT": "timeout",
    "PREPARE_TIME_EXCEEDED": "timeout",
    # The engine rejected the agent's move: a defect, not a budget effect.
    "INCORRECT_MOVE": "disqualification",
    "BOT_EXCEPTION": "disqualification",
    "PATRON_SELECTION_FAILURE": "disqualification",
    "INTERNAL_ERROR": "disqualification",
}
NON_CLEAN_CATEGORIES = ("turn_limit", "timeout", "disqualification", "unknown")


# --------------------------------------------------------------------------
# Config loading and validation
# --------------------------------------------------------------------------

class ConfigError(Exception):
    pass


def _is_tbd(value):
    return isinstance(value, str) and value.strip().upper() == TBD_SENTINEL


def resolve_config_path(path):
    """Accepts a path, or the bare name of a config in experiments/configs/."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    for candidate in (path, path + ".json"):
        guess = os.path.join(CONFIG_DIR, candidate)
        if os.path.isfile(guess):
            return os.path.abspath(guess)
    raise ConfigError(f"config not found: {path} (looked in {CONFIG_DIR} too)")


def _normalize_matchups(raw_matchups, defaults, where):
    matchups = []
    for i, m in enumerate(raw_matchups):
        for key in ("bot_a", "bot_b"):
            if not m.get(key):
                raise ConfigError(f"{where}[{i}]: missing required field '{key}'")
        env = dict(defaults.get("env") or {})
        env.update(m.get("env") or {})
        env = {str(k): str(v) for k, v in env.items()}

        games = m.get("games", defaults.get("games", DEFAULT_GAMES))
        timeout = m.get("timeout", defaults.get("timeout", DEFAULT_TIMEOUT_S))
        label = m.get("label") or f"{m['bot_a']}_vs_{m['bot_b']}"

        # A matchup with a TBD value still loads, so --dry-run and --calibrate can run;
        # run_task refuses it (see tbd_reason).
        tbd = sorted(k for k, v in env.items() if _is_tbd(v))
        if _is_tbd(timeout):
            tbd.append("timeout")

        matchups.append({
            "index": i,
            "label": label,
            "bot_a": m["bot_a"],
            "bot_b": m["bot_b"],
            "games": games,
            "timeout": timeout,
            "env": env,
            "note": m.get("note", ""),
            "optional": bool(m.get("optional", False)),
            "tbd": tbd,
        })
    return matchups


def tbd_reason(matchup):
    """The refusal message for a matchup that still has TBD values, or None."""
    if not matchup["tbd"]:
        return None
    return (f"matchup [{matchup['index']}] {matchup['label']} still has {TBD_SENTINEL} placeholders: "
            f"{', '.join(matchup['tbd'])}.\n"
            f"  Fill them in before running this config -- see the config's own notes, and "
            f"--calibrate for where the numbers come from.")


def _validate_matchups(matchups, where):
    seen = {}
    for m in matchups:
        # Labels become directory names, so a duplicate would merge two conditions.
        if m["label"] in seen:
            raise ConfigError(
                f"{where}: duplicate label {m['label']!r} (matchups {seen[m['label']]} and {m['index']}). "
                f"Labels become result directory names, so they must be unique -- give each "
                f"repeat of the same pair its own \"label\".")
        seen[m["label"]] = m["index"]
        if re.search(r"[^A-Za-z0-9._+-]", m["label"]):
            raise ConfigError(f"{where}[{m['index']}]: label {m['label']!r} must be a bare filename "
                              f"(letters, digits, and . _ + - only)")

        # games cannot be TBD: task ids are cumulative sums of game counts.
        if not isinstance(m["games"], int) or isinstance(m["games"], bool) or m["games"] < 1:
            raise ConfigError(f"{where}[{m['index']}] ({m['label']}): games must be a positive integer "
                              f"(it determines the task-id layout, so it cannot be {TBD_SENTINEL}), "
                              f"got {m['games']!r}")
        if not _is_tbd(m["timeout"]) and (not isinstance(m["timeout"], int)
                                          or isinstance(m["timeout"], bool) or m["timeout"] < 1):
            raise ConfigError(f"{where}[{m['index']}] ({m['label']}): timeout must be a positive integer "
                              f"number of seconds (or {TBD_SENTINEL!r}), got {m['timeout']!r}")
        if m["games"] % 2 != 0:
            print(f"  [WARN] {where}[{m['index']}] ({m['label']}): {m['games']} games is odd, so the seat "
                  f"split is {m['games'] // 2}/{m['games'] - m['games'] // 2} rather than even.",
                  file=sys.stderr)


def load_config(path, seed_base_override=None):
    path = resolve_config_path(path)
    with open(path) as f:
        raw = json.load(f)

    defaults = raw.get("defaults") or {}
    if not raw.get("matchups"):
        raise ConfigError(f"{path}: config has no 'matchups'")

    matchups = _normalize_matchups(raw["matchups"], defaults, "matchups")
    _validate_matchups(matchups, "matchups")

    offset = 0
    for m in matchups:
        m["task_offset"] = offset
        offset += m["games"]

    calibration_raw = raw.get("calibration") or {}
    calib_defaults = dict(defaults)
    calib_defaults.update({k: v for k, v in calibration_raw.items() if k in ("games", "timeout", "env")})
    if calibration_raw.get("matchups"):
        calib_matchups = _normalize_matchups(calibration_raw["matchups"], calib_defaults, "calibration.matchups")
    else:
        # Without a calibration block, calibrate on the run's own matchups, which must
        # then carry no TBD values.
        calib_matchups = _normalize_matchups(raw["matchups"], calib_defaults, "calibration.matchups")
        for m in calib_matchups:
            m["games"] = calibration_raw.get("games", DEFAULT_CALIBRATION_GAMES)
    _validate_matchups(calib_matchups, "calibration.matchups")

    seed_base = seed_base_override if seed_base_override is not None else raw.get("seed_base")
    if seed_base is None:
        raise ConfigError(
            f"{path}: no 'seed_base' in the config and no --seed-base given. It must be fixed and "
            f"explicit -- a time-derived default would mean a retried task silently regenerates a "
            f"different game than originally planned.")

    allowed = raw.get("allowed_onnx_sha256") or [SHIPPED_ONNX_SHA256]
    if isinstance(allowed, str):
        allowed = [allowed]

    builtin = (raw.get("builtin_onnx_sha256") or SHIPPED_ONNX_SHA256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", builtin):
        raise ConfigError(
            f"{path}: 'builtin_onnx_sha256' must be a 64-character hex sha256, got {builtin!r}.")

    config = {
        "path": path,
        "name": raw.get("name") or os.path.splitext(os.path.basename(path))[0],
        "description": raw.get("description", ""),
        "seed_base": int(seed_base),
        "patrons": raw.get("patrons", DEFAULT_PATRONS),
        "allowed_onnx_sha256": [h.strip().lower() for h in allowed],
        "builtin_onnx_sha256": builtin,
        "bot_log": raw.get("bot_log", "parse"),
        "matchups": matchups,
        "calibration_matchups": calib_matchups,
        "total_tasks": offset,
    }

    # The loaded-model check reads the bot log, so a matchup that overrides
    # SOT_MODEL_PATH needs the log on. A failed override falls back to the built-in
    # model without any error.
    if config["bot_log"] not in ("parse", "keep"):
        overriding = [m["label"] for m in matchups
                      if m["env"].get("SOT_MODEL_PATH")]
        if overriding:
            raise ConfigError(
                f"{path}: bot_log is {config['bot_log']!r}, but these matchups set "
                f"SOT_MODEL_PATH: {', '.join(overriding)}.\n"
                f"  The harness verifies what each game actually loaded by reading the "
                f"bot log, and cannot do that with the log off.\n"
                f"  A model override that fails to load is SILENT -- the bot falls back "
                f"to GameRunner's built-in copy and plays a full, plausible-looking game "
                f"against the wrong model.\n"
                f"  Set \"bot_log\": \"parse\".")
    return config


# --------------------------------------------------------------------------
# Task resolution
# --------------------------------------------------------------------------

def resolve_task(config, task_id, matchups=None, subdir=""):
    matchups = matchups if matchups is not None else config["matchups"]
    total = sum(m["games"] for m in matchups)
    if not (0 <= task_id < total):
        raise ValueError(f"task_id {task_id} out of range [0, {total})")

    offset = 0
    for m in matchups:
        if task_id < offset + m["games"]:
            game_index = task_id - offset
            break
        offset += m["games"]
    else:  # pragma: no cover - guarded by the range check above
        raise ValueError(f"task_id {task_id} did not land in any matchup")

    swapped = game_index >= m["games"] // 2
    bot_a, bot_b = m["bot_a"], m["bot_b"]
    p1, p2 = (bot_b, bot_a) if swapped else (bot_a, bot_b)

    return {
        "task_id": task_id,
        "matchup_index": m["index"],
        "matchup": m["label"],
        "bot_a": bot_a,
        "bot_b": bot_b,
        "games_in_matchup": m["games"],
        "game_index": game_index,
        "swapped": swapped,
        "seed": config["seed_base"] + task_id,
        "p1": p1,
        "p2": p2,
        "timeout": m["timeout"],
        "env": m["env"],
        "subdir": subdir,
        "tbd_reason": tbd_reason(m),
    }


def matchup_dir(out_dir, task):
    base = os.path.join(out_dir, task["subdir"]) if task["subdir"] else out_dir
    return os.path.join(base, f"matchup_{task['matchup_index']:02d}_{task['matchup']}")


def result_path(out_dir, task):
    return os.path.join(matchup_dir(out_dir, task), f"task_{task['task_id']:04d}.json")


def load_existing_result(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    return data if data.get("completed") is True else None


def write_result_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def resolve_winner(p1_wins, p2_wins, swapped):
    """Returns "bot_a", "bot_b", or "draw", undoing the seat swap (see the
    module docstring)."""
    if p1_wins == 1:
        return "bot_b" if swapped else "bot_a"
    if p2_wins == 1:
        return "bot_a" if swapped else "bot_b"
    return "draw"


def resolve_end_reason(parsed):
    """Coarse bucket from the aggregate stats block, used when GameRunner's
    output has no GAME_END_REASON line."""
    for key in ("prestige40", "prestige80", "patron_favor", "turn_limit", "other"):
        if parsed.get(key, (0, 0))[0] == 1:
            return key
    return "unknown"


def classify(reason_detail):
    if reason_detail is None:
        return "unknown"
    return REASON_CATEGORY.get(reason_detail, "unknown")


def resolve_offender(winner):
    """Who caused a timeout/disqualification. The engine hands the win to the
    opponent in every such case except INTERNAL_ERROR (nobody wins), so the
    offender is whoever did NOT win."""
    if winner == "bot_a":
        return "bot_b"
    if winner == "bot_b":
        return "bot_a"
    return "unknown"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    """Sample standard deviation; 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def proc_timeout_for(task):
    """Process watchdog, scaled by the matchup's engine timeout. SLURM's --time
    is the real limit; this stops a hang from wedging a direct invocation."""
    timeout = task["timeout"] if isinstance(task["timeout"], int) else 0
    return max(1800, timeout * 180)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_loaded_models(bot_log_path):
    """bot name -> {"path", "size", "sha256"} for every bot that logged a
    successful model load. Last line wins if a bot somehow logged twice."""
    loaded = {}
    if not bot_log_path or not os.path.isfile(bot_log_path):
        return loaded
    try:
        with open(bot_log_path, errors="replace") as f:
            for line in f:
                m = LOADED_MODEL_PATTERN.search(line)
                if m:
                    loaded[m.group(1)] = {
                        "path": m.group(2),
                        "size": int(m.group(3)),
                        "sha256": m.group(4).lower(),
                    }
    except OSError:
        pass
    return loaded


def expected_model_sha(task, config, bot):
    """Expected (sha256, source) for this bot in this task, or (None, None).

    Only SOT_MODEL_PATH_BOTS read the override; the rest load GameRunner's
    built-in copy, even when the other side of the matchup is overridden.
    """
    if bot not in MODEL_LOADING_BOTS:
        return None, None
    override = task["env"].get("SOT_MODEL_PATH")
    if bot in SOT_MODEL_PATH_BOTS and override:
        if not os.path.isfile(override):
            return None, None      # check_task_model already aborted the task
        return sha256_file(override), f"SOT_MODEL_PATH ({override})"
    return config["builtin_onnx_sha256"], "GameRunner's built-in model"


def verify_loaded_models(task, config, loaded):
    """Check that each model-loading bot loaded its row's own model. Returns an
    error string, or None.

    allowed_onnx_sha256 checks only the file a row points at. A model that fails
    to load is silent (the bot plays on its heuristic), so the hash the bot logged
    is the only evidence of what it actually ran.
    """
    if config["bot_log"] not in ("parse", "keep"):
        return None

    problems = []
    for bot in (task["bot_a"], task["bot_b"]):
        want, source = expected_model_sha(task, config, bot)
        if want is None:
            continue
        got = loaded.get(bot)
        if got is None:
            problems.append(
                f"{bot} logged no successful model load.\n"
                f"    Expected {want[:12]}... from {source}.\n"
                f"    The bot logs 'FAILED to load ONNX model' and falls back to its "
                f"heuristic evaluator rather than crashing, so a missing line here is "
                f"a game played by the wrong evaluator -- not a missing log.")
        elif got["sha256"] != want:
            problems.append(
                f"{bot} loaded the WRONG model.\n"
                f"    expected: {want}  ({source})\n"
                f"    loaded  : {got['sha256']}\n"
                f"    from    : {got['path']}")
    if not problems:
        return None
    return ("loaded-model verification failed:\n  - " + "\n  - ".join(problems))


def check_task_model(task, allowed_hashes):
    """Check that a row's SOT_MODEL_PATH exists and is in allowed_onnx_sha256.
    Returns an error string, or None."""
    model_path = task["env"].get("SOT_MODEL_PATH")
    if not model_path:
        return None
    if not os.path.isfile(model_path):
        return (f"SOT_MODEL_PATH does not exist: {model_path}\n"
                f"  The bot would silently fall back to whichever model it can find and keep playing.")
    actual = sha256_file(model_path)
    if actual not in allowed_hashes:
        return (f"SOT_MODEL_PATH sha256 is not in the config's allowed_onnx_sha256 list.\n"
                f"  file    : {model_path}\n"
                f"  actual  : {actual}\n"
                f"  allowed : {', '.join(allowed_hashes)}")
    return None


# --------------------------------------------------------------------------
# Running one game
# --------------------------------------------------------------------------

def build_env(task, bot_log_path):
    """The task's environment: the ambient one without the SOT_* variables
    this harness manages, plus the matchup's env and the bot-log settings."""
    env = os.environ.copy()
    for key in ("SOT_LOG", "SOT_LOG_FILE", "SOT_DUMP_DIR", "SOT_ALPHA0",
                "SOT_TIME_SCALE", "SOT_BASELINE_TIME_SCALE", "SOT_MODEL_PATH"):
        env.pop(key, None)
    env.update(task["env"])
    if bot_log_path is not None:
        env["SOT_LOG"] = "1"
        env["SOT_LOG_FILE"] = bot_log_path
    return env


def parse_evals_per_turn(bot_log_path):
    """Mean evaluations per turn, keyed by agent class. BotLog does not know
    the seat, so in self-play both instances share a class; that case is
    averaged and flagged."""
    if not bot_log_path or not os.path.isfile(bot_log_path):
        return {}, []
    try:
        with open(bot_log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return {}, []

    per_class = {}
    for m in EVALS_PER_TURN_PATTERN.finditer(text):
        cls, total_evals, turns, mean = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        per_class.setdefault(cls, []).append({"total_evals": total_evals, "turns": turns, "mean": mean})

    ambiguous = [cls for cls, rows in per_class.items() if len(rows) > 1]
    summary = {}
    for cls, rows in per_class.items():
        summary[cls] = {
            "mean_evals_per_turn": sum(r["mean"] for r in rows) / len(rows),
            "total_evals": sum(r["total_evals"] for r in rows),
            "turns": sum(r["turns"] for r in rows),
            "instances": len(rows),
        }
    return summary, ambiguous


def run_task(binary, task, out_dir, config, proc_timeout_s):
    """Runs exactly one game for `task`. Returns (ok: bool, message: str,
    result: dict|None). Writes a result file on success only."""
    if task["tbd_reason"]:
        return False, f"ABORTED before running -- {task['tbd_reason']}", None
    err = check_task_model(task, config["allowed_onnx_sha256"])
    if err:
        return False, f"ABORTED before running -- {err}", None

    path = result_path(out_dir, task)
    bot_log_path = None
    if config["bot_log"] in ("parse", "keep"):
        log_dir = os.path.join(matchup_dir(out_dir, task), "botlogs")
        os.makedirs(log_dir, exist_ok=True)
        bot_log_path = os.path.join(log_dir, f"task_{task['task_id']:04d}.log")
        # BotLog appends; a retried task must not inherit the dead run's lines.
        if os.path.exists(bot_log_path):
            os.remove(bot_log_path)

    cmd = [binary, task["p1"], task["p2"], "--runs", "1",
           "--timeout", str(task["timeout"]),
           "--seed", str(task["seed"]), "--patrons", config["patrons"]]
    env = build_env(task, bot_log_path)

    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(binary), env=env,
                              capture_output=True, text=True, timeout=proc_timeout_s)
        stdout, stderr, returncode = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        return False, f"killed by {proc_timeout_s}s watchdog -- no data, task is retriable", None
    wall_clock_s = time.time() - start

    parsed = {}
    for key, pattern in LINE_PATTERNS.items():
        m = pattern.search(stdout)
        if m:
            parsed[key] = (int(m.group(1)), int(m.group(2)))

    if returncode != 0 or not REQUIRED_KEYS.issubset(parsed):
        head = "\n".join((stderr.strip().splitlines() or stdout.strip().splitlines() or ["(no output)"])[:20])
        return False, (f"exit code {returncode}, parsed {len(parsed)}/{len(REQUIRED_KEYS)} stats lines "
                       f"-- no data, task is retriable. stderr/stdout head:\n{head}"), None

    p1_wins, p2_wins = parsed["p1_wins"][0], parsed["p2_wins"][0]
    winner = resolve_winner(p1_wins, p2_wins, task["swapped"])
    end_reason = resolve_end_reason(parsed)

    detail_match = GAME_END_REASON_PATTERN.search(stdout)
    reason_detail = detail_match.group(1) if detail_match else None
    category = classify(reason_detail)
    # Older GameRunner builds print no GAME_END_REASON line; fall back to the
    # coarse bucket, which cannot separate timeouts from disqualifications.
    if reason_detail is None:
        category = {"prestige40": "clean", "prestige80": "clean", "patron_favor": "clean",
                    "turn_limit": "turn_limit"}.get(end_reason, "unknown")

    evals_per_turn, ambiguous = parse_evals_per_turn(bot_log_path)
    loaded_models = parse_loaded_models(bot_log_path)
    load_err = verify_loaded_models(task, config, loaded_models)

    # Keep the log when the check fails, as evidence; otherwise bot_log=parse
    # deletes it.
    if bot_log_path and config["bot_log"] == "parse" and load_err is None:
        try:
            os.remove(bot_log_path)
        except OSError:
            pass

    if load_err:
        return False, (f"ABORTED after running -- {load_err}\n"
                       f"  No result written. The game itself completed; it was played "
                       f"against the wrong evaluator, which is worse than not having run "
                       f"it.\n"
                       f"  Bot log kept at {bot_log_path}"), None

    result = {
        "task_id": task["task_id"],
        "matchup_index": task["matchup_index"],
        "matchup": task["matchup"],
        "config": config["name"],
        "bot_a": task["bot_a"],
        "bot_b": task["bot_b"],
        "game_index": task["game_index"],
        "games_in_matchup": task["games_in_matchup"],
        "swapped": task["swapped"],
        "seed": task["seed"],
        "p1": task["p1"],
        "p2": task["p2"],
        "timeout": task["timeout"],
        "env": task["env"],
        "completed": True,
        "clean": category == "clean",
        "winner": winner,
        "end_reason": end_reason,
        "end_reason_detail": reason_detail,
        "category": category,
        "offender": resolve_offender(winner) if category in ("timeout", "disqualification") else None,
        "evals_per_turn": evals_per_turn,
        "evals_per_turn_ambiguous": ambiguous,
        # The model each bot actually loaded, from its own log.
        "loaded_models": loaded_models,
        "wall_clock_s": round(wall_clock_s, 1),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_result_atomic(path, result)

    evals_str = ", ".join(f"{k}={v['mean_evals_per_turn']:.1f}/turn" for k, v in sorted(evals_per_turn.items()))
    return True, (f"winner={winner} reason={reason_detail or end_reason} category={category} "
                  f"wall_clock={wall_clock_s:.1f}s" + (f" [{evals_str}]" if evals_str else "")), result


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

def print_config_header(config):
    print(f"Config         : {config['name']}  ({config['path']})")
    if config["description"]:
        print(f"Description    : {config['description']}")
    print(f"Seed base      : {config['seed_base']}  (seed = seed_base + task_id)")
    print(f"Patrons        : {config['patrons']}")
    print(f"Bot log        : {config['bot_log']}")
    print(f"Allowed onnx   : {', '.join(h[:12] + '...' for h in config['allowed_onnx_sha256'])}"
          f"   (models a row may load via SOT_MODEL_PATH)")
    print(f"Built-in onnx  : {config['builtin_onnx_sha256'][:12]}..."
          f"   (GameRunner's own copy; checked separately)")
    if config["bot_log"] in ("parse", "keep"):
        print("Loaded-model   : verified per task against each row's own expected hash")
    else:
        print(f"Loaded-model   : NOT verified (bot_log={config['bot_log']!r}); no row "
              f"overrides SOT_MODEL_PATH, so every bot loads the built-in copy above")


def print_task_plan(config, task):
    print(f"task_id         : {task['task_id']} / {config['total_tasks']}")
    print(f"matchup         : [{task['matchup_index']}] {task['matchup']}  "
          f"({task['bot_a']} vs {task['bot_b']})")
    print(f"game_index      : {task['game_index']} / {task['games_in_matchup']} "
          f"({'swapped' if task['swapped'] else 'not swapped'})")
    print(f"seed            : {task['seed']}")
    print(f"engine --timeout: {task['timeout']}s")
    print(f"env             : {task['env'] or '(none)'}")
    print(f"GameRunner P1/P2: {task['p1']} vs {task['p2']}")


def print_full_plan(config, out_dir):
    print_config_header(config)
    print(f"Total tasks    : {config['total_tasks']}  (task_id 0..{config['total_tasks'] - 1})")
    print()
    header = f"{'idx':>3}  {'label':<40} {'bot_a vs bot_b':<46} {'to':>4} {'task_id range':<15} {'done':>6} {'pend':>6}  env"
    print(header)
    print("-" * len(header))
    total_done = 0
    for m in config["matchups"]:
        lo = m["task_offset"]
        hi = lo + m["games"] - 1
        done = 0
        for task_id in range(lo, hi + 1):
            task = resolve_task(config, task_id)
            if load_existing_result(result_path(out_dir, task)) is not None:
                done += 1
        total_done += done
        env_str = " ".join(f"{k}={v}" for k, v in sorted(m["env"].items())) or "-"
        pair = f"{m['bot_a']} vs {m['bot_b']}"
        flag = " (optional)" if m["optional"] else ""
        print(f"{m['index']:>3}  {m['label']:<40} {pair:<46} {str(m['timeout']):>4} "
              f"{f'{lo}-{hi}':<15} {done:>6} {m['games'] - done:>6}  {env_str}{flag}")
        if m["tbd"]:
            print(f"     *** NOT RUNNABLE: {TBD_SENTINEL} placeholders in {', '.join(m['tbd'])} -- "
                  f"every task in this range will refuse to run until they are filled in.")
        if m["note"]:
            print(f"     note: {m['note']}")
    print()
    print(f"Total: {total_done}/{config['total_tasks']} already done, "
          f"{config['total_tasks'] - total_done} pending.")
    print()
    print("Submit a single matchup by restricting the array to its task_id range, e.g.")
    first = config["matchups"][0]
    print(f"  sbatch --array={first['task_offset']}-{first['task_offset'] + first['games'] - 1}%32 "
          f"scripts/slurm_experiment.sh")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def run_calibration(config, binary, out_dir, games_override):
    """Run a few games per calibration matchup and report mean evaluations
    per turn per agent, for choosing an equal-effort SOT_TIME_SCALE. Results go
    in a calibration/ subdirectory and resume like a normal run."""
    matchups = [dict(m) for m in config["calibration_matchups"]]
    if games_override is not None:
        for m in matchups:
            m["games"] = games_override
    for i, m in enumerate(matchups):
        m["index"] = i
    total = sum(m["games"] for m in matchups)

    print_config_header(config)
    print(f"CALIBRATION    : {len(matchups)} matchup(s), {total} game(s) total, run sequentially")
    print()

    all_results = []
    for task_id in range(total):
        task = resolve_task(config, task_id, matchups=matchups, subdir="calibration")
        path = result_path(out_dir, task)
        existing = load_existing_result(path)
        if existing is not None:
            print(f"  [{task_id + 1}/{total}] {task['matchup']} game {task['game_index']}: "
                  f"already done (resumed)")
            all_results.append((task, existing))
            continue

        print(f"  [{task_id + 1}/{total}] {task['matchup']} game {task['game_index']} "
              f"(P1={task['p1']}, seed={task['seed']}, --timeout {task['timeout']}) ...", flush=True)
        ok, message, result = run_task(binary, task, out_dir, config, proc_timeout_for(task))
        print(f"        {message}")
        if ok:
            all_results.append((task, result))

    print()
    print("=" * 100)
    print("CALIBRATION SUMMARY -- mean evaluations per turn")
    print("=" * 100)
    print()
    print("Only agents that report throughput appear here. Stock SakkirinaSolo reports none;")
    print("SakkirinaScaled at SOT_BASELINE_TIME_SCALE=1.0 is timing-identical to it and does,")
    print("which is why the calibration matchup uses SakkirinaScaled rather than SakkirinaSolo.")
    print()

    by_matchup = {}
    for task, result in all_results:
        by_matchup.setdefault(task["matchup"], []).append(result)

    for label, results in by_matchup.items():
        # Reported per turn and per game. The two agree only if both agents played
        # the same number of turns, so the turn counts are printed too.
        per_turn, per_game, turns = {}, {}, {}
        for r in results:
            for cls, stats in (r.get("evals_per_turn") or {}).items():
                per_turn.setdefault(cls, []).append(stats["mean_evals_per_turn"])
                per_game.setdefault(cls, []).append(stats["total_evals"])
                turns.setdefault(cls, []).append(stats["turns"])
        print(f"[{label}]  n={len(results)} game(s)")
        if not per_turn:
            print("  no agent in this matchup reported evaluations per turn "
                  "(neither side is DeepSetsBotExp or SakkirinaScaled?)")
            print()
            continue
        env = results[0].get("env") or {}
        print(f"  env: {' '.join(f'{k}={v}' for k, v in sorted(env.items())) or '(none)'}")
        print()
        header = (f"  {'agent':<22}{'evals/turn':>14}{'sd':>11}{'min':>11}{'max':>11}"
                  f"{'evals/game':>13}{'turns':>8}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cls in sorted(per_turn):
            print(f"  {cls:<22}{mean(per_turn[cls]):>14.1f}{stdev(per_turn[cls]):>11.1f}"
                  f"{min(per_turn[cls]):>11.1f}{max(per_turn[cls]):>11.1f}"
                  f"{mean(per_game[cls]):>13.0f}{mean(turns[cls]):>8.1f}")

        exp_cls = next((c for c in per_turn if c.startswith("DeepSets")), None)
        base_cls = next((c for c in per_turn if c != exp_cls), None)
        if exp_cls and base_cls:
            exp_turn, base_turn = mean(per_turn[exp_cls]), mean(per_turn[base_cls])
            exp_game, base_game = mean(per_game[exp_cls]), mean(per_game[base_cls])
            # Per-game ratio per game, then averaged, rather than a ratio of
            # averages: a single long game would otherwise dominate.
            paired = [b / a for a, b in zip(per_game[exp_cls], per_game[base_cls]) if a > 0]
            print()
            if exp_turn > 0:
                print(f"  RATIO per turn : {base_cls} / {exp_cls} = {base_turn / exp_turn:.3f}x")
            if exp_game > 0:
                print(f"  RATIO per game : {base_cls} / {exp_cls} = {base_game / exp_game:.3f}x")
            if paired:
                print(f"  RATIO per game, computed per game then averaged: {mean(paired):.3f}x "
                      f"(sd {stdev(paired):.3f}, range {min(paired):.3f} .. {max(paired):.3f}, "
                      f"n={len(paired)})")

            alpha0 = env.get("SOT_ALPHA0")
            if alpha0 is None or float(alpha0) > 0.0:
                print()
                print(f"  *** WARNING: SOT_ALPHA0={alpha0 if alpha0 else '<unset, defaults to 0.7>'}.")
                print(f"      Above 0, {exp_cls} runs BOTH evaluators inside a single counted")
                print(f"      evaluation for every turn in the blend window, so its evals/turn is")
                print(f"      depressed by work the baseline never does. Counting the same EVENT is")
                print(f"      not the same as counting the same WORK. Calibrate equal effort at")
                print(f"      SOT_ALPHA0=0, where one counted evaluation is exactly one network")
                print(f"      forward pass.")

            if exp_turn > 0:
                print()
                print(f"  SUGGESTED SOT_TIME_SCALE for equal effort: "
                      f"{(base_turn / exp_turn) * float(env.get('SOT_TIME_SCALE', 1.0)):.4f}")
                print(f"  (That assumes evaluations per turn is linear in the time budget, which is")
                print(f"   approximately but not exactly true -- tree reuse and the rule-based fast")
                print(f"   paths do not scale with the clock. Put the suggestion into")
                print(f"   experiments/configs/equal_effort.json and re-run --calibrate against it to")
                print(f"   confirm the two numbers actually meet before spending 400 games on it.)")
        print()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Experiment config JSON -- a path, or a bare name in experiments/configs/")
    parser.add_argument("--binary", help="Path to the built GameRunner binary (required unless --dry-run)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Global task id. Required unless --dry-run is given with no --task-id "
                             "(which prints the whole plan instead) or --calibrate is given.")
    parser.add_argument("--out-dir", required=True, help="Output directory for result JSON files")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="Overrides the config's seed_base. Must stay fixed across the whole array "
                             "and any resubmission, or retried tasks would silently regenerate different "
                             "games than originally planned.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan (whole plan if --task-id is omitted, one task's plan if "
                             "given); run nothing.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the config's calibration matchups sequentially and report mean "
                             "evaluations per turn per agent. Does not touch the main results.")
    parser.add_argument("--calibration-games", type=int, default=None,
                        help=f"Override the calibration game count (config default, else "
                             f"{DEFAULT_CALIBRATION_GAMES}).")
    parser.add_argument("--allow-onnx-sha256", action="append", default=[],
                        help="Extra allowed onnx sha256, repeatable. Adds to the config's list.")
    parser.add_argument("--configuration", default="unknown", help="For traceability only")
    parser.add_argument("--onnx-sha256", default=None, help="For traceability only")
    parser.add_argument("--bots-dll-sha256", default=None, help="For traceability only")
    args = parser.parse_args()

    try:
        config = load_config(args.config, args.seed_base)
    except (ConfigError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: {e}")
    config["allowed_onnx_sha256"] += [h.strip().lower() for h in args.allow_onnx_sha256]

    out_dir = os.path.abspath(args.out_dir)

    if args.dry_run:
        if args.task_id is None:
            print_full_plan(config, out_dir)
        else:
            try:
                print_config_header(config)
                print()
                print_task_plan(config, resolve_task(config, args.task_id))
            except ValueError as e:
                sys.exit(f"ERROR: {e}")
        return

    if not args.binary:
        sys.exit("ERROR: --binary is required.")
    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        sys.exit(f"ERROR: binary not found or not executable: {binary}")

    if args.calibrate:
        run_calibration(config, binary, out_dir, args.calibration_games)
        return

    if args.task_id is None:
        sys.exit("ERROR: --task-id is required (except for a plan-only --dry-run, or --calibrate).")

    try:
        task = resolve_task(config, args.task_id)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    path = result_path(out_dir, task)

    print(f"Task {task['task_id']}/{config['total_tasks']}: [{task['matchup_index']}] {task['matchup']} "
          f"({task['bot_a']} vs {task['bot_b']}), game {task['game_index']}/{task['games_in_matchup']}, "
          f"swapped={task['swapped']}, seed={task['seed']}, --timeout {task['timeout']}")
    if task["env"]:
        print(f"Env            : {' '.join(f'{k}={v}' for k, v in sorted(task['env'].items()))}")
    if args.configuration != "unknown":
        print(f"Configuration  : {args.configuration}")
    if args.bots_dll_sha256:
        print(f"Bots.dll sha256: {args.bots_dll_sha256}")
    if args.onnx_sha256:
        print(f"Onnx sha256    : {args.onnx_sha256}")

    existing = load_existing_result(path)
    if existing is not None:
        print(f"Result already exists at {path} -- skipping (winner={existing.get('winner')}, "
              f"category={existing.get('category')}). Delete the file to force a re-run.")
        return

    ok, message, _ = run_task(binary, task, out_dir, config, proc_timeout_for(task))
    print(message)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
