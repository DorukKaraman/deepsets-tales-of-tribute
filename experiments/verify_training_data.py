"""
SUPERSEDED: this targets the PRE-SCHEMA-V2 data format (the single
Train_/Val_Sakkirina.jsonl.gz pair, 101/17-dimensional features). Current
generated data is verified with tools/verify_generated_data.py instead. Kept
for completeness, since it is the check that was actually run against the
older dataset.

Read-only verification of the Sakkirina training data, answering the
"Verify first" section of the pre-rewrite retrain checklist.

Modifies nothing. Streams Train_Sakkirina.jsonl.gz and Val_Sakkirina.jsonl.gz
line by line -- never loads either file into memory. The only in-memory state
kept across the whole pass is a set of StateId strings per file (needed for
the duplicate check and the train/val leakage check) plus small counters/lists
whose cardinality is bounded by the number of distinct patrons, decks, cards,
or inferred games, not by the number of records. On a very large train file
that StateId set is itself a real memory cost (millions of GUID-sized
strings); use --limit for a quick pass if that's a problem.

Usage:
    python3 experiments/verify_training_data.py                # full files
    python3 experiments/verify_training_data.py --limit 500000  # quick pass
    python3 experiments/verify_training_data.py --json-summary  # also write a
                                                           # summary JSON
"""
import argparse
import gzip
import json
import os
import statistics
import sys
import time
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
OUT_DIR = os.path.join(SCRIPT_DIR, "out")

DEFAULT_TRAIN_PATH = os.path.join(REPO_ROOT, "GameRunner", "Train_Sakkirina.jsonl.gz")
DEFAULT_VAL_PATH = os.path.join(REPO_ROOT, "GameRunner", "Val_Sakkirina.jsonl.gz")
JSON_SUMMARY_PATH = os.path.join(OUT_DIR, "verify_training_data_summary.json")

sys.path.insert(0, TRAINING_DIR)
from card_db import CARD_EFFECTS  # noqa: E402

PROGRESS_INTERVAL = 100_000

# Mirrors StateParser.extract_global_context exactly (PATRON_ORDER and
# favor_map are local to that function, not importable) -- keep in sync if
# that file changes. This is exactly the 9-entry list item 5 checks against.
PATRON_ORDER = [
    "TREASURY", "ANSEI", "DUKE_OF_CROWS", "RAJHIN", "PSIJIC",
    "ORGNUM", "HLAALU", "PELIN", "RED_EAGLE",
]

# PlayerEnum, per the task's own KNOWN SCHEMA section.
PLAYER1, PLAYER2, NO_PLAYER_SELECTED = 0, 1, 2


def progress(msg):
    print(msg, file=sys.stderr, flush=True)


def pick_effect_probe_ids(n=5):
    """CommonIds with the most non-zero entries in their CARD_EFFECTS vector --
    i.e. cards "known to have non-trivial effects" per item 8, chosen from the
    actual db rather than hand-picked, so this stays correct if card_db.py
    changes."""
    scored = [(cid, sum(1 for x in vec if x != 0)) for cid, vec in CARD_EFFECTS.items()]
    scored.sort(key=lambda t: -t[1])
    return [cid for cid, _ in scored[:n] if _ > 0]


def iter_all_cards(state):
    """Every card object reachable from one state, across every location:
    tavern, our hand/played/cooldown/draw/known-upcoming-draws, our agents
    (unwrapped to their RepresentingCard), and the enemy's hand+draw/played/
    cooldown/agents. This is the single shared traversal behind items 6, 7,
    and 8 -- deck ids, CommonId coverage, and the Effects field all come from
    the same set of cards."""
    cp = state.get("CurrentPlayer") or {}
    ep = state.get("EnemyPlayer") or {}

    for card in state.get("TavernAvailableCards") or []:
        yield card

    for key in ("Hand", "Played", "CooldownPile", "DrawPile", "KnownUpcomingDraws"):
        for card in cp.get(key) or []:
            yield card
    for agent in cp.get("Agents") or []:
        rc = agent.get("RepresentingCard")
        if rc is not None:
            yield rc

    for key in ("HandAndDraw", "Played", "CooldownPile"):
        for card in ep.get(key) or []:
            yield card
    for agent in ep.get("Agents") or []:
        rc = agent.get("RepresentingCard")
        if rc is not None:
            yield rc


def effects_is_nonempty(effects):
    if not effects:
        return False
    return any(e not in (None, {}) for e in effects)


def process_file(path, label, limit, compare_against_ids, effect_probe_ids):
    """One streaming pass over one JSONL.gz file. Computes items 1, 2, 4, 5, 6,
    7, 8, 9 for this file; if compare_against_ids is given (only for the train
    pass), also computes item 3 (StateId overlap with that set). Returns
    (stats_dict, this_file's_state_id_set)."""
    state_ids = set()
    missing_state_id_count = 0
    total_records = 0
    outcome_1_count = 0
    leakage_count = 0

    # Item 2: game boundaries via max(prestige) non-decreasing within a game.
    prev_max_prestige = None
    current_game_len = 0
    current_game_outcomes = set()
    game_lengths = []
    games_with_inconsistent_outcome = 0

    # Item 4: patron sign bug scope.
    player2_current_count = 0
    player_id_raw_values = Counter()

    # Item 5: patron coverage.
    patron_key_counter = Counter()
    patrons_array_counter = Counter()

    # Item 6: deck id range.
    deck_counter = Counter()

    # Item 7: card coverage.
    common_id_counter = Counter()

    # Item 8: effects field.
    effects_ever_nonempty = False
    effects_nonempty_examples = []
    effect_probe_examples = {}  # common_id -> (name, raw effects)

    # Item 9: enemy fields.
    enemy_handdraw_present_count = 0
    enemy_handdraw_len_sum = 0
    enemy_played_nonempty_count = 0

    def finalize_game():
        if current_game_len > 0:
            game_lengths.append(current_game_len)
            if len(current_game_outcomes) > 1:
                nonlocal games_with_inconsistent_outcome
                games_with_inconsistent_outcome += 1

    start_time = time.time()
    line_num = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if limit is not None and line_num > limit:
                line_num -= 1
                break
            if line_num % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                progress(f"  [{label}] {line_num:,} lines ({elapsed:.0f}s elapsed)")

            record = json.loads(line)
            outcome = record.get("outcome")
            state = record["data"]["state"]
            total_records += 1
            if outcome == 1:
                outcome_1_count += 1

            # --- Item 1: StateId uniqueness ---
            sid = state.get("StateId")
            if sid is None:
                missing_state_id_count += 1
            else:
                state_ids.add(sid)
                # --- Item 3: train/val leakage (train pass only) ---
                if compare_against_ids is not None and sid in compare_against_ids:
                    leakage_count += 1

            cp = state.get("CurrentPlayer") or {}
            ep = state.get("EnemyPlayer") or {}

            # --- Item 2: game boundaries + outcome-constancy sanity check ---
            cur_prestige = cp.get("Prestige") or 0
            enemy_prestige = ep.get("Prestige") or 0
            max_prestige = max(cur_prestige, enemy_prestige)
            if prev_max_prestige is not None and max_prestige < prev_max_prestige:
                finalize_game()
                current_game_len = 0
                current_game_outcomes = set()
            prev_max_prestige = max_prestige
            current_game_len += 1
            current_game_outcomes.add(outcome)

            # --- Item 4: patron sign bug scope ---
            player_id = cp.get("PlayerID")
            player_id_raw_values[player_id] += 1
            if player_id == PLAYER2:
                player2_current_count += 1

            # --- Item 5: patron coverage ---
            patron_all = ((state.get("PatronStates") or {}).get("All")) or {}
            for k in patron_all.keys():
                patron_key_counter[k] += 1
            for p in state.get("Patrons") or []:
                patrons_array_counter[p] += 1

            # --- Items 6/7/8: card-level stats, one shared traversal ---
            for card in iter_all_cards(state):
                if card is None:
                    continue
                deck_counter[card.get("Deck")] += 1
                cid = card.get("CommonId")
                common_id_counter[cid] += 1

                effects = card.get("Effects")
                if effects_is_nonempty(effects):
                    effects_ever_nonempty = True
                    if len(effects_nonempty_examples) < 5:
                        effects_nonempty_examples.append((card.get("Name"), cid, effects))

                if effect_probe_ids and cid in effect_probe_ids and cid not in effect_probe_examples:
                    effect_probe_examples[cid] = (card.get("Name"), effects)

            # --- Item 9: enemy fields ---
            hand_and_draw = ep.get("HandAndDraw")
            if hand_and_draw is not None:
                enemy_handdraw_present_count += 1
                enemy_handdraw_len_sum += len(hand_and_draw)
            if ep.get("Played"):
                enemy_played_nonempty_count += 1

    finalize_game()  # flush whatever game was in progress at end of file

    elapsed = time.time() - start_time
    progress(f"  [{label}] done: {total_records:,} lines in {elapsed:.0f}s")

    stats = {
        "label": label,
        "path": path,
        "limited": limit is not None,
        "total_records": total_records,
        "missing_state_id_count": missing_state_id_count,
        "distinct_state_ids": len(state_ids),
        "outcome_1_count": outcome_1_count,
        "outcome_base_rate": (outcome_1_count / total_records) if total_records else None,
        "leakage_count": leakage_count,
        "game_lengths": game_lengths,
        "games_with_inconsistent_outcome": games_with_inconsistent_outcome,
        "player2_current_count": player2_current_count,
        "player_id_raw_values": dict(player_id_raw_values),
        "patron_key_counter": dict(patron_key_counter),
        "patrons_array_counter": dict(patrons_array_counter),
        "deck_counter": dict(deck_counter),
        "common_id_counter": dict(common_id_counter),
        "effects_ever_nonempty": effects_ever_nonempty,
        "effects_nonempty_examples": effects_nonempty_examples,
        "effect_probe_examples": effect_probe_examples,
        "enemy_handdraw_present_count": enemy_handdraw_present_count,
        "enemy_handdraw_len_sum": enemy_handdraw_len_sum,
        "enemy_played_nonempty_count": enemy_played_nonempty_count,
    }
    return stats, state_ids


def fmt_pct(n, d):
    return f"{(100.0 * n / d):.2f}%" if d else "N/A"


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_report(train_stats, val_stats, effect_probe_ids):
    print_section("1. BASIC COUNTS")
    for s in (val_stats, train_stats):
        print(f"  [{s['label']}] path={s['path']}" + (" (LIMITED run)" if s["limited"] else ""))
        print(f"  [{s['label']}] records: {s['total_records']:,}")
        print(f"  [{s['label']}] outcome==1: {s['outcome_1_count']:,} "
              f"(base rate {fmt_pct(s['outcome_1_count'], s['total_records'])})")
        print(f"  [{s['label']}] distinct StateId: {s['distinct_state_ids']:,} "
              f"of {s['total_records']:,} records "
              f"({s['total_records'] - s['distinct_state_ids']:,} duplicate StateId occurrences)")
        if s["missing_state_id_count"]:
            print(f"  [{s['label']}] WARNING: {s['missing_state_id_count']:,} records had no StateId at all")
        print()

    print_section("2. GAME BOUNDARIES (heuristic: drop in max(prestige) => new game)")
    for s in (val_stats, train_stats):
        lengths = s["game_lengths"]
        n_games = len(lengths)
        print(f"  [{s['label']}] estimated games: {n_games:,}")
        if n_games:
            print(f"  [{s['label']}] states/game: mean={statistics.mean(lengths):.1f} "
                  f"median={statistics.median(lengths):.1f} "
                  f"min={min(lengths)} max={max(lengths)}")
            inconsistent = s["games_with_inconsistent_outcome"]
            print(f"  [{s['label']}] games with non-constant outcome: {inconsistent:,} / {n_games:,} "
                  f"({fmt_pct(inconsistent, n_games)})", end="")
            if n_games and inconsistent / n_games > 0.02:
                print("  <-- high: the boundary heuristic may be wrong, treat games/lengths above as suspect")
            else:
                print()
        print()

    print_section("3. TRAIN/VAL STATE-ID LEAKAGE")
    n = train_stats["total_records"]
    overlap = train_stats["leakage_count"]
    print(f"  val distinct StateIds: {val_stats['distinct_state_ids']:,}")
    print(f"  train records checked against that set: {n:,}")
    print(f"  train records whose StateId is ALSO in val: {overlap:,} ({fmt_pct(overlap, n)})")
    if overlap == 0:
        print("  No overlap found: consistent with a split by GAME, not by state.")
    else:
        print("  Overlap found: split included the same StateId in both train and val.")
        print("  Validation accuracy measured against this split is inflated by leakage.")

    print_section("4. PATRON SIGN BUG SCOPE (StateParser.py favor_map hardcodes PLAYER1==us)")
    for s in (val_stats, train_stats):
        n = s["total_records"]
        p2 = s["player2_current_count"]
        print(f"  [{s['label']}] CurrentPlayer.PlayerID raw values seen: {s['player_id_raw_values']}")
        print(f"  [{s['label']}] CurrentPlayer.PlayerID == PLAYER2 ({PLAYER2}): {p2:,} / {n:,} "
              f"({fmt_pct(p2, n)}) records with all 9 patron features sign-inverted")
        print()

    print_section("5. PATRON COVERAGE")
    for s in (val_stats, train_stats):
        print(f"  [{s['label']}] PatronStates.All keys seen (count = records containing that key):")
        for key, cnt in sorted(s["patron_key_counter"].items(), key=lambda kv: -kv[1]):
            unknown = "  <-- NOT in PATRON_ORDER" if key not in PATRON_ORDER else ""
            print(f"    {key:20s} {cnt:>10,}{unknown}")
        unseen_keys = set(s["patron_key_counter"]) - set(PATRON_ORDER)
        if unseen_keys:
            print(f"  [{s['label']}] keys not in PATRON_ORDER (exact spelling as serialized): {sorted(unseen_keys)}")
        else:
            print(f"  [{s['label']}] no PatronStates.All key falls outside PATRON_ORDER in this file")
        print(f"  [{s['label']}] Patrons array values (patron id -> record count):")
        for pid, cnt in sorted(s["patrons_array_counter"].items()):
            print(f"    {pid:>4}  {cnt:>10,}")
        print()

    print_section("6. DECK ID RANGE (StateParser.py: `if 0 <= deck <= 8` else silently zeroed)")
    for s in (val_stats, train_stats):
        decks = s["deck_counter"]
        if not decks:
            print(f"  [{s['label']}] no cards observed")
            continue
        max_deck = max(decks)
        print(f"  [{s['label']}] distinct Deck values: {sorted(decks)} (max={max_deck})")
        out_of_range = sum(cnt for d, cnt in decks.items() if not (isinstance(d, int) and 0 <= d <= 8))
        total_cards = sum(decks.values())
        print(f"  [{s['label']}] card instances with Deck outside [0,8]: {out_of_range:,} / {total_cards:,} "
              f"({fmt_pct(out_of_range, total_cards)})")
        for d, cnt in sorted(decks.items(), key=lambda kv: (kv[0] is None, kv[0])):
            print(f"    Deck={d!r:>6}  {cnt:>10,}")
        print()

    print_section("7. CARD COVERAGE (CommonId vs ValueNetwork/card_db.py CARD_EFFECTS)")
    db_keys = set(CARD_EFFECTS.keys())
    for s in (val_stats, train_stats):
        seen = s["common_id_counter"]
        seen_ids = set(seen.keys())
        missing_from_db = seen_ids - db_keys
        unused_in_db = db_keys - seen_ids
        print(f"  [{s['label']}] distinct CommonIds seen: {len(seen_ids):,} "
              f"(card_db.py has {len(db_keys):,})")
        if missing_from_db:
            print(f"  [{s['label']}] CommonIds in data but MISSING from card_db (zero effect vector):")
            for cid in sorted(missing_from_db, key=lambda c: (c is None, c)):
                print(f"    CommonId={cid!r}  seen {seen[cid]:,} times")
        else:
            print(f"  [{s['label']}] every CommonId seen in this file exists in card_db")
        if unused_in_db:
            print(f"  [{s['label']}] card_db keys never seen in this file ({len(unused_in_db)}): "
                  f"{sorted(unused_in_db)}")
        else:
            print(f"  [{s['label']}] every card_db key is exercised at least once in this file")
        print()

    print_section("8. EFFECTS FIELD (independent cross-check on card_db.py, if it has content)")
    for s in (val_stats, train_stats):
        print(f"  [{s['label']}] probe CommonIds (most non-zero card_db entries): {effect_probe_ids}")
        for cid in effect_probe_ids:
            example = s["effect_probe_examples"].get(cid)
            if example is None:
                print(f"    CommonId={cid}: not observed in this file")
            else:
                name, effects = example
                print(f"    CommonId={cid} ({name}): raw Effects = {effects}")
        if s["effects_ever_nonempty"]:
            print(f"  [{s['label']}] Effects DOES serialize real content -- examples:")
            for name, cid, effects in s["effects_nonempty_examples"]:
                print(f"    {name} (CommonId={cid}): {effects}")
            print(f"  [{s['label']}] ==> this is an independent cross-check on card_db.py; use it.")
        else:
            print(f"  [{s['label']}] Effects was empty ({{}}/null only) on every card instance observed "
                  f"-- no independent cross-check available from this field.")
        print()

    print_section("9. ENEMY_UNSEEN AVAILABILITY (EnemyPlayer.HandAndDraw / EnemyPlayer.Played)")
    for s in (val_stats, train_stats):
        n = s["total_records"]
        present = s["enemy_handdraw_present_count"]
        mean_len = (s["enemy_handdraw_len_sum"] / present) if present else None
        print(f"  [{s['label']}] EnemyPlayer.HandAndDraw present: {present:,} / {n:,} ({fmt_pct(present, n)})")
        if mean_len is not None:
            print(f"  [{s['label']}] EnemyPlayer.HandAndDraw mean length when present: {mean_len:.2f}")
        played_nonempty = s["enemy_played_nonempty_count"]
        print(f"  [{s['label']}] EnemyPlayer.Played non-empty: {played_nonempty:,} / {n:,} "
              f"({fmt_pct(played_nonempty, n)})", end="")
        if n and played_nonempty / n < 0.02:
            print("  <-- rarely non-empty: this location slot looks like dead weight")
        else:
            print()
        print()


def build_json_summary(train_stats, val_stats):
    def trim(s):
        s = dict(s)
        s["game_lengths"] = {
            "count": len(s["game_lengths"]),
            "mean": statistics.mean(s["game_lengths"]) if s["game_lengths"] else None,
            "median": statistics.median(s["game_lengths"]) if s["game_lengths"] else None,
            "min": min(s["game_lengths"]) if s["game_lengths"] else None,
            "max": max(s["game_lengths"]) if s["game_lengths"] else None,
        }
        s["effect_probe_examples"] = {str(k): v for k, v in s["effect_probe_examples"].items()}
        s["patron_key_counter"] = {str(k): v for k, v in s["patron_key_counter"].items()}
        s["patrons_array_counter"] = {str(k): v for k, v in s["patrons_array_counter"].items()}
        s["deck_counter"] = {str(k): v for k, v in s["deck_counter"].items()}
        s["common_id_counter"] = {str(k): v for k, v in s["common_id_counter"].items()}
        s["player_id_raw_values"] = {str(k): v for k, v in s["player_id_raw_values"].items()}
        return s

    return {"train": trim(train_stats), "val": trim(val_stats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=DEFAULT_TRAIN_PATH, help="Path to Train_Sakkirina.jsonl.gz")
    parser.add_argument("--val", default=DEFAULT_VAL_PATH, help="Path to Val_Sakkirina.jsonl.gz")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process at most this many lines per file (quick pass). Default: full files.")
    parser.add_argument("--json-summary", action="store_true",
                         help=f"Also write a summary JSON to {JSON_SUMMARY_PATH}")
    args = parser.parse_args()

    if not os.path.isfile(args.val):
        sys.exit(f"ERROR: val file not found: {args.val}")
    if not os.path.isfile(args.train):
        sys.exit(f"ERROR: train file not found: {args.train}")

    effect_probe_ids = pick_effect_probe_ids(5)

    # Val must be read first in full (or up to --limit) so its StateId set
    # exists before the train pass checks for leakage against it.
    progress("Pass 1/2: reading val file to build the StateId reference set...")
    val_stats, val_ids = process_file(args.val, "val", args.limit,
                                       compare_against_ids=None, effect_probe_ids=effect_probe_ids)

    progress("Pass 2/2: reading train file (checking StateId leakage against val)...")
    train_stats, _train_ids = process_file(args.train, "train", args.limit,
                                            compare_against_ids=val_ids, effect_probe_ids=effect_probe_ids)

    print_report(train_stats, val_stats, effect_probe_ids)

    if args.json_summary:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(JSON_SUMMARY_PATH, "w") as f:
            json.dump(build_json_summary(train_stats, val_stats), f, indent=2, default=str)
        print()
        print(f"Wrote summary JSON to {JSON_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
