"""
Read-only pre-cluster gate for tools/generate_data.py output. Modifies
nothing. Point it at a --data-dir (the directory generate_data.py wrote
job_NNNN/ subdirectories into) and it streams every *.jsonl.gz shard found
under it, recursively.

Two-pass-in-one-streaming-pass design: item 7 (opening diversity) needs each
game's records from BOTH its bot1 and bot2 shards merged into true
chronological order, but the two shards are separate files and may be visited
in either order. Rather than a real two-pass read, each record is reduced
immediately to a small tuple (player, completed_actions_len, action,
outcome) and appended to a per-game_id list; only these tuples are kept in
memory, not full states, so this scales to a much larger run than 50 games
without needing --limit. CurrentPlayer.PlayerID and Deck-id counts are
accumulated as running Counters, not retained per-record.

"cards acquired" (item 7) is read as CommandEnum.BUY_CARD moves specifically
-- Move.CommandEnum has no ACQUIRE_CARD variant; ACQUIRE_CARD only exists as
a CompletedActionType logged when a card EFFECT (e.g. "Acquire 4") pulls a
card for free, which is not a move a bot ever explicitly chooses and isn't
cleanly nameable from the action string. BUY_CARD is the only directly-logged
top-level "a card entered someone's deck" event, and matches the methodology
already used earlier in this session for opening-purchase analysis.

Usage:
    python3 tools/verify_generated_data.py /tmp/gen50
    python3 tools/verify_generated_data.py /tmp/gen50 --limit 5000  # per-shard cap
"""
import argparse
import glob
import gzip
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

PROGRESS_INTERVAL = 50_000

# CompletedActionType.END_TURN's ordinal in Engine/src/Board/CompletedAction.cs
# (last entry in that enum). A state is "start of turn" under the exact same
# rule SakkirinaGen.Play() itself uses to detect a fresh turn: no completed
# actions yet, or the most recent one was an END_TURN.
END_TURN_TYPE = 26

COMPETITION_PATRONS = {"ANSEI", "DUKE_OF_CROWS", "RAJHIN", "ORGNUM", "PELIN", "SAINT_ALESSIA"}
ALLOWED_PATRON_KEYS = COMPETITION_PATRONS | {"TREASURY"}
ALL_15_DRAFT_COMBOS = set(combinations(sorted(COMPETITION_PATRONS), 4))


def progress(msg):
    print(msg, file=sys.stderr, flush=True)


def find_shards(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.jsonl.gz"), recursive=True))


SHARD_ROLE_RE = re.compile(r"_bot([12])\.jsonl\.gz$")


def shard_role(path):
    m = SHARD_ROLE_RE.search(os.path.basename(path))
    return f"bot{m.group(1)}" if m else "unknown"


def is_start_of_turn(state):
    completed = state.get("CompletedActions") or []
    return len(completed) == 0 or completed[-1].get("Type") == END_TURN_TYPE


def iter_all_cards(state):
    """Every card object reachable from one state -- same traversal as
    tools/verify_training_data.py's iter_all_cards, for the Deck-id check."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="Directory containing job_NNNN/*.jsonl.gz shards "
                                          "(generate_data.py's --out-dir)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max records to read per shard (default: no limit)")
    args = parser.parse_args()

    shards = find_shards(args.data_dir)
    if not shards:
        sys.exit(f"ERROR: no *.jsonl.gz shards found under {args.data_dir}")

    # --- accumulators -----------------------------------------------------
    total_records = 0
    records_per_shard = {}
    records_per_role = Counter()          # "bot1" / "bot2" / "unknown"
    records_per_player_field = Counter()  # 0 / 1
    outcome_1_count = 0
    start_of_turn_count = 0

    current_player_id_counter = Counter()
    player_field_vs_state_mismatches = []  # (shard, game_id, player_field, state_playerid)

    unexpected_patron_keys = Counter()
    unexpected_patron_examples = defaultdict(list)
    game_patron_combo = {}   # game_id -> sorted tuple of 4 non-treasury patron names

    deck_counter = Counter()

    # game_id -> list of (player, completed_actions_len, action, outcome)
    game_records = defaultdict(list)

    start_time = time.time()
    for shard_path in shards:
        role = shard_role(shard_path)
        n_this_shard = 0
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if args.limit is not None and line_num > args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total_records += 1
                n_this_shard += 1
                records_per_role[role] += 1

                outcome = rec["outcome"]
                game_id = rec["game_id"]
                player = rec["player"]
                state = rec["data"]["state"]
                action = rec["data"]["action"]

                records_per_player_field[player] += 1
                if outcome == 1:
                    outcome_1_count += 1
                if is_start_of_turn(state):
                    start_of_turn_count += 1

                cp_id = state.get("CurrentPlayer", {}).get("PlayerID")
                current_player_id_counter[cp_id] += 1
                if cp_id != player:
                    player_field_vs_state_mismatches.append((shard_path, game_id, player, cp_id))

                patron_keys = set((state.get("PatronStates") or {}).get("All", {}).keys())
                bad_keys = patron_keys - ALLOWED_PATRON_KEYS
                for k in bad_keys:
                    unexpected_patron_keys[k] += 1
                    if len(unexpected_patron_examples[k]) < 3:
                        unexpected_patron_examples[k].append((shard_path, game_id))
                if game_id not in game_patron_combo:
                    combo = tuple(sorted(patron_keys & COMPETITION_PATRONS))
                    if len(combo) == 4:
                        game_patron_combo[game_id] = combo

                for card in iter_all_cards(state):
                    deck_counter[card.get("Deck")] += 1

                completed_len = len(state.get("CompletedActions") or [])
                game_records[game_id].append((player, completed_len, action, outcome))

                if total_records % PROGRESS_INTERVAL == 0:
                    progress(f"  {total_records:,} records ({time.time() - start_time:.0f}s elapsed)")

        records_per_shard[shard_path] = n_this_shard

    progress(f"Done reading {total_records:,} records from {len(shards)} shards "
             f"in {time.time() - start_time:.0f}s.")
    print()

    # =======================================================================
    print("=" * 78)
    print("1. RECORDS: TOTAL, PER SHARD, PER PLAYER FIELD")
    print("=" * 78)
    print(f"  total records: {total_records:,}")
    print(f"  shards: {len(shards)}")
    for path in shards:
        rel = os.path.relpath(path, args.data_dir)
        print(f"    {rel:70s} {records_per_shard[path]:8,d}  ({shard_role(path)})")
    print()
    print(f"  by shard role : bot1={records_per_role.get('bot1', 0):,}  bot2={records_per_role.get('bot2', 0):,}")
    b1, b2 = records_per_role.get("bot1", 0), records_per_role.get("bot2", 0)
    if b1 and b2:
        gap_pct = abs(b1 - b2) / max(b1, b2) * 100
        verdict = "OK" if gap_pct <= 5 else "WARNING: gap exceeds 5% -- one perspective may be under-logged"
        print(f"  bot1 vs bot2 gap: {gap_pct:.2f}%  [{verdict}]")
    print()
    print(f"  by player field: player=0: {records_per_player_field.get(0, 0):,}  "
          f"player=1: {records_per_player_field.get(1, 0):,}")
    p0, p1 = records_per_player_field.get(0, 0), records_per_player_field.get(1, 0)
    if p0 and p1:
        gap_pct = abs(p0 - p1) / max(p0, p1) * 100
        verdict = "OK" if gap_pct <= 5 else "WARNING: gap exceeds 5%"
        print(f"  player=0 vs player=1 gap: {gap_pct:.2f}%  [{verdict}]")
    print()

    # =======================================================================
    print("=" * 78)
    print("2. OUTCOME BASE RATE")
    print("=" * 78)
    base_rate = outcome_1_count / total_records if total_records else 0.0
    verdict = "OK, near 0.50" if 0.45 <= base_rate <= 0.55 else "WARNING: not near 0.50 -- the two-perspective fix may not be working"
    print(f"  outcome==1: {outcome_1_count:,} / {total_records:,} = {base_rate:.4f}  [{verdict}]")
    print()

    # =======================================================================
    print("=" * 78)
    print("3. PER-GAME OUTCOME CONSISTENCY + COMPLEMENTARITY")
    print("=" * 78)
    inconsistent_within_player = []   # game_id, player, set of outcomes seen
    non_complementary = []            # game_id, {player: outcome}
    single_perspective_games = []     # game_id, which player(s) present

    for game_id, records in game_records.items():
        by_player_outcomes = defaultdict(set)
        for player, _, _, outcome in records:
            by_player_outcomes[player].add(outcome)

        for player, outcomes in by_player_outcomes.items():
            if len(outcomes) > 1:
                inconsistent_within_player.append((game_id, player, sorted(outcomes)))

        players_present = sorted(by_player_outcomes)
        if players_present == [0, 1]:
            o0 = next(iter(by_player_outcomes[0]))
            o1 = next(iter(by_player_outcomes[1]))
            if len(by_player_outcomes[0]) == 1 and len(by_player_outcomes[1]) == 1 and o0 == o1:
                non_complementary.append((game_id, {0: o0, 1: o1}))
        else:
            single_perspective_games.append((game_id, players_present))

    print(f"  games checked: {len(game_records):,}")
    print(f"  outcome inconsistent WITHIN a (game_id, player): {len(inconsistent_within_player)}  <-- labelling bug if >0")
    for gid, player, outcomes in inconsistent_within_player[:10]:
        print(f"      game_id={gid} player={player} outcomes_seen={outcomes}")
    print(f"  outcome NOT complementary between the two players of a game: {len(non_complementary)}  <-- labelling bug if >0")
    for gid, outcomes in non_complementary[:10]:
        print(f"      game_id={gid} outcomes={outcomes}")
    print(f"  games with only ONE player's shard present (can't check complementarity): {len(single_perspective_games)}")
    for gid, players in single_perspective_games[:10]:
        print(f"      game_id={gid} players_present={players}")
    print()

    # =======================================================================
    print("=" * 78)
    print("4. STATES PER GAME")
    print("=" * 78)
    lengths = [len(recs) for recs in game_records.values()]
    print(f"  distinct game_ids: {len(game_records)}  (expected 50: {'MATCH' if len(game_records) == 50 else 'MISMATCH'})")
    if lengths:
        print(f"  states/game: mean={statistics.mean(lengths):.2f}  median={statistics.median(lengths):.1f}  "
              f"min={min(lengths)}  max={max(lengths)}")
    print()

    # =======================================================================
    print("=" * 78)
    print("5. CurrentPlayer.PlayerID DISTRIBUTION")
    print("=" * 78)
    print(f"  raw values seen: {dict(current_player_id_counter)}")
    has_both = 0 in current_player_id_counter and 1 in current_player_id_counter
    print(f"  both 0 and 1 present: {'YES' if has_both else 'NO -- BUG'}")
    if has_both:
        c0, c1 = current_player_id_counter[0], current_player_id_counter[1]
        gap_pct = abs(c0 - c1) / max(c0, c1) * 100
        print(f"  0 vs 1 gap: {gap_pct:.2f}%  [{'OK' if gap_pct <= 5 else 'WARNING: not roughly even'}]")
    print(f"  records where top-level 'player' field != state.CurrentPlayer.PlayerID: "
          f"{len(player_field_vs_state_mismatches)}  <-- should always be 0 by construction")
    for shard, gid, pf, cpid in player_field_vs_state_mismatches[:10]:
        print(f"      shard={os.path.relpath(shard, args.data_dir)} game_id={gid} player_field={pf} state_playerid={cpid}")
    print()

    # =======================================================================
    print("=" * 78)
    print("6. PATRON COVERAGE")
    print("=" * 78)
    print(f"  unexpected patron keys (not in the 6-patron pool + TREASURY): {len(unexpected_patron_keys)}")
    for k, cnt in unexpected_patron_keys.most_common():
        examples = unexpected_patron_examples[k]
        print(f"      {k!r}: {cnt} records, e.g. {examples}")
    if not unexpected_patron_keys:
        print("      none -- every PatronStates.All key is one of the 7 allowed patrons.")
    print()
    combo_counter = Counter(game_patron_combo.values())
    print(f"  games with a resolvable 4-patron draft: {len(game_patron_combo)} / {len(game_records)}")
    print(f"  draft combo counts (of {len(ALL_15_DRAFT_COMBOS)} possible):")
    for combo in sorted(ALL_15_DRAFT_COMBOS):
        cnt = combo_counter.get(combo, 0)
        marker = "" if cnt else "  <-- never occurred in this sample"
        print(f"      {', '.join(combo):55s} {cnt:3d}{marker}")
    unexpected_combos = set(combo_counter) - ALL_15_DRAFT_COMBOS
    if unexpected_combos:
        print(f"  WARNING: {len(unexpected_combos)} combo(s) seen that aren't among the 15 expected:")
        for combo in unexpected_combos:
            print(f"      {combo} : {combo_counter[combo]}")
    print()

    # =======================================================================
    print("=" * 78)
    print("7. OPENING DIVERSITY (first 3 BUY_CARD moves per game, both perspectives merged)")
    print("=" * 78)
    opening_sequences = {}
    short_games = []
    for game_id, records in game_records.items():
        ordered = sorted(records, key=lambda r: r[1])  # by CompletedActions length
        buys = [action.split(" ", 1)[1] if " " in action else action
                for _, _, action, _ in ordered if action.startswith("BUY_CARD")]
        first3 = tuple(buys[:3])
        opening_sequences[game_id] = first3
        if len(buys) < 3:
            short_games.append((game_id, buys))

    seq_counter = Counter(opening_sequences.values())
    distinct = len(seq_counter)
    print(f"  games with a reconstructed opening: {len(opening_sequences)}")
    print(f"  distinct first-3-BUY_CARD sequences: {distinct} / {len(opening_sequences)} games")
    if short_games:
        print(f"  games with fewer than 3 BUY_CARD moves total (partial sequence): {len(short_games)}")
        for gid, buys in short_games[:5]:
            print(f"      game_id={gid} buys={buys}")
    print()
    print("  sequence frequency (most common first):")
    for seq, cnt in seq_counter.most_common(20):
        pct = cnt / len(opening_sequences) * 100
        print(f"      {cnt:3d} ({pct:5.1f}%)  {seq}")
    most_common_seq, most_common_cnt = seq_counter.most_common(1)[0]
    dominant_pct = most_common_cnt / len(opening_sequences) * 100
    if dominant_pct > 30 or distinct < len(opening_sequences) * 0.5:
        print()
        print(f"  WARNING: openings are concentrated (top sequence covers {dominant_pct:.1f}% of games, "
              f"only {distinct}/{len(opening_sequences)} distinct) -- consider raising "
              f"EXPLORATION_TEMPERATURE above 1.0 before the cluster run.")
    else:
        print()
        print(f"  Openings look reasonably diverse (top sequence covers {dominant_pct:.1f}% of games, "
              f"{distinct}/{len(opening_sequences)} distinct).")
    print()

    # =======================================================================
    print("=" * 78)
    print("8. DECK FIELD")
    print("=" * 78)
    print(f"  distinct Deck values seen: {sorted(k for k in deck_counter if k is not None)}")
    for deck_id in sorted((k for k in deck_counter if k is not None)):
        print(f"      Deck={deck_id:>3}  {deck_counter[deck_id]:>10,}")
    if None in deck_counter:
        print(f"      Deck=None (missing field): {deck_counter[None]:,}")
    has_9 = deck_counter.get(9, 0) > 0
    print(f"  Deck=9 (Saint Alessia) present: {'YES' if has_9 else 'NO -- unexpected, Alessia is in the patron pool'}"
          f"  ({deck_counter.get(9, 0):,} card instances)")
    print()

    # =======================================================================
    print("=" * 78)
    print("9. START-OF-TURN FRACTION")
    print("=" * 78)
    print("  A state is start-of-turn under the same rule SakkirinaGen.Play() itself")
    print("  uses to detect a fresh turn: CompletedActions is empty, or its last entry")
    print("  is END_TURN. This is the population the value network is evaluating at")
    print("  inference ~99.85% of the time (end-of-turn == the START of the opponent's")
    print("  next turn from their own Play() call).")
    start_of_turn_frac = start_of_turn_count / total_records if total_records else 0.0
    print(f"  start-of-turn records: {start_of_turn_count:,} / {total_records:,} = {start_of_turn_frac:.4f}")
    print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    problems = []
    if b1 and b2 and abs(b1 - b2) / max(b1, b2) > 0.05:
        problems.append("bot1/bot2 shard record-count gap exceeds 5%")
    if not (0.45 <= base_rate <= 0.55):
        problems.append(f"outcome base rate {base_rate:.4f} not near 0.50")
    if inconsistent_within_player:
        problems.append(f"{len(inconsistent_within_player)} (game_id, player) pairs with inconsistent outcome")
    if non_complementary:
        problems.append(f"{len(non_complementary)} games with non-complementary outcomes")
    if len(game_records) != 50:
        problems.append(f"expected 50 distinct game_ids, found {len(game_records)}")
    if not has_both:
        problems.append("CurrentPlayer.PlayerID does not have both 0 and 1 present")
    if player_field_vs_state_mismatches:
        problems.append(f"{len(player_field_vs_state_mismatches)} player-field/state mismatches")
    if unexpected_patron_keys:
        problems.append(f"{len(unexpected_patron_keys)} unexpected patron key(s)")
    if dominant_pct > 30 or distinct < len(opening_sequences) * 0.5:
        problems.append("opening sequences are concentrated -- raise EXPLORATION_TEMPERATURE")
    if not has_9:
        problems.append("Deck=9 (Saint Alessia) never appears")

    if problems:
        print(f"  {len(problems)} issue(s) found -- DO NOT run the cluster job until these are resolved:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)
    else:
        print("  No issues found. Looks safe to scale up to the cluster run.")


if __name__ == "__main__":
    main()
