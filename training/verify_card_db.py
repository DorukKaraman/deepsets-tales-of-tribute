"""
Read-only audit of card_db.py against its source of truth, Engine/cards.json.

card_db.py supplies 76 of the value network's 101 node-feature dimensions and
is the SOLE source of truth for card effects (the logged "Effects" field in
game states is always empty, so there is no independent cross-check). This
script re-derives card_db.py's parsing from scratch (a verbatim copy of
generate_db.py's WORD_TO_INDEX / parse_effect_text, not an import of
generate_db.py itself, since importing that module would re-run its top-level
code and overwrite card_db.py) and reports every place the parse can silently
drop information.

Does not modify card_db.py, generate_db.py, or cards.json. Writes only to
tools/out/unknown_effect_tokens.txt and tools/out/card_db_roundtrip_mismatches.txt.
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CARDS_JSON_PATH = REPO_ROOT / "Engine" / "cards.json"
OUT_DIR = REPO_ROOT / "tools" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from card_db import CARD_EFFECTS  # read-only: card_db.py has no side effects on import

# --- verbatim copy of generate_db.py's parsing logic -----------------------
WORD_TO_INDEX = {
    'Coin': 0,
    'Power': 1,
    'Prestige': 2,
    'OppLosePrestige': 3,
    'Replace': 4,
    'Acquire': 5,
    'Destroy': 6,
    'Remove': 7,             # Deck Thinning (Self)
    'Donate': 8,             # Deck Clogging (Opponent)
    'Draw': 9,
    'Discard': 10,
    'Return': 11,            # Cooldown to Top of Deck
    'ReturnAgent': 12,
    'Toss': 13,
    'Knockout': 14, 'KnockOut': 14,
    'KnockOutAll': 15,       # Board wipe
    'Patron': 16,
    'Summerset': 17, 'CreateSummersetSacking': 17,
    'Heal': 18,
}
COMBO_FIELDS = ["Activation", "Combo 2", "Combo 3", "Combo 4"]

# Canonical display name per index (first key seen wins; duplicates are pure
# synonyms and don't affect the numeric vector either way).
INDEX_TO_WORD = {}
for _w, _i in WORD_TO_INDEX.items():
    INDEX_TO_WORD.setdefault(_i, _w)

DECK_TO_ID = {
    "Ansei": 0, "Crows": 1, "Rajhin": 2, "Psijic": 3, "Orgnum": 4,
    "Hlaalu": 5, "Pelin": 6, "Red Eagle": 7, "Treasury": 8, "Saint Alessia": 9,
}
COMPETITION_DECK_IDS = {0, 1, 2, 4, 6, 8, 9}  # ANSEI, DUKE_OF_CROWS, RAJHIN, ORGNUM, PELIN, TREASURY, SAINT_ALESSIA
DROPPABLE_DECK_IDS = {3, 5, 7}                # PSIJIC, HLAALU, RED_EAGLE


def parse_effect_text(effect_text):
    """Exact copy of generate_db.py's parser. Returns (19-dim list, list of
    (word, raw_text) unknown-token hits) for this one field's text."""
    values = [0.0] * 19
    unknown_hits = []
    if not effect_text:
        return values, unknown_hits

    text = str(effect_text)
    is_or_choice = "OR" in text
    multiplier = 0.5 if is_or_choice else 1.0

    matches = re.finditer(r"([a-zA-Z]+)\s*(\d+)", text)
    for match in matches:
        word = match.group(1)
        value = float(match.group(2))
        if word in WORD_TO_INDEX:
            idx = WORD_TO_INDEX[word]
            values[idx] += value * multiplier
        elif word != "OR" and word != "AND":
            unknown_hits.append(word)

    return values, unknown_hits


def find_numberless_known_words(text):
    """Known effect words in `text` that are NOT immediately followed by
    (optional whitespace +) digits -- i.e. words the parsing regex would
    never capture as (word, number) starting at that position, because no
    number directly follows them."""
    if not text:
        return []
    text = str(text)
    hits = []
    for m in re.finditer(r"[A-Za-z]+", text):
        word = m.group(0)
        if word not in WORD_TO_INDEX:
            continue
        rest = text[m.end():]
        if not re.match(r"\s*\d+", rest):
            hits.append(word)
    return hits


def render_vector(vec76):
    parts = []
    for i, field in enumerate(COMBO_FIELDS):
        block = vec76[i * 19:(i + 1) * 19]
        nonzero = [(INDEX_TO_WORD.get(idx, f"idx{idx}"), v) for idx, v in enumerate(block) if v != 0]
        if nonzero:
            inner = ", ".join(f"{w} {v:g}" for w, v in nonzero)
            parts.append(f"{field}: {inner}")
    return " | ".join(parts) if parts else "(no effects)"


def render_source(card):
    parts = []
    for field in COMBO_FIELDS:
        v = card.get(field)
        if v:
            parts.append(f"{field}: {v}")
    return " | ".join(parts) if parts else "(no effect fields)"


# --- load ---------------------------------------------------------------
with open(CARDS_JSON_PATH, "r") as f:
    cards = json.load(f)

print(f"Loaded {len(cards)} cards from {CARDS_JSON_PATH}")
print(f"Loaded {len(CARD_EFFECTS)} entries from card_db.py")
print()

# --- single pass: re-derive everything from cards.json -------------------
derived = {}                       # card_id -> 76-dim list (independently re-derived)
unknown_counter = Counter()        # token -> count
unknown_examples = defaultdict(list)   # token -> [card names] (up to 3)
numberless_by_card = defaultdict(list)  # card_id -> [(field, word)]
or_fields = []                     # (card, field, text, block)
card_by_id = {}

unknown_by_card = defaultdict(list)  # card_id -> [(field, word)]

for card in cards:
    cid = card.get("id")
    if cid is None:
        continue
    card_by_id[cid] = card

    full = []
    for field in COMBO_FIELDS:
        text = card.get(field)
        block, unknown_hits = parse_effect_text(text)
        full.extend(block)

        for word in unknown_hits:
            unknown_counter[word] += 1
            unknown_by_card[cid].append((field, word))
            if len(unknown_examples[word]) < 3 and card["Name"] not in unknown_examples[word]:
                unknown_examples[word].append(card["Name"])

        for word in find_numberless_known_words(text):
            numberless_by_card[cid].append((field, word))

        if text and "OR" in str(text):
            or_fields.append((card, field, text, block))

    derived[cid] = full

print("=" * 78)
print("1. COVERAGE")
print("=" * 78)

zero_vector_ids = [cid for cid, v in CARD_EFFECTS.items() if all(x == 0.0 for x in v)]
genuine_no_effect = []
parse_failures = []
for cid in zero_vector_ids:
    card = card_by_id.get(cid)
    if card is None:
        continue
    has_real_text = any(card.get(f) for f in COMBO_FIELDS)
    if has_real_text:
        parse_failures.append(card)
    else:
        genuine_no_effect.append(card)

print(f"  cards with all-zero 76-vector: {len(zero_vector_ids)} / {len(CARD_EFFECTS)}")
print(f"    - genuinely no effect text in any field (expected zero): {len(genuine_no_effect)}")
print(f"    - HAS effect text but vector is all-zero (PARSE FAILURE): {len(parse_failures)}")
for card in parse_failures:
    print(f"        id={card['id']:3d}  {card['Name']:35s}  {render_source(card)}")

# Staleness cross-check: does card_db.py match a fresh re-derivation from cards.json?
stale = []
for cid, stored in CARD_EFFECTS.items():
    if cid not in derived:
        stale.append((cid, "not present in current cards.json"))
        continue
    if stored != derived[cid]:
        stale.append((cid, "vector differs from fresh re-derivation"))
missing_from_db = [cid for cid in card_by_id if cid not in CARD_EFFECTS]
print(f"  card_db.py entries stale vs. a fresh re-derivation from cards.json: {len(stale)}")
for cid, reason in stale:
    name = card_by_id.get(cid, {}).get("Name", "?")
    print(f"        id={cid:3d}  {name:35s}  {reason}")
print(f"  cards.json ids missing from card_db.py entirely: {len(missing_from_db)} -> {missing_from_db}")
print()

print("=" * 78)
print("2. UNKNOWN TOKENS")
print("=" * 78)
if not unknown_counter:
    print("  none -- every alphabetic token adjacent to a number is a known effect word.")
else:
    for word, count in unknown_counter.most_common():
        examples = ", ".join(unknown_examples[word])
        print(f"  {word:20s} count={count:4d}  examples: {examples}")

unknown_tokens_path = OUT_DIR / "unknown_effect_tokens.txt"
with open(unknown_tokens_path, "w") as f:
    f.write("token\tcount\texample_cards\n")
    for word, count in unknown_counter.most_common():
        f.write(f"{word}\t{count}\t{', '.join(unknown_examples[word])}\n")
print(f"  full list written to {unknown_tokens_path}")
print()

print("=" * 78)
print("3. NUMBERLESS EFFECTS (known effect word with no trailing digits -- invisible to the regex)")
print("=" * 78)
if not numberless_by_card:
    print("  none found.")
else:
    total_hits = sum(len(v) for v in numberless_by_card.values())
    print(f"  {len(numberless_by_card)} cards, {total_hits} numberless known-word occurrences:")
    for cid, hits in numberless_by_card.items():
        card = card_by_id[cid]
        hit_str = ", ".join(f"{field}:'{word}'" for field, word in hits)
        print(f"        id={cid:3d}  {card['Name']:35s}  {hit_str}")
knockout_all_numberless = [cid for cid, hits in numberless_by_card.items()
                            if any(w in ("KnockOutAll",) for _, w in hits)]
if not knockout_all_numberless:
    print("  NOTE: expected KNOCKOUT_ALL ('Knockout all agents') to appear here as a")
    print("        known numberless example -- it does NOT in the current cards.json.")
    print("        Every 'KnockOutAll' occurrence found has a trailing count (e.g. 'KnockOutAll 1').")
print()

print("=" * 78)
print("4. ROUND-TRIP")
print("=" * 78)
roundtrip_mismatches = []
for cid, card in card_by_id.items():
    stored = CARD_EFFECTS.get(cid)
    if stored is None:
        continue
    rendered = render_vector(stored)
    source = render_source(card)
    missing_info = (cid in unknown_by_card) or (cid in numberless_by_card)
    if missing_info:
        roundtrip_mismatches.append((card, source, rendered))

print(f"  cards where the rendered vector is missing information present in source: {len(roundtrip_mismatches)} / {len(card_by_id)}")
print(f"  sample (first 10):")
for card, source, rendered in roundtrip_mismatches[:10]:
    print(f"        id={card['id']:3d}  {card['Name']}")
    print(f"            source:   {source}")
    print(f"            rendered: {rendered}")

roundtrip_path = OUT_DIR / "card_db_roundtrip_mismatches.txt"
with open(roundtrip_path, "w") as f:
    f.write("All cards where the rendered card_db.py vector loses information present in cards.json.\n\n")
    for card, source, rendered in roundtrip_mismatches:
        f.write(f"id={card['id']}  {card['Name']}  (Deck={card.get('Deck')})\n")
        f.write(f"    source:   {source}\n")
        f.write(f"    rendered: {rendered}\n\n")
print(f"  full mismatch list written to {roundtrip_path}")
print()

print("=" * 78)
print("5. COMBO DISTRIBUTION")
print("=" * 78)
block_nonzero_counts = [0, 0, 0, 0]
for cid, vec in CARD_EFFECTS.items():
    for i in range(4):
        block = vec[i * 19:(i + 1) * 19]
        if any(x != 0.0 for x in block):
            block_nonzero_counts[i] += 1

for i, field in enumerate(COMBO_FIELDS):
    print(f"  {field:12s}: {block_nonzero_counts[i]} cards with non-zero entries in this block")

combo4_field_count = sum(1 for card in cards if card.get("Combo 4"))
print(f"  sanity check: cards.json cards with a non-null 'Combo 4' field: {combo4_field_count}")
if combo4_field_count == block_nonzero_counts[3]:
    print(f"  MATCH: Combo 4 non-zero-block count equals the number of cards with a Combo 4 field.")
else:
    print(f"  MISMATCH: {block_nonzero_counts[3]} non-zero Combo-4 vectors vs {combo4_field_count} cards "
          f"with a Combo 4 field -- {abs(combo4_field_count - block_nonzero_counts[3])} card(s) have a "
          f"Combo 4 field that parsed to an all-zero block, or vice versa.")
    combo4_text_but_zero = [c for c in cards if c.get("Combo 4") and
                             all(x == 0.0 for x in derived.get(c["id"], [0]*76)[57:76])]
    for c in combo4_text_but_zero:
        print(f"        id={c['id']:3d}  {c['Name']:35s}  Combo 4: {c['Combo 4']!r}")
print()

print("=" * 78)
print("6. \"OR\" EFFECTS")
print("=" * 78)
print(f"  {len(or_fields)} (card, field) pairs contain 'OR':")
for card, field, text, block in or_fields:
    nonzero = [(INDEX_TO_WORD.get(idx, f"idx{idx}"), v) for idx, v in enumerate(block) if v != 0]
    vec_str = ", ".join(f"{w} {v:g}" for w, v in nonzero)
    print(f"        id={card['id']:3d}  {card['Name']:30s}  {field:12s}  text={text!r:40s}  vector=[{vec_str}]")
print()

print("=" * 78)
print("7. COMPETITION SUBSET (ANSEI, DUKE_OF_CROWS, RAJHIN, ORGNUM, PELIN, SAINT_ALESSIA, TREASURY)")
print("=" * 78)
comp_cards = [c for c in cards if DECK_TO_ID.get(c.get("Deck")) in COMPETITION_DECK_IDS]
drop_cards = [c for c in cards if DECK_TO_ID.get(c.get("Deck")) in DROPPABLE_DECK_IDS]
print(f"  cards in competition decks: {len(comp_cards)} / {len(cards)}")
print(f"  cards in droppable decks (PSIJIC, HLAALU, RED_EAGLE): {len(drop_cards)} -- can be excluded from card_db")
print()

comp_ids = {c["id"] for c in comp_cards}

comp_zero_vector = [cid for cid in zero_vector_ids if cid in comp_ids]
comp_parse_failures = [c for c in parse_failures if c["id"] in comp_ids]
print(f"  [7.1 coverage, competition subset] all-zero vectors: {len(comp_zero_vector)}, "
      f"of which parse failures (has text, zero vector): {len(comp_parse_failures)}")
for card in comp_parse_failures:
    print(f"        id={card['id']:3d}  {card['Name']:35s}  {render_source(card)}")

comp_unknown_counter = Counter()
comp_unknown_examples = defaultdict(list)
for card in comp_cards:
    for field in COMBO_FIELDS:
        text = card.get(field)
        _, unk = parse_effect_text(text)
        for word in unk:
            comp_unknown_counter[word] += 1
            if len(comp_unknown_examples[word]) < 3 and card["Name"] not in comp_unknown_examples[word]:
                comp_unknown_examples[word].append(card["Name"])
print(f"  [7.2 unknown tokens, competition subset] {len(comp_unknown_counter)} distinct tokens:")
for word, count in comp_unknown_counter.most_common():
    print(f"        {word:20s} count={count:4d}  examples: {', '.join(comp_unknown_examples[word])}")

comp_numberless = {cid: hits for cid, hits in numberless_by_card.items() if cid in comp_ids}
comp_numberless_hits = sum(len(v) for v in comp_numberless.values())
print(f"  [7.3 numberless effects, competition subset] {len(comp_numberless)} cards, {comp_numberless_hits} hits")
for cid, hits in comp_numberless.items():
    card = card_by_id[cid]
    hit_str = ", ".join(f"{field}:'{word}'" for field, word in hits)
    print(f"        id={cid:3d}  {card['Name']:35s}  {hit_str}")

comp_block_counts = [0, 0, 0, 0]
for cid in comp_ids:
    vec = CARD_EFFECTS.get(cid)
    if vec is None:
        continue
    for i in range(4):
        block = vec[i * 19:(i + 1) * 19]
        if any(x != 0.0 for x in block):
            comp_block_counts[i] += 1
print(f"  [7.4 combo distribution, competition subset]")
for i, field in enumerate(COMBO_FIELDS):
    print(f"        {field:12s}: {comp_block_counts[i]} cards")
comp_combo4_field_count = sum(1 for c in comp_cards if c.get("Combo 4"))
print(f"        sanity check: {comp_combo4_field_count} competition cards have a Combo 4 field "
      f"vs {comp_block_counts[3]} with a non-zero Combo-4 block")

comp_or_fields = [x for x in or_fields if x[0]["id"] in comp_ids]
print(f"  [7.5 OR effects, competition subset] {len(comp_or_fields)} (card, field) pairs:")
for card, field, text, block in comp_or_fields:
    nonzero = [(INDEX_TO_WORD.get(idx, f"idx{idx}"), v) for idx, v in enumerate(block) if v != 0]
    vec_str = ", ".join(f"{w} {v:g}" for w, v in nonzero)
    print(f"        id={card['id']:3d}  {card['Name']:30s}  {field:12s}  text={text!r:40s}  vector=[{vec_str}]")

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  cards.json: {len(cards)} cards | card_db.py: {len(CARD_EFFECTS)} entries")
print(f"  parse failures (real text, zero vector): {len(parse_failures)} total, {len(comp_parse_failures)} in competition decks")
print(f"  stale card_db.py entries vs fresh re-derivation: {len(stale)}")
print(f"  distinct unknown tokens: {len(unknown_counter)} total, {len(comp_unknown_counter)} in competition decks")
print(f"  cards with numberless known-effect words: {len(numberless_by_card)} total, {len(comp_numberless)} in competition decks")
print(f"  round-trip mismatches (info lost in vector vs. source): {len(roundtrip_mismatches)}")
print(f"  OR-effect (card, field) pairs: {len(or_fields)} total, {len(comp_or_fields)} in competition decks")
print(f"  droppable-deck cards (PSIJIC/HLAALU/RED_EAGLE): {len(drop_cards)}")
print(f"  output files: {unknown_tokens_path}")
print(f"                {roundtrip_path}")
