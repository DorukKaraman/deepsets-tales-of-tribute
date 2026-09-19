"""
Maps the card data from cards.json into dictionary of CardId -> 76-dim effect vector.
"""
import json
import re

# 19 effect tokens we currently track
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
    'Heal': 18
}

def parse_effect_text(effect_text):
    """Parse one effect text into a 19-dim list."""
    values = [0.0] * 19
    if not effect_text:
        return values
        
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
            print(f"unknown effect token: {word} ({text})")
            
    return values

with open("../Engine/cards.json", "r") as f:
    cards_data = json.load(f)

card_effects = {}
combo_fields = ["Activation", "Combo 2", "Combo 3", "Combo 4"]

for card in cards_data:
    card_id = card.get("id")
    if card_id is None:
        continue
        
    full_effects = []  # 19 dims per field, 4 fields total
    for key in combo_fields:
        full_effects.extend(parse_effect_text(card.get(key)))
        
    card_effects[card_id] = full_effects

with open("card_db.py", "w") as f:
    f.write(f"CARD_EFFECTS = {card_effects}\n")

print("done")