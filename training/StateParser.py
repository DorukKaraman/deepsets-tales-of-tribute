"""
Turns gamestate into tensors for training.

Schema mirrored exactly in Bots/src/DeepSetsCore.cs (class FeatureExtractor) -- any change here
MUST be made there too, in the same commit. After editing either file, run
python tools/verify_parity.py before proceeding.
"""

import torch
from torch_geometric.data import Data
from card_db import CARD_EFFECTS

NODE_DIM = 99
GLOBAL_DIM = 19

LOC_TAVERN         = 0.0
LOC_MY_HAND        = 1.0
LOC_MY_PLAYED      = 2.0
LOC_MY_COOLDOWN    = 3.0
LOC_MY_DRAW        = 4.0
LOC_MY_AGENT       = 5.0
LOC_ENEMY_AGENT    = 6.0
LOC_ENEMY_COOLDOWN = 7.0
LOC_ENEMY_UNSEEN   = 8.0  # EnemyPlayer.HandAndDraw -- replaces the old, always-
                          # empty ENEMY_PLAYED slot. Enemy deck composition is
                          # public in ToT (all acquisitions come from the
                          # face-up tavern); only the order of what's left in
                          # hand/draw is hidden.

# Competition-only patron pool: PSIJIC, HLAALU, RED_EAGLE can never appear
# (verified against the engine's PatronId enum and against 50 games of
# generated data -- deck values seen were exactly [0,1,2,4,6,8,9]).
# id -> one-hot slot in the node vector's Deck block. Anything not in this
# map stays all-zero, deliberately.
DECK_ID_TO_SLOT = {
    0: 0,  # ANSEI
    1: 1,  # DUKE_OF_CROWS
    2: 2,  # RAJHIN
    4: 3,  # ORGNUM
    6: 4,  # PELIN
    8: 5,  # TREASURY
    9: 6,  # SAINT_ALESSIA
}

# Competition-only, favour-eligible patrons for the global vector's favour
# block. TREASURY is deliberately excluded: it has no favour mechanic
# (neither player can be favoured by it), confirmed empirically -- its
# PatronStates.All value is NO_PLAYER_SELECTED in 1753/1753 sampled records
# -- so its slot was a constant 0.0 contributing nothing. TREASURY still
# appears in the node vector's Deck one-hot (DECK_ID_TO_SLOT) since Treasury
# cards exist on the board and must be encoded; only this favour list drops
# it. Matches DeepSetsCore.cs's FeatureExtractor PatronOrder exactly.
PATRON_ORDER = [
    "ANSEI", "DUKE_OF_CROWS", "RAJHIN", "ORGNUM", "PELIN", "SAINT_ALESSIA",
]


def encode_card(card_dict, location_id=0.0):
    """
    Build the NODE_DIM-dim card vector.
    """
    vector = torch.zeros(NODE_DIM, dtype=torch.float32)

    if card_dict is None:
        return vector

    # [Indices 0-6] Deck (7 dims, One-Hot over the live competition patrons
    # only -- anything else, e.g. PSIJIC/HLAALU/RED_EAGLE, stays all-zero
    # since it can never appear in a competition game).
    deck = int(card_dict.get('Deck', -1))
    if deck in DECK_ID_TO_SLOT:
        vector[DECK_ID_TO_SLOT[deck]] = 1.0

    # [Index 7] Cost (1 dim, Float, normalized)
    vector[7] = float(card_dict.get('Cost', 0)) / 10.0

    # [Indices 8-11] Type (4 dims, One-Hot)
    card_type = int(card_dict.get('Type', -1))
    if 0 <= card_type <= 3:
        vector[8 + card_type] = 1.0

    # [Index 12] HP (normalized)
    vector[12] = float(card_dict.get('HP', -1)) / 40.0

    # [Index 13] Taunt
    vector[13] = 1.0 if card_dict.get('Taunt', False) else 0.0

    # [Indices 14-89] Card Effects (76 dims, Float, normalized)
    common_id = card_dict.get('CommonId')
    if common_id in CARD_EFFECTS:
        effects_list = CARD_EFFECTS[common_id]
        vector[14:90] = torch.tensor(effects_list, dtype=torch.float32) / 5.0

    # [Indices 90-98] Location (9 dims, One-Hot)
    loc_id = int(location_id)
    if 0 <= loc_id <= 8:
        vector[90 + loc_id] = 1.0

    return vector


def extract_global_context(game_state):
    """
    Pull the main resource values into a GLOBAL_DIM-dim tensor.
    """
    global_vec = torch.zeros(GLOBAL_DIM, dtype=torch.float32)

    # [Indices 0-6] Player Resources
    current = game_state.get("CurrentPlayer", {})
    global_vec[0] = float(current.get("Coins", 0)) / 10.0
    global_vec[1] = float(current.get("Power", 0)) / 10.0
    global_vec[2] = float(current.get("Prestige", 0)) / 40.0
    global_vec[3] = float(current.get("PatronCalls", 1))

    enemy = game_state.get("EnemyPlayer", {})
    global_vec[4] = float(enemy.get("Coins", 0)) / 10.0
    global_vec[5] = float(enemy.get("Power", 0)) / 10.0
    global_vec[6] = float(enemy.get("Prestige", 0)) / 40.0

    # [Indices 7-12] Patron favour. PatronStates.All values are PlayerEnum
    # ints (0=PLAYER1, 1=PLAYER2, 2=NO_PLAYER_SELECTED) -- resolved against
    # the logged CurrentPlayer.PlayerID, NOT hardcoded to PLAYER1. This was a
    # no-op on the old (bot1-only) dataset because PlayerID was always 0; now
    # that both perspectives are logged, PlayerID is 0 or 1 and a hardcoded
    # favor_map would be wrong on every PlayerID==1 record.
    current_player_id = int(current.get("PlayerID", 0))
    patron_dict = game_state.get("PatronStates", {}).get("All", {})

    for i, patron_name in enumerate(PATRON_ORDER):
        if patron_name in patron_dict:
            raw_favor = patron_dict[patron_name]
            if raw_favor == current_player_id:
                global_vec[7 + i] = 1.0   # Us
            elif raw_favor == 2:
                global_vec[7 + i] = 0.0   # Neutral
            else:
                global_vec[7 + i] = -1.0  # Them
        else:
            # Patron wasn't drafted for this match
            global_vec[7 + i] = -2.0

    my_prestige = float(current.get("Prestige", 0))
    enemy_prestige = float(enemy.get("Prestige", 0))

    # [Index 13] Prestige clock
    global_vec[13] = min(max(my_prestige, enemy_prestige) / 40.0, 1.2)

    # [Index 14] Prestige differential
    global_vec[14] = (my_prestige - enemy_prestige) / 40.0

    # [Index 15] My deck size (hand + played + cooldown + draw, NOT agents)
    global_vec[15] = (len(current.get("Hand", [])) + len(current.get("Played", [])) +
                       len(current.get("CooldownPile", [])) + len(current.get("DrawPile", []))) / 30.0

    # [Index 16] Enemy known deck size (HandAndDraw + cooldown, NOT agents)
    global_vec[16] = (len(enemy.get("HandAndDraw", [])) + len(enemy.get("CooldownPile", []))) / 30.0

    # [Indices 17-18] Agent counts
    global_vec[17] = len(current.get("Agents", [])) / 5.0
    global_vec[18] = len(enemy.get("Agents", [])) / 5.0

    return global_vec


def json_to_pyg_graph(game_state):
    """
    Pack one game state into a PyG Data object.
    """
    all_nodes = []

    # Parse Tavern Cards
    tavern_cards = game_state.get("TavernAvailableCards", [])
    for card in tavern_cards:
        all_nodes.append(encode_card(card, LOC_TAVERN))

    # Parse Current Player Hand
    current = game_state.get("CurrentPlayer", {})
    for card in current.get("Hand", []):
        all_nodes.append(encode_card(card, LOC_MY_HAND))

    # Parse Current Player Played
    for card in current.get("Played", []):
        all_nodes.append(encode_card(card, LOC_MY_PLAYED))

    # Parse Current Player Cooldown
    for card in current.get("CooldownPile", []):
        all_nodes.append(encode_card(card, LOC_MY_COOLDOWN))

    # Parse Current Player DrawPile
    for card in current.get("DrawPile", []):
        all_nodes.append(encode_card(card, LOC_MY_DRAW))

    # Parse Current Player Agents
    for agent in current.get("Agents", []):
        card = agent.get("RepresentingCard", agent)
        card_with_stats = dict(card)
        card_with_stats['HP'] = agent.get('CurrentHp', card.get('HP', -1))
        card_with_stats['Taunt'] = agent.get('Taunt', card.get('Taunt', False))
        all_nodes.append(encode_card(card_with_stats, LOC_MY_AGENT))

    # Parse Enemy Player Agents
    enemy = game_state.get("EnemyPlayer", {})
    for agent in enemy.get("Agents", []):
        card = agent.get("RepresentingCard", agent)
        card_with_stats = dict(card)
        card_with_stats['HP'] = agent.get('CurrentHp', card.get('HP', -1))
        card_with_stats['Taunt'] = agent.get('Taunt', card.get('Taunt', False))
        all_nodes.append(encode_card(card_with_stats, LOC_ENEMY_AGENT))

    # Parse Enemy Player CooldownPile
    for card in enemy.get("CooldownPile", []):
        all_nodes.append(encode_card(card, LOC_ENEMY_COOLDOWN))

    # Parse Enemy Player HandAndDraw (publicly known deck composition; order hidden)
    for card in enemy.get("HandAndDraw", []):
        all_nodes.append(encode_card(card, LOC_ENEMY_UNSEEN))

    # If the board is completely empty, insert one dummy 0 vector card. (Does not happen but is a failsafe)
    if len(all_nodes) == 0:
        all_nodes.append(torch.zeros(NODE_DIM, dtype=torch.float32))

    x = torch.stack(all_nodes)  # Shape: [num_nodes, NODE_DIM]

    u = extract_global_context(game_state).unsqueeze(0)  # Shape: [1, GLOBAL_DIM]

    edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, u=u)
