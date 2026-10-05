"""
Turns a logged game state into the node matrix and global vector the value
network trains on.

The same schema is implemented in C# by FeatureExtractor in
Bots/src/DeepSetsCore.cs. A change to one must be made to the other; check
the two with tools/verify_parity.py.
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
LOC_ENEMY_UNSEEN   = 8.0  # Enemy hand and draw pile combined. Deck composition is
                          # public; only the order and the hand/draw split are hidden.

# Patron id -> one-hot slot in the node vector's Deck block. Only the
# competition pool is mapped; PSIJIC, HLAALU and RED_EAGLE never appear and
# encode as all-zero.
DECK_ID_TO_SLOT = {
    0: 0,  # ANSEI
    1: 1,  # DUKE_OF_CROWS
    2: 2,  # RAJHIN
    4: 3,  # ORGNUM
    6: 4,  # PELIN
    8: 5,  # TREASURY
    9: 6,  # SAINT_ALESSIA
}

# Patrons that can be favoured, in global-vector order; matches PatronOrder in
# DeepSetsCore.cs. TREASURY has no favour mechanic, so it is left out here,
# though its cards are still encoded through DECK_ID_TO_SLOT.
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

    # [Indices 0-6] Deck (7 dims, one-hot over the competition patrons)
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

    # [Indices 7-12] Patron favour. PatronStates.All holds PlayerEnum ints
    # (0=PLAYER1, 1=PLAYER2, 2=NO_PLAYER_SELECTED). Compare against the logged
    # CurrentPlayer.PlayerID, since records are logged from both seats.
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

    # [Index 15] My deck size (hand + played + cooldown + draw, excluding agents)
    global_vec[15] = (len(current.get("Hand", [])) + len(current.get("Played", [])) +
                       len(current.get("CooldownPile", [])) + len(current.get("DrawPile", []))) / 30.0

    # [Index 16] Enemy known deck size (HandAndDraw + cooldown, excluding agents)
    global_vec[16] = (len(enemy.get("HandAndDraw", [])) + len(enemy.get("CooldownPile", []))) / 30.0

    # [Indices 17-18] Agent counts
    global_vec[17] = len(current.get("Agents", [])) / 5.0
    global_vec[18] = len(enemy.get("Agents", [])) / 5.0

    return global_vec


def json_to_pyg_graph(game_state):
    """
    Pack one game state into a PyG Data object.
    """
    # Node order is part of the schema: the flat-MLP ablation is not
    # permutation-invariant, and DeepSetsCore.cs emits the same order.
    all_nodes = []

    tavern_cards = game_state.get("TavernAvailableCards", [])
    for card in tavern_cards:
        all_nodes.append(encode_card(card, LOC_TAVERN))

    current = game_state.get("CurrentPlayer", {})
    for card in current.get("Hand", []):
        all_nodes.append(encode_card(card, LOC_MY_HAND))

    for card in current.get("Played", []):
        all_nodes.append(encode_card(card, LOC_MY_PLAYED))

    for card in current.get("CooldownPile", []):
        all_nodes.append(encode_card(card, LOC_MY_COOLDOWN))

    for card in current.get("DrawPile", []):
        all_nodes.append(encode_card(card, LOC_MY_DRAW))

    for agent in current.get("Agents", []):
        card = agent.get("RepresentingCard", agent)
        card_with_stats = dict(card)
        card_with_stats['HP'] = agent.get('CurrentHp', card.get('HP', -1))
        card_with_stats['Taunt'] = agent.get('Taunt', card.get('Taunt', False))
        all_nodes.append(encode_card(card_with_stats, LOC_MY_AGENT))

    enemy = game_state.get("EnemyPlayer", {})
    for agent in enemy.get("Agents", []):
        card = agent.get("RepresentingCard", agent)
        card_with_stats = dict(card)
        card_with_stats['HP'] = agent.get('CurrentHp', card.get('HP', -1))
        card_with_stats['Taunt'] = agent.get('Taunt', card.get('Taunt', False))
        all_nodes.append(encode_card(card_with_stats, LOC_ENEMY_AGENT))

    for card in enemy.get("CooldownPile", []):
        all_nodes.append(encode_card(card, LOC_ENEMY_COOLDOWN))

    for card in enemy.get("HandAndDraw", []):
        all_nodes.append(encode_card(card, LOC_ENEMY_UNSEEN))

    # An empty board does not occur; one zero node keeps the shapes valid if it does.
    if len(all_nodes) == 0:
        all_nodes.append(torch.zeros(NODE_DIM, dtype=torch.float32))

    x = torch.stack(all_nodes)  # Shape: [num_nodes, NODE_DIM]

    u = extract_global_context(game_state).unsqueeze(0)  # Shape: [1, GLOBAL_DIM]

    edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, u=u)
