"""
Padded, flattened encoder for the flat-MLP ablation.

Lays the cards StateParser encodes out as one fixed-size vector, in a fixed
order, with the tail zero-padded:

    [ node 0 | node 1 | ... | node MAX_NODES-1 | global ]
    [   99   |   99   |     |        99        |   19   ]   = FLAT_DIM

Nodes follow json_to_pyg_graph's emission order and the features come from
StateParser unchanged; only the assembly is defined here. Its three
implementations (per state, batched, ONNX graph) are checked against each
other by tools/verify_flat_parity.py and export_flat_to_onnx.py.

MAX_NODES is the corpus maximum, so nothing is truncated. A smaller cap would
drop rows from the end of the emission order, the enemy hand+draw block, and
only in the states that hold the most of it.
"""

import numpy as np
import torch

from StateParser import NODE_DIM, GLOBAL_DIM, json_to_pyg_graph

# Corpus maximum. Changing it changes FLAT_DIM and the size of the first layer.
MAX_NODES = 128

FLAT_NODE_DIM = MAX_NODES * NODE_DIM      # 12,672
FLAT_DIM = FLAT_NODE_DIM + GLOBAL_DIM     # 12,691

# Canonical row order, used by the matched_sorted arm: location first, then a
# fixed random projection of the 90 feature dimensions. The key depends only on
# a row's values, so the sorted matrix depends only on the multiset of rows.
#
# Distinct rows need distinct keys, or argsort's tie-breaking decides their
# order and need not agree between PyTorch and onnxruntime. SORT_KEY_LOC_SCALE
# keeps keys small enough (max ~34) for float32 to separate them, and must
# exceed the projection's range (about 0.03-2.6) so location stays primary.
# tools/verify_flat_parity.py checks for collisions.
#
# float32 rather than float64 because MPS has no float64.
SORT_KEY_SEED = 20260927
SORT_KEY_LOC_SCALE = 4.0

_sort_proj_np = np.random.default_rng(SORT_KEY_SEED).random(NODE_DIM - 9)
SORT_PROJECTION = torch.tensor(_sort_proj_np, dtype=torch.float32)
_LOC_WEIGHTS = torch.arange(9, dtype=torch.float32)


def row_sort_key(x):
    """[..., NODE_DIM] -> [...] float32 sort key. location * scale + projection."""
    proj = SORT_PROJECTION.to(x.device)
    locs = _LOC_WEIGHTS.to(x.device)
    return ((x[..., NODE_DIM - 9:] @ locs) * SORT_KEY_LOC_SCALE
            + x[..., :NODE_DIM - 9] @ proj)


def canonical_order(x):
    """[n, NODE_DIM] -> the same rows in canonical order."""
    return x[torch.argsort(row_sort_key(x), dim=0)]


def pad_and_flatten(x, u, sort=False):
    """[n, NODE_DIM] node matrix + [1, GLOBAL_DIM] globals -> [FLAT_DIM] vector.

    The reference definition of the flat layout. With sort=True the rows are
    put in canonical order before padding; sorting afterwards would move the
    zero rows, whose key is 0, to the front. Rows beyond MAX_NODES are dropped.
    """
    if x.shape[0] > MAX_NODES:
        x = x[:MAX_NODES]
    if sort:
        x = canonical_order(x)

    padded = x.new_zeros(MAX_NODES, NODE_DIM)
    padded[:x.shape[0]] = x
    return torch.cat([padded.reshape(-1), u.reshape(-1)])


def batch_pad_and_flatten(x, batch, u, num_graphs=None, sort=False):
    """PyG batch -> [B, FLAT_DIM]; the training-time path.

    Flattening here rather than in the dataset keeps the shuffle buffer holding
    the same compact graphs as the DeepSets run. A padded state is about 4x
    larger, and shrinking the buffer to fit would change the data order.
    """
    from torch_geometric.utils import to_dense_batch

    dense, mask = to_dense_batch(x, batch, batch_size=num_graphs,
                                 max_num_nodes=MAX_NODES)  # [B, MAX_NODES, NODE_DIM]
    if sort:
        # to_dense_batch has already padded. An infinite key keeps the padding
        # rows at the tail, matching pad_and_flatten.
        key = row_sort_key(dense).masked_fill(~mask, float("inf"))
        order = torch.argsort(key, dim=1)
        dense = torch.gather(dense, 1, order.unsqueeze(-1).expand(-1, -1, NODE_DIM))
    return torch.cat([dense.reshape(dense.shape[0], FLAT_NODE_DIM), u], dim=1)


def json_to_flat_vector(game_state, sort=False):
    """One logged game state -> [FLAT_DIM] float32 vector.

    The exported ONNX graph does this padding itself from the same
    (node_features, global_features) inputs as the DeepSets model, so a bot
    needs no separate flat encoder.
    """
    graph = json_to_pyg_graph(game_state)
    return pad_and_flatten(graph.x, graph.u, sort=sort)
