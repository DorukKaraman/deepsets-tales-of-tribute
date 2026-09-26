"""
Padded, flattened encoder for the flat-MLP ablation.

WHAT THIS IS FOR. The DeepSets model's whole architectural claim is that a game
state is a SET of cards: encode each card independently, mean-pool, and the
result is permutation-invariant and independent of how many cards there are.
This module builds the input for the ablation that tests whether that structure
contributes anything, or whether the 99-dim card features alone carry the
result. It lays the same cards out as one fixed-size vector, in a fixed order,
with the tail zero-padded -- destroying both permutation invariance and size
independence while changing nothing about the features themselves.

NO FEATURE CODE LIVES HERE. Card and global encoding are imported from
StateParser, and the node matrix is taken straight from json_to_pyg_graph's
output rather than re-walking the nine card lists. The two encoders therefore
agree card-for-card by construction, not by a test that has to be kept passing:
there is no second copy of the schema to drift from the first. Only the
ASSEMBLY -- pad, flatten, concatenate -- is new, and that is what
flat_parity_check.py verifies.

LAYOUT.  [ node 0 | node 1 | ... | node MAX_NODES-1 | global ]
         [   99   |   99   |     |        99        |   19   ]   = FLAT_DIM
Nodes appear in json_to_pyg_graph's emission order: tavern, my hand, my played,
my cooldown, my draw, my agents, enemy agents, enemy cooldown, enemy hand+draw.
Rows past the state's node count are all-zero.

WHY MAX_NODES = 128 AND NOT SOMETHING CHEAPER. 128 is the true maximum over the
whole 3,116,065-state corpus, so nothing is ever truncated. That costs real
capacity in the matched model -- see ValueNetworkFlat -- and a smaller cap looks
almost free on the numbers: 96 truncates 0.011% of states, 60 truncates 1.01%.
It is not free. Truncation drops nodes from the END of the emission order, which
is ENEMY_UNSEEN (enemy hand+draw), the single largest contributor at 10.2 nodes
per state and 30% of all nodes -- and it drops them only in the states that have
the most of them. That is not 1% of states slightly degraded; it is 1% of states
with a specific, systematically chosen part of their input removed. The DeepSets
model sees those cards. A truncated flat model would not, and any accuracy gap
could then be attributed to missing information rather than to missing
structure, which is the one thing this ablation exists to measure.

So the cap is set where the comparison stays clean, and the cost is paid in the
first layer's width and stated openly rather than hidden.

PADDING WASTE IS A PROPERTY OF THE DATA, NOT JUST A PROBLEM FOR THIS BASELINE.
The median state has 33 nodes, so at MAX_NODES = 128 about 74% of the input
vector is structural zeros in a typical state; the mean is 34.3 nodes (73%), and
even the 99th percentile at 61 nodes leaves 52% padding. The distribution has a
long, thin tail rather than a wide body: 128 is reached by exactly ONE game out
of 12,160, a 1,071-state runaway that contributed 499 of the corpus's 936 states
above 85 nodes. A fixed-size encoding has to budget for that game and then carry
the empty space through every ordinary state. The set encoder never allocates
the space at all, which is a genuine argument for the architecture and not
merely an artefact of how this baseline was built.

Corpus statistics (all 3,116,065 states, both the heuristic and neural halves):
    min 25   median 33   mean 34.3   p90 44   p95 49   p99 61   p99.9 77   max 128
"""

import torch

from StateParser import NODE_DIM, GLOBAL_DIM, json_to_pyg_graph

# The true corpus maximum. Changing this changes the input dimension, the
# parameter budget's first layer, and the truncation rate -- see the module
# docstring before touching it.
MAX_NODES = 128

FLAT_NODE_DIM = MAX_NODES * NODE_DIM      # 12,672
FLAT_DIM = FLAT_NODE_DIM + GLOBAL_DIM     # 12,691


def pad_and_flatten(x, u):
    """[n, NODE_DIM] node matrix + [1, GLOBAL_DIM] globals -> [FLAT_DIM] vector.

    The single definition of the flat layout. Both json_to_flat_vector (one
    state, from JSON) and batch_pad_and_flatten (a PyG batch, during training)
    route through the same rule, and export_flat_to_onnx.py's wrapper
    reimplements it in traceable ops that flat_parity_check.py holds to this
    one's output.

    Rows beyond MAX_NODES are dropped. Not reachable on this corpus -- 128 is
    the measured maximum -- but a state from a future dataset must not silently
    produce a wrong-sized vector and a shape error somewhere downstream.
    """
    if x.shape[0] > MAX_NODES:
        x = x[:MAX_NODES]

    padded = x.new_zeros(MAX_NODES, NODE_DIM)
    padded[:x.shape[0]] = x
    return torch.cat([padded.reshape(-1), u.reshape(-1)])


def batch_pad_and_flatten(x, batch, u, num_graphs=None):
    """PyG batch -> [B, FLAT_DIM], the training-time path.

    WHY THE SHUFFLE BUFFER STILL HOLDS COMPACT GRAPHS. The obvious design is to
    flatten in the dataset, so the buffer holds [FLAT_DIM] vectors. That would
    make the flat run un-runnable and the comparison invalid at the same time:
    at 12,691 float32 a padded state is 50.8 KB against a compact state's ~13.6
    KB, and stream_dataset's buffer is 100,000 graphs PER WORKER -- about 35 GB
    across 7 workers, against ~10 GB for the DeepSets path, on a 24 GB
    allocation. Shrinking the buffer to fit would change the shuffling regime,
    so the flat and DeepSets runs would differ in data order as well as in
    architecture.

    Flattening at batch assembly instead means both runs stream the identical
    SakkirinaStreamDataset, hold identical objects in identically sized buffers,
    and see identical batches for a given seed. The only difference left between
    the two runs is the model, which is the point of an ablation.
    """
    from torch_geometric.utils import to_dense_batch

    dense, _ = to_dense_batch(x, batch, batch_size=num_graphs,
                              max_num_nodes=MAX_NODES)   # [B, MAX_NODES, NODE_DIM]
    return torch.cat([dense.reshape(dense.shape[0], FLAT_NODE_DIM), u], dim=1)


def json_to_flat_vector(game_state):
    """One logged game state -> [FLAT_DIM] float32 vector.

    The reference definition of the flat encoding, and the thing a C#
    FlatFeatureExtractor would have to match if either flat model ever earns a
    bot. Note that it may not need one: export_flat_to_onnx.py gives the
    exported graph the SAME (node_features, global_features) inputs the DeepSets
    model takes and pads inside the graph, so an existing bot could feed it
    through the unchanged FeatureExtractor in DeepSetsCore.cs.
    """
    graph = json_to_pyg_graph(game_state)
    return pad_and_flatten(graph.x, graph.u)
