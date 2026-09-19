import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from StateParser import NODE_DIM, GLOBAL_DIM

class TributeValueNetwork(nn.Module):
    def __init__(self, node_in_dim=NODE_DIM, global_in_dim=GLOBAL_DIM, hidden_dim=128):
        super(TributeValueNetwork, self).__init__()

        # 1. Card (Node) Encoder
        # Translates the NODE_DIM-dim card features into a 128-dim embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Global Context Encoder
        # Translates the GLOBAL_DIM-dim resources into 128-dim embedding
        self.global_encoder = nn.Sequential(
            nn.Linear(global_in_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3. Final Evaluator
        # Takes the pooled nodes + global context (128 + 128 = 256 dims) and outputs a logit
        self.evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, u = data.x, data.u
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Process every card [num_cards, NODE_DIM] -> [num_cards, 128]
        node_embeddings = self.node_encoder(x)

        # Pool the cards together [num_cards, 128] -> [1, 128]
        pooled_nodes = global_mean_pool(node_embeddings, batch)

        # Global Context [1, GLOBAL_DIM] -> [1, 128]
        global_embeddings = self.global_encoder(u)
        
        # Merge and Evaluate [1, 256]
        combined = torch.cat([pooled_nodes, global_embeddings], dim=1)
        
        # Output [1, 1]
        win_prob = self.evaluator(combined)
        
        return win_prob
