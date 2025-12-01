import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class PepFragGNN(nn.Module):
    def __init__(self, in_dim=21, hidden_dim=64, num_layers=3, out_dim=78):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.act = nn.ReLU()

        # Graph-level readout: mean over nodes
        self.readout = global_mean_pool

        # Final MLP to predict 78 probabilities
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),   # output 0..1
        )

    def forward(self, x, edge_index, batch):
        """
        x: [N, in_dim]        (all nodes from all peptides in batch)
        edge_index: [2, E]
        batch: [N]            (graph id per node, for pooling)
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)

        # graph-level embedding
        g = self.readout(x, batch)   # [num_graphs, hidden_dim]

        out = self.mlp(g)            # [num_graphs, 78]
        return out
