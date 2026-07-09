"""Message-passing backbones (GCN / GAT / GIN) with residual + LayerNorm.

Produces per-node embeddings; the readout heads (edge-level or pooled) live in
their own modules. Kept small so the whole model stays well under 1M params.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv


def _make_conv(name: str, in_dim: int, out_dim: int):
    name = name.lower()
    if name == "gcn":
        return GCNConv(in_dim, out_dim)
    if name == "gat":
        # single head keeps output dim == out_dim and params low
        return GATConv(in_dim, out_dim, heads=1)
    if name == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        return GINConv(mlp)
    raise ValueError(f"Unknown backbone '{name}' (use gcn|gat|gin).")


class GNNBackbone(nn.Module):
    """Stack of message-passing layers with residual connections + LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int,
                 backbone: str = "gcn", dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            _make_conv(backbone, hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index)
            m = self.act(norm(m))
            m = self.dropout(m)
            h = h + m  # residual
        return h
