"""Bag-of-amino-acids MLP baseline.

Predicts the pooled [ion x position] profile from order-invariant features:
amino-acid composition (recovered from the one-hot node channels), peptide
length, and precursor charge. Any GNN worth its cost must beat this order-blind
model, since a path-graph GCN with mean pooling is close to a bag-of-AA in
disguise (§2.7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from ..data.features import NUM_AA_CHANNELS


class BagOfAAMLP(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 128,
                 max_charge: int = 8, dropout: float = 0.1):
        super().__init__()
        in_dim = NUM_AA_CHANNELS + 2  # composition + length + charge
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.max_charge = max_charge

    def _features(self, batch) -> torch.Tensor:
        aa = batch.x[:, :NUM_AA_CHANNELS]              # one-hot slice
        comp = global_add_pool(aa, batch.batch)        # [G, 21] counts
        length = comp.sum(dim=1, keepdim=True)         # [G, 1]
        comp = comp / length.clamp(min=1)              # normalise to fractions
        charge = batch.charge.float().unsqueeze(1) / self.max_charge
        return torch.cat([comp, length / 40.0, charge], dim=1)

    def forward(self, batch) -> torch.Tensor:
        return self.net(self._features(batch))         # logits [G, out_dim]
