"""CleavageGNN: edge-level (cleavage-site) fragment-probability model.

This is the project's core reformulation. Instead of pooling the peptide into a
single vector and predicting all positions from fixed MLP weights, we predict
each fragment **at its cleavage-site edge** from that bond's local node
embeddings, a global context vector, and the precursor-charge embedding:

    y_k = MLP([h_k, h_{k+1}, h_k * h_{k+1}, |h_k - h_{k+1}|, ctx, emb(z)])

Outputs are logits; the sigmoid is applied in the loss (BCEWithLogits) and at
inference for numerical stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from ..config import Config
from ..data.graph import node_feature_dim
from .backbones import GNNBackbone


class CleavageGNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        h = cfg.model.hidden_dim
        in_dim = node_feature_dim(cfg.graph)

        self.backbone = GNNBackbone(
            in_dim=in_dim, hidden_dim=h, num_layers=cfg.model.num_layers,
            backbone=cfg.model.backbone, dropout=cfg.model.dropout,
        )

        self.use_charge = cfg.model.use_charge_embedding
        charge_dim = 0
        if self.use_charge:
            charge_dim = 16
            # +1 so charge index z in 1..max maps into range; index 0 unused.
            self.charge_emb = nn.Embedding(cfg.model.max_precursor_charge + 1, charge_dim)

        # Readout input: [h_k, h_{k+1}, h_k*h_{k+1}, |h_k-h_{k+1}|, ctx] + charge.
        readout_in = 4 * h + h + charge_dim
        self.head = nn.Sequential(
            nn.Linear(readout_in, h),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(h, cfg.num_fragment_channels),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.backbone(x, edge_index)                 # [N, H]
        ctx = global_mean_pool(h, batch)                 # [G, H]

        cleave = data.cleave_index                       # [2, S] global node ids
        if cleave.numel() == 0:
            return h.new_zeros((0, self.cfg.num_fragment_channels))

        left = h[cleave[0]]                              # [S, H]
        right = h[cleave[1]]                             # [S, H]
        site_graph = batch[cleave[0]]                    # [S] graph id per site

        parts = [left, right, left * right, (left - right).abs(), ctx[site_graph]]
        if self.use_charge:
            z = data.charge[site_graph].clamp(min=1)     # [S]
            parts.append(self.charge_emb(z))
        feats = torch.cat(parts, dim=-1)
        return self.head(feats)                          # [S, C] logits
