"""PooledGNN: the original global-mean-pool model, kept as the headline ablation.

Same backbone as CleavageGNN, but collapses the peptide to one vector and maps
it to the fixed-width [ion x position] output. This is deliberately the *weaker*
formulation (§2.1 of PROJECT_REVIEW.md); comparing it to CleavageGNN isolates
the value of the edge-level reformulation.
"""
from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from ..config import Config
from ..data.graph import node_feature_dim
from .backbones import GNNBackbone


class PooledGNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        h = cfg.model.hidden_dim
        self.backbone = GNNBackbone(
            in_dim=node_feature_dim(cfg.graph), hidden_dim=h,
            num_layers=cfg.model.num_layers, backbone=cfg.model.backbone,
            dropout=cfg.model.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(h, cfg.pooled_out_dim),
        )

    def forward(self, data):
        h = self.backbone(data.x, data.edge_index)
        g = global_mean_pool(h, data.batch)   # [G, H]
        return self.head(g)                   # [G, pooled_out_dim] logits
