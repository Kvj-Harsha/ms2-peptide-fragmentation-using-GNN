"""Global baseline: predict the per-channel training-set mean profile.

This is the "does the graph do anything" floor (§2.2). If the GNN barely beats
it, the structure adds nothing. Operates in the fixed-width pooled target space.
"""
from __future__ import annotations

import torch


class GlobalMeanBaseline:
    def __init__(self, out_dim: int):
        self.out_dim = out_dim
        self.mean_profile = torch.zeros(out_dim)

    def fit(self, loader) -> "GlobalMeanBaseline":
        """Masked per-channel mean over all training peptides."""
        total = torch.zeros(self.out_dim, dtype=torch.float64)
        count = torch.zeros(self.out_dim, dtype=torch.float64)
        for batch in loader:
            y = batch.y_pool.double()          # [G, out_dim]
            m = batch.mask_pool.double()       # [G, out_dim]
            total += (y * m).sum(dim=0)
            count += m.sum(dim=0)
        self.mean_profile = (total / count.clamp(min=1)).float()
        return self

    @torch.no_grad()
    def predict(self, batch) -> torch.Tensor:
        """Broadcast the mean profile to every peptide in the batch."""
        g = batch.y_pool.size(0)
        return self.mean_profile.unsqueeze(0).expand(g, -1).clone()
