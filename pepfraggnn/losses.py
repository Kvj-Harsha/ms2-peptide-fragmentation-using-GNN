"""Masked losses operating on logits with soft probabilistic targets."""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def masked_bce_with_logits(logits, targets, mask):
    """BCEWithLogits (numerically stable) averaged over valid entries only."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return (loss * m).sum() / denom


def masked_mse(logits, targets, mask):
    pred = torch.sigmoid(logits)
    loss = (pred - targets) ** 2
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def masked_spectral_angle_loss(logits, targets, mask, groups):
    """1 - SA averaged per peptide.

    ``groups`` maps each row of ``logits`` to a peptide id so the cosine is taken
    over each peptide's own fragments (SA is a per-spectrum quantity).
    """
    pred = torch.sigmoid(logits) * mask.float()
    tgt = targets * mask.float()
    total = 0.0
    uniq = torch.unique(groups)
    for g in uniq:
        sel = groups == g
        p = pred[sel].reshape(-1)
        t = tgt[sel].reshape(-1)
        pn = p.norm().clamp(min=EPS)
        tn = t.norm().clamp(min=EPS)
        cos = torch.clamp((p @ t) / (pn * tn), -1.0, 1.0)
        sa = 1.0 - (2.0 / torch.pi) * torch.arccos(cos)
        total = total + (1.0 - sa)
    return total / len(uniq)


def build_loss(name: str):
    name = name.lower()
    if name == "bce":
        return masked_bce_with_logits
    if name == "mse":
        return masked_mse
    if name in ("spectral_angle", "sa"):
        return masked_spectral_angle_loss
    raise ValueError(f"Unknown loss '{name}' (use bce|mse|spectral_angle).")
