"""Training and evaluation loops shared by the GNN models and the MLP baseline.

Handles both readout modes uniformly: it extracts (logits, targets, mask,
per-peptide grouping) from a batch, so the same loop trains the edge-level
CleavageGNN, the pooled ablation, and the Bag-of-AA baseline.
"""
from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from .losses import masked_spectral_angle_loss
from .metrics import MetricAccumulator


def unpack_batch(model_out, data, readout: str):
    """Return (logits, targets, mask, groups) flattened to [rows, channels].

    ``groups`` assigns each row to a peptide id (within the batch) so metrics and
    the spectral-angle loss can be computed per peptide.
    """
    if readout == "edge":
        logits = model_out                     # [S, C]
        targets = data.edge_y                  # [S, C]
        mask = data.edge_mask                  # [S, C]
        if data.cleave_index.numel() > 0:
            groups = data.batch[data.cleave_index[0]]
        else:
            groups = torch.zeros(0, dtype=torch.long, device=logits.device)
    elif readout == "pool":
        logits = model_out                     # [G, P]
        targets = data.y_pool                  # [G, P]
        mask = data.mask_pool                  # [G, P]
        groups = torch.arange(logits.size(0), device=logits.device)
    else:
        raise ValueError(f"Unknown readout '{readout}'.")
    return logits, targets, mask, groups


def train_one_epoch(model, loader, optimizer, loss_fn: Callable, device,
                    readout: str, cfg) -> float:
    model.train()
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = max(1, cfg.train.grad_accum_steps)
    total, n = 0.0, 0

    optimizer.zero_grad(set_to_none=True)
    for step, data in enumerate(tqdm(loader, desc="train", leave=False)):
        data = data.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(data)
            logits, targets, mask, groups = unpack_batch(out, data, readout)
            if mask.sum() == 0:
                continue
            if loss_fn is masked_spectral_angle_loss:
                loss = loss_fn(logits, targets, mask, groups)
            else:
                loss = loss_fn(logits, targets, mask)
            loss = loss / accum

        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * accum
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, readout: str) -> dict:
    """Compute benchmark metrics over a loader (predictions in probability space)."""
    model.eval()
    acc = MetricAccumulator()
    for data in loader:
        data = data.to(device)
        out = model(data)
        logits, targets, mask, groups = unpack_batch(out, data, readout)
        if mask.numel() == 0:
            continue
        probs = torch.sigmoid(logits)
        probs, targets, mask, groups = (
            probs.cpu(), targets.cpu(), mask.cpu(), groups.cpu())
        for g in torch.unique(groups):
            sel = groups == g
            acc.update(probs[sel].numpy(), targets[sel].numpy(), mask[sel].numpy())
    return acc.compute()


@torch.no_grad()
def evaluate_baseline(predict_fn, loader, device) -> dict:
    """Evaluate a callable baseline that returns pooled probabilities per batch."""
    acc = MetricAccumulator()
    for data in loader:
        data = data.to(device)
        probs = predict_fn(data).cpu()          # [G, P] in [0,1]
        targets = data.y_pool.cpu()
        mask = data.mask_pool.cpu()
        for i in range(probs.size(0)):
            acc.update(probs[i].numpy(), targets[i].numpy(), mask[i].numpy())
    return acc.compute()
