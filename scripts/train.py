"""Train CleavageGNN (or the pooled ablation) on Pep2Prob.

Single source of truth for hyperparameters is pepfraggnn.config. Example:

    python scripts/train.py --model.readout edge --model.backbone gcn \
        --train.epochs 20 --out_dir runs/cleavage_gcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from pepfraggnn.config import add_config_args, config_from_args
from pepfraggnn.data.dataset import Pep2ProbDataset
from pepfraggnn.engine import evaluate, train_one_epoch
from pepfraggnn.losses import build_loss
from pepfraggnn.metrics import format_metrics
from pepfraggnn.models import build_model
from pepfraggnn.seed import resolve_device, set_seed
from pepfraggnn.utils import count_parameters


def main():
    parser = argparse.ArgumentParser(description="Train a PepFragGNN model.")
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args())

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")
    print(f"Device: {device} | readout={cfg.model.readout} backbone={cfg.model.backbone}")

    train_ds = Pep2ProbDataset(cfg, "train")
    val_ds = Pep2ProbDataset(cfg, "val")
    print(f"Train peptides: {len(train_ds)} | Val peptides: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers)

    model = build_model(cfg).to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total ({trainable:,} trainable)")

    loss_fn = build_loss(cfg.train.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=1e-5)

    best_sa = -1.0
    patience = 0
    ckpt_path = out_dir / "model.pt"
    for epoch in range(1, cfg.train.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                               device, cfg.model.readout, cfg)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, cfg.model.readout)
        print(f"Epoch {epoch:02d}/{cfg.train.epochs} | loss={loss:.4f} | "
              f"val {format_metrics(val_metrics)}")

        if val_metrics["spectral_angle"] > best_sa:
            best_sa = val_metrics["spectral_angle"]
            patience = 0
            torch.save({"state_dict": model.state_dict(),
                        "config": cfg.to_dict(),
                        "val_spectral_angle": best_sa}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val SA={best_sa:.4f}).")
                break

    print(f"\nBest val SA={best_sa:.4f}. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
