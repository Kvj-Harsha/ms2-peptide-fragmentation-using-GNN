"""Reproduce the Global and Bag-of-AA baselines with benchmark metrics.

    python scripts/run_baselines.py --out_dir runs/baselines
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from pepfraggnn.baselines import BagOfAAMLP, GlobalMeanBaseline
from pepfraggnn.config import add_config_args, config_from_args
from pepfraggnn.data.dataset import Pep2ProbDataset
from pepfraggnn.engine import evaluate_baseline
from pepfraggnn.losses import masked_bce_with_logits
from pepfraggnn.metrics import format_metrics
from pepfraggnn.seed import resolve_device, set_seed
from pepfraggnn.utils import append_csv_row, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Run reference baselines.")
    add_config_args(parser)
    parser.add_argument("--results_csv", default="results/baseline_results.csv")
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    P = cfg.pooled_out_dim

    train_ds = Pep2ProbDataset(cfg, "train")
    test_ds = Pep2ProbDataset(cfg, "test")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)

    # ---- Global mean profile -------------------------------------------- #
    print("Fitting Global baseline...")
    glob = GlobalMeanBaseline(P).fit(train_loader)
    m = evaluate_baseline(lambda d: glob.predict(d).to(device), test_loader, device)
    print(f"[Global]      {format_metrics(m)}")
    append_csv_row(args.results_csv, {"model": "global", "params": 0,
                                      "seed": cfg.seed, **_round(m)})

    # ---- Bag-of-AA MLP -------------------------------------------------- #
    print("Training Bag-of-AA MLP...")
    mlp = BagOfAAMLP(P, hidden_dim=cfg.model.hidden_dim,
                     max_charge=cfg.model.max_precursor_charge).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    for epoch in range(1, cfg.train.epochs + 1):
        mlp.train()
        for data in train_loader:
            data = data.to(device)
            logits = mlp(data)
            loss = masked_bce_with_logits(logits, data.y_pool, data.mask_pool)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    total, _ = count_parameters(mlp)
    mlp.eval()
    m = evaluate_baseline(lambda d: torch.sigmoid(mlp(d)), test_loader, device)
    print(f"[Bag-of-AA]   {format_metrics(m)}")
    append_csv_row(args.results_csv, {"model": "bag_of_aa", "params": total,
                                      "seed": cfg.seed, **_round(m)})
    print(f"Baseline results written to {args.results_csv}")


def _round(m):
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}


if __name__ == "__main__":
    main()
