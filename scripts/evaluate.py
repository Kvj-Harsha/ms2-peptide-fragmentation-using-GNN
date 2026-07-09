"""Evaluate a trained checkpoint on the test split with benchmark metrics.

    python scripts/evaluate.py --ckpt runs/cleavage_gcn/model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from pepfraggnn.config import Config
from pepfraggnn.data.dataset import Pep2ProbDataset
from pepfraggnn.engine import evaluate
from pepfraggnn.metrics import format_metrics
from pepfraggnn.models import build_model
from pepfraggnn.seed import resolve_device, set_seed
from pepfraggnn.utils import append_csv_row, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--results_csv", default="results/main_results.csv")
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = Config.from_dict(payload["config"])
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    model = build_model(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    total, _ = count_parameters(model)

    ds = Pep2ProbDataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device, cfg.model.readout)

    print(f"[{args.split}] {format_metrics(metrics)}")
    append_csv_row(args.results_csv, {
        "model": f"{cfg.model.readout}-{cfg.model.backbone}",
        "params": total,
        "split": args.split,
        "seed": cfg.seed,
        **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()},
    })
    print(f"Appended results to {args.results_csv}")


if __name__ == "__main__":
    main()
