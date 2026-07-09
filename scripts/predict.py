"""Predict per-cleavage-site b/y fragment probabilities for a peptide.

    python scripts/predict.py --ckpt runs/cleavage_gcn/model.pt --peptide PEPTIDER
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from pepfraggnn.config import Config
from pepfraggnn.data.dataset import make_synthetic_dataset
from pepfraggnn.engine import unpack_batch
from pepfraggnn.models import build_model
from pepfraggnn.seed import resolve_device

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def predict_peptide(model, cfg, peptide: str, device):
    ds = make_synthetic_dataset([peptide], cfg=cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            logits, _, _, _ = unpack_batch(out, data, cfg.model.readout)
            return torch.sigmoid(logits).cpu()  # [S, C] for edge readout


def main():
    parser = argparse.ArgumentParser(description="Predict fragment probabilities.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--peptide", required=True)
    args = parser.parse_args()

    peptide = args.peptide.strip().upper()
    invalid = sorted(set(peptide) - VALID_AA)
    if invalid:
        print(f"Warning: non-canonical residues treated as UNK: {invalid}")

    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = Config.from_dict(payload["config"])
    device = resolve_device(cfg.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["state_dict"])

    probs = predict_peptide(model, cfg, peptide, device)
    if cfg.model.readout != "edge":
        print("Note: this checkpoint uses the pooled readout; showing raw output.")
        print(probs.tolist())
        return

    L = len(peptide)
    print(f"\nPeptide: {peptide}  (length {L})")
    print(f"{'site':>4}  {'bond':>7}  {'b-ion':>7}  {'y-ion':>7}")
    for k in range(1, L):
        b = probs[k - 1, 0].item()
        y = probs[k - 1, 1].item()
        bond = f"{peptide[k-1]}|{peptide[k]}"
        print(f"{k:>4}  {bond:>7}  {b:>7.4f}  {y:>7.4f}")


if __name__ == "__main__":
    main()
