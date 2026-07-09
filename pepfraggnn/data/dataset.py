"""PyG dataset wrapping Pep2Prob rows as peptide graphs + targets."""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data, Dataset

from ..config import Config
from .graph import build_peptide_graph
from .loader import load_split
from .targets import extract_edge_targets, extract_pooled_targets


class PeptideData(Data):
    """Data object with correct batch-increment for the cleavage-site index."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "cleave_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "cleave_index":
            return -1  # [2, n_sites] -> concat along sites
        return super().__cat_dim__(key, value, *args, **kwargs)


def _read_charge(row: dict, column: str, max_charge: int) -> int:
    raw = row.get(column, None)
    try:
        z = int(raw)
    except (TypeError, ValueError):
        z = 2  # tryptic-peptide default when charge is unavailable
    return max(1, min(z, max_charge))


class Pep2ProbDataset(Dataset):
    """Builds peptide graphs and both edge-level and pooled targets per row."""

    def __init__(self, cfg: Config, split: str):
        super().__init__()
        self.cfg = cfg
        self.split = split
        rows = {"train": cfg.data.train_rows,
                "val": cfg.data.val_rows,
                "test": cfg.data.test_rows}[split]
        self.ds = load_split(
            hf_name=cfg.data.hf_name,
            split=split,
            num_rows=rows,
            seed=cfg.seed,
            cache_dir=cfg.data.cache_dir,
            val_fraction=cfg.data.val_fraction,
        )

    def len(self) -> int:  # PyG Dataset API
        return len(self.ds)

    def get(self, idx: int) -> PeptideData:
        row = self.ds[idx]
        peptide = str(row[self.cfg.data.peptide_column]).strip().upper()

        g = build_peptide_graph(peptide, self.cfg.graph)
        edge_y, edge_mask = extract_edge_targets(row, peptide)
        pool_y, pool_mask = extract_pooled_targets(row, peptide)
        charge = _read_charge(row, self.cfg.data.charge_column,
                              self.cfg.model.max_precursor_charge)

        return PeptideData(
            x=g.x,
            edge_index=g.edge_index,
            cleave_index=g.cleave_index,
            edge_y=edge_y,
            edge_mask=edge_mask,
            y_pool=pool_y.unsqueeze(0),
            mask_pool=pool_mask.unsqueeze(0),
            charge=torch.tensor([charge], dtype=torch.long),
            num_nodes=g.x.size(0),
        )


def make_synthetic_dataset(peptides, probs=None, cfg: Optional[Config] = None):
    """Build an in-memory list of PeptideData from raw peptide strings.

    Used by the unit tests and by ``predict`` so the pipeline can run without
    downloading Pep2Prob. ``probs`` is an optional list of row-dicts of the same
    key format as the real dataset; when omitted, targets are zero and masked.
    """
    cfg = cfg or Config()
    out = []
    for i, pep in enumerate(peptides):
        pep = pep.strip().upper()
        row = probs[i] if probs is not None else {}
        g = build_peptide_graph(pep, cfg.graph)
        edge_y, edge_mask = extract_edge_targets(row, pep)
        pool_y, pool_mask = extract_pooled_targets(row, pep)
        z = _read_charge(row, cfg.data.charge_column, cfg.model.max_precursor_charge)
        out.append(PeptideData(
            x=g.x, edge_index=g.edge_index, cleave_index=g.cleave_index,
            edge_y=edge_y, edge_mask=edge_mask,
            y_pool=pool_y.unsqueeze(0), mask_pool=pool_mask.unsqueeze(0),
            charge=torch.tensor([z], dtype=torch.long),
            num_nodes=g.x.size(0),
        ))
    return out
