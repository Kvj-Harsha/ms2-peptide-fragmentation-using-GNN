"""Peptide -> graph construction.

Fixes the prototype's edge cases (L=1 division-by-zero, empty edge_index,
missing UNK handling) and enriches the graph with physicochemical features and
optional charge-carrier skip edges. Also emits an explicit ``cleave_index``
mapping each backbone cleavage site to its (left, right) residue nodes, which
the edge-level readout consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import GraphConfig
from .features import (
    BASIC_RESIDUES,
    NUM_AA_CHANNELS,
    NUM_PHYSCHEM,
    aa_one_hot,
    physchem_tensor,
)


def node_feature_dim(graph_cfg: GraphConfig) -> int:
    """Width of each node feature vector for the given graph config."""
    dim = NUM_AA_CHANNELS
    if graph_cfg.use_physchem:
        dim += NUM_PHYSCHEM
    if graph_cfg.use_position:
        dim += 1
    if graph_cfg.use_terminal_flags:
        dim += 2
    return dim


@dataclass
class PeptideGraph:
    x: torch.Tensor              # [L, F] node features
    edge_index: torch.Tensor     # [2, E] undirected edges
    cleave_index: torch.Tensor   # [2, L-1] (left_node, right_node) per cleavage site


def build_peptide_graph(peptide: str, graph_cfg: Optional[GraphConfig] = None) -> PeptideGraph:
    """Build a residue graph for a peptide string.

    Node features (concatenated in this order):
      AA one-hot (21) [+ physchem (6)] [+ normalised position (1)]
      [+ terminal flags (2)].
    Edges: backbone i<->i+1, optionally plus charge-carrier skip edges linking
    every basic residue (R/K/H) to all backbone nodes.
    """
    cfg = graph_cfg or GraphConfig()
    L = len(peptide)
    if L == 0:
        raise ValueError("Cannot build a graph for an empty peptide.")

    feats = []
    for i, aa in enumerate(peptide):
        parts = [aa_one_hot(aa)]
        if cfg.use_physchem:
            parts.append(physchem_tensor(aa))
        if cfg.use_position:
            # L==1 guard: a single residue sits at position 0.0.
            pos = 0.0 if L == 1 else i / (L - 1)
            parts.append(torch.tensor([pos], dtype=torch.float32))
        if cfg.use_terminal_flags:
            parts.append(torch.tensor(
                [1.0 if i == 0 else 0.0, 1.0 if i == L - 1 else 0.0],
                dtype=torch.float32,
            ))
        feats.append(torch.cat(parts, dim=0))
    x = torch.stack(feats)  # [L, F]

    # ---- edges ----------------------------------------------------------- #
    edges: list[list[int]] = []
    for i in range(L - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    if cfg.add_charge_carrier_edges and L > 1:
        basic_nodes = [i for i, aa in enumerate(peptide) if aa in BASIC_RESIDUES]
        for b in basic_nodes:
            for j in range(L):
                if j != b and abs(j - b) > 1:  # skip self and existing backbone
                    edges.append([b, j])
                    edges.append([j, b])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Single-residue peptide: no edges. Keep a well-formed [2, 0] tensor.
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ---- cleavage-site index (left, right) per backbone bond ------------- #
    if L > 1:
        left = torch.arange(0, L - 1, dtype=torch.long)
        right = torch.arange(1, L, dtype=torch.long)
        cleave_index = torch.stack([left, right], dim=0)  # [2, L-1]
    else:
        cleave_index = torch.zeros((2, 0), dtype=torch.long)

    return PeptideGraph(x=x, edge_index=edge_index, cleave_index=cleave_index)
