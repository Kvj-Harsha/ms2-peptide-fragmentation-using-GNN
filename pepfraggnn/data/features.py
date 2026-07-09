"""Amino-acid vocabulary and physicochemical node features.

Fixes two documented bugs from the prototype:
  * an explicit UNK channel (the old code silently mapped unknown residues to
    an all-zero vector while the README claimed "20 AA + unknown");
  * physicochemical scalars motivated by the mobile-proton fragmentation model
    (basicity / proton affinity, hydrophobicity, mass, side-chain pKa).
"""
from __future__ import annotations

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_IDX = len(AMINO_ACIDS)          # index 20 -> the explicit UNK channel
NUM_AA_CHANNELS = len(AMINO_ACIDS) + 1  # 21: 20 canonical + UNK

# Basic residues that carry/retain protons (mobile-proton model).
BASIC_RESIDUES = set("RKH")
# Residues whose backbone chemistry produces well-known cleavage biases.
PROLINE = "P"
ACIDIC_RESIDUES = set("DE")

# Monoisotopic residue masses (Da). UNK falls back to the mean.
_RESIDUE_MASS = {
    "A": 71.037, "R": 156.101, "N": 114.043, "D": 115.027, "C": 103.009,
    "E": 129.043, "Q": 128.059, "G": 57.021, "H": 137.059, "I": 113.084,
    "L": 113.084, "K": 128.095, "M": 131.040, "F": 147.068, "P": 97.053,
    "S": 87.032, "T": 101.048, "W": 186.079, "Y": 163.063, "V": 99.068,
}
# Kyte-Doolittle hydrophobicity.
_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5, "Q": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Side-chain proton affinity proxy (gas-phase basicity, kJ/mol; 0 for non-basic).
_PROTON_AFFINITY = {
    "R": 1000.0, "K": 918.0, "H": 950.0,
}

# Number of physicochemical scalar features appended per residue.
NUM_PHYSCHEM = 6  # mass, hydrophobicity, proton_affinity, is_basic, is_pro, is_acidic


def _physchem_vector(aa: str) -> list[float]:
    mass = _RESIDUE_MASS.get(aa, 118.9)          # mean residue mass fallback
    hydro = _HYDROPHOBICITY.get(aa, 0.0)
    pa = _PROTON_AFFINITY.get(aa, 0.0)
    return [
        mass / 200.0,                            # scale roughly to [0, 1]
        hydro / 4.5,                             # scale to ~[-1, 1]
        pa / 1000.0,                             # scale to ~[0, 1]
        1.0 if aa in BASIC_RESIDUES else 0.0,
        1.0 if aa == PROLINE else 0.0,
        1.0 if aa in ACIDIC_RESIDUES else 0.0,
    ]


def aa_index(aa: str) -> int:
    """Return the vocabulary index for a residue, or UNK_IDX if non-canonical."""
    return AA_TO_IDX.get(aa, UNK_IDX)


def aa_one_hot(aa: str) -> torch.Tensor:
    """One-hot over 21 channels (20 canonical AAs + UNK)."""
    vec = torch.zeros(NUM_AA_CHANNELS)
    vec[aa_index(aa)] = 1.0
    return vec


def physchem_tensor(aa: str) -> torch.Tensor:
    return torch.tensor(_physchem_vector(aa), dtype=torch.float32)
