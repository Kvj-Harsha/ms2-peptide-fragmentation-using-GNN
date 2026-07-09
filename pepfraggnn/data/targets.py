"""Target extraction with length-derived masking.

Pep2Prob stores, per precursor row, the empirical probability of each fragment
under a stringified tuple key like ``"('b', '1', '2')"`` = (ion, frag_charge,
fragment_position). We expose two views:

  * **edge targets** — for cleavage site ``k`` (1..L-1): the b-ion at position
    ``k`` and the y-ion at position ``L-k``. This is what CleavageGNN predicts.
  * **pooled targets** — the flat fixed-width [ion x pos] vector the original
    global-pool ablation predicts.

Masks are derived from **peptide length and explicit key presence**, never from
a magic ``-1.0`` sentinel.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..config import FRAG_CHARGES, ION_TYPES, MAX_CLEAVAGE_SITES


def _lookup(row: dict, ion: str, charge: str, pos: int) -> Optional[float]:
    """Return the probability for (ion, charge, pos) or None if absent.

    Handles the stringified-tuple column layout used by Pep2Prob and a couple of
    tolerant fallbacks so the loader is robust to minor schema differences.
    """
    for key in (
        str((ion, charge, str(pos))),
        str((ion, charge, pos)),
        f"{ion}{charge}_{pos}",
        f"{ion}_{charge}_{pos}",
    ):
        if key in row:
            val = row[key]
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def extract_edge_targets(row: dict, peptide: str):
    """Per-cleavage-site targets and mask.

    Returns
    -------
    targets : Tensor [L-1, C]   probabilities (0 where masked)
    mask    : Tensor [L-1, C]   True where a valid probability exists
    where C = len(ION_TYPES) * len(FRAG_CHARGES).
    """
    L = len(peptide)
    n_sites = max(L - 1, 0)
    channels = [(ion, ch) for ion in ION_TYPES for ch in FRAG_CHARGES]
    C = len(channels)

    targets = torch.zeros((n_sites, C), dtype=torch.float32)
    mask = torch.zeros((n_sites, C), dtype=torch.bool)

    for k in range(1, L):  # cleavage site k in 1..L-1
        site = k - 1
        for c, (ion, ch) in enumerate(channels):
            pos = k if ion == "b" else (L - k)  # y-ion position counts from C-term
            val = _lookup(row, ion, ch, pos)
            if val is not None:
                targets[site, c] = val
                mask[site, c] = True
    return targets, mask


def extract_pooled_targets(row: dict, peptide: str,
                           max_sites: int = MAX_CLEAVAGE_SITES):
    """Flat fixed-width targets/mask for the global-pool ablation model.

    Layout matches the original prototype: for each ion type, positions
    1..max_sites laid out contiguously. Mask is valid only where the position is
    <= L-1 (length-derived) AND the key is present.
    """
    L = len(peptide)
    out = []
    msk = []
    for ion in ION_TYPES:
        for ch in FRAG_CHARGES:
            for pos in range(1, max_sites + 1):
                val = _lookup(row, ion, ch, pos)
                valid = (pos <= L - 1) and (val is not None)
                out.append(val if (val is not None) else 0.0)
                msk.append(valid)
    targets = torch.tensor(out, dtype=torch.float32)
    mask = torch.tensor(msk, dtype=torch.bool)
    return targets, mask
