"""Pep2Prob loading with seeded, uniform-random subsampling.

Replaces the prototype's ``dataset[:N]`` first-N slice (a biased sample when the
data is ordered) with a fixed-seed uniform random subsample, and prefers the
dataset's own train/val/test splits when they exist.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _available_splits(hf_name: str, cache_dir: Optional[str]):
    from datasets import get_dataset_split_names
    try:
        return set(get_dataset_split_names(hf_name))
    except Exception:
        return set()


def load_split(hf_name: str, split: str, num_rows: Optional[int], seed: int,
               cache_dir: Optional[str] = None, val_fraction: float = 0.1):
    """Return a (possibly subsampled) HuggingFace dataset for ``split``.

    If the requested split is not published, val/test are carved
    deterministically out of ``train`` using ``seed`` so there is no leakage
    across the train/val/test boundary.
    """
    from datasets import load_dataset

    available = _available_splits(hf_name, cache_dir)

    if split in available:
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)
    elif "train" in available or not available:
        # Fall back to deterministically slicing the train split.
        full = load_dataset(hf_name, split="train", cache_dir=cache_dir)
        ds = _derive_split(full, split, seed, val_fraction)
    else:
        # Use whichever single split exists.
        only = sorted(available)[0]
        full = load_dataset(hf_name, split=only, cache_dir=cache_dir)
        ds = _derive_split(full, split, seed, val_fraction)

    if num_rows is not None and num_rows < len(ds):
        idx = _seeded_indices(len(ds), num_rows, seed)
        ds = ds.select(idx)
    return ds


def _derive_split(full, split: str, seed: int, val_fraction: float):
    """Deterministically partition ``full`` into train/val/test.

    test = last ``val_fraction`` block, val = preceding ``val_fraction`` block,
    train = the rest. A fixed-seed permutation makes the partition reproducible
    and order-independent.
    """
    n = len(full)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(val_fraction * n))
    n_val = int(round(val_fraction * n))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    chosen = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return full.select(sorted(int(i) for i in chosen))


def _seeded_indices(n: int, k: int, seed: int):
    """k unique row indices in [0, n), sampled uniformly with a fixed seed."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))
