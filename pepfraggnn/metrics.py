"""Pep2Prob benchmark metrics: Spectral Angle, L1, MSE, and type-2 accuracy.

These replace the prototype's inflated per-peptide Pearson (§2.5). Spectral
Angle is the benchmark's headline metric; we also keep per-peptide SA so the
training loop can early-stop on it and so downstream code can run paired
significance tests (CleavageGNN vs pooled, vs baselines).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

EPS = 1e-8
PRESENCE_TAU = 0.001  # type-2 presence threshold


def spectral_angle(pred: np.ndarray, true: np.ndarray) -> float:
    """1 - (2/pi) * arccos(cosine(pred, true)); higher is better, in [0, 1]."""
    pn = np.linalg.norm(pred)
    tn = np.linalg.norm(true)
    if pn < EPS or tn < EPS:
        return 0.0
    cos = float(np.dot(pred, true) / (pn * tn))
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - (2.0 / np.pi) * np.arccos(cos)


@dataclass
class MetricAccumulator:
    """Accumulate per-peptide predictions/targets and reduce to benchmark metrics.

    Feed already-probability-space predictions (post-sigmoid) together with the
    boolean validity mask, one peptide at a time.
    """

    per_peptide_sa: List[float] = field(default_factory=list)
    _abs_err_sum: float = 0.0
    _sq_err_sum: float = 0.0
    _n_values: int = 0
    # type-2 confusion counts
    _tp: int = 0
    _tn: int = 0
    _fp: int = 0
    _fn: int = 0

    def update(self, pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        true = np.asarray(true, dtype=np.float64).reshape(-1)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.sum() == 0:
            return
        p = pred[mask]
        t = true[mask]

        self.per_peptide_sa.append(spectral_angle(p, t))

        err = p - t
        self._abs_err_sum += float(np.abs(err).sum())
        self._sq_err_sum += float((err ** 2).sum())
        self._n_values += int(mask.sum())

        p_present = p > PRESENCE_TAU
        t_present = t > PRESENCE_TAU
        self._tp += int(np.sum(p_present & t_present))
        self._tn += int(np.sum(~p_present & ~t_present))
        self._fp += int(np.sum(p_present & ~t_present))
        self._fn += int(np.sum(~p_present & t_present))

    def compute(self) -> Dict[str, float]:
        sa = np.asarray(self.per_peptide_sa, dtype=np.float64)
        n = max(self._n_values, 1)
        total = self._tp + self._tn + self._fp + self._fn
        tp, tn, fp, fn = self._tp, self._tn, self._fp, self._fn
        return {
            "spectral_angle": float(sa.mean()) if sa.size else 0.0,
            "spectral_angle_std": float(sa.std()) if sa.size else 0.0,
            "l1": self._abs_err_sum / n,
            "mse": self._sq_err_sum / n,
            "accuracy": (tp + tn) / total if total else 0.0,
            "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "n_peptides": int(sa.size),
        }


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"SA={m['spectral_angle']:.4f}±{m['spectral_angle_std']:.4f}  "
        f"L1={m['l1']:.4f}  MSE={m['mse']:.4f}  "
        f"Acc={m['accuracy']:.4f}  Sens={m['sensitivity']:.4f}  "
        f"Spec={m['specificity']:.4f}  (n={m['n_peptides']})"
    )
