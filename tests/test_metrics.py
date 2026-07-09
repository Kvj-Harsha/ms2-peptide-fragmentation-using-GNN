import numpy as np

from pepfraggnn.metrics import MetricAccumulator, spectral_angle


def test_spectral_angle_perfect_match_is_one():
    v = np.array([0.1, 0.8, 0.3, 0.5])
    assert abs(spectral_angle(v, v) - 1.0) < 1e-6


def test_spectral_angle_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(spectral_angle(a, b) - 0.0) < 1e-6


def test_accumulator_masks_and_reduces():
    acc = MetricAccumulator()
    pred = np.array([0.9, 0.1, 0.5])
    true = np.array([1.0, 0.0, 0.0])
    mask = np.array([True, True, False])   # ignore the 3rd
    acc.update(pred, true, mask)
    m = acc.compute()
    assert m["n_peptides"] == 1
    # only 2 values counted: |0.9-1|+|0.1-0| = 0.2 over 2 -> 0.1
    assert abs(m["l1"] - 0.1) < 1e-6


def test_type2_presence_counts():
    acc = MetricAccumulator()
    pred = np.array([0.5, 0.0])   # present, absent
    true = np.array([0.5, 0.0])
    mask = np.array([True, True])
    acc.update(pred, true, mask)
    m = acc.compute()
    assert m["accuracy"] == 1.0
