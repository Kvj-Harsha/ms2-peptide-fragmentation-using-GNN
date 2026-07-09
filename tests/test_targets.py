from pepfraggnn.data.targets import extract_edge_targets, extract_pooled_targets


def _row(peptide):
    # b_k at ('b','1',k); y_{L-k} at ('y','1',L-k). Build a fully-specified row.
    L = len(peptide)
    row = {}
    for k in range(1, L):
        row[str(("b", "1", str(k)))] = 0.1 * k
        row[str(("y", "1", str(L - k)))] = 0.2 * k
    return row


def test_edge_targets_shape_and_mask():
    pep = "PEPTIDE"          # L=7 -> 6 cleavage sites
    row = _row(pep)
    t, m = extract_edge_targets(row, pep)
    assert t.shape == (6, 2)
    assert m.all()          # every b/y present
    # b-ion channel 0 at site k-1 should equal 0.1*k
    assert abs(t[0, 0].item() - 0.1) < 1e-6
    assert abs(t[5, 0].item() - 0.6) < 1e-6


def test_missing_keys_are_masked_not_sentinel():
    pep = "PEPTIDE"
    row = {}                # nothing present
    t, m = extract_edge_targets(row, pep)
    assert not m.any()      # all masked
    assert (t == 0).all()   # zeros, never -1.0 sentinel


def test_pooled_targets_length_derived_mask():
    pep = "ACD"             # L=3 -> valid positions 1,2 per ion
    row = _row(pep)
    t, m = extract_pooled_targets(row, pep, max_sites=39)
    # 2 ions * 39 positions = 78 slots; only 2 valid per ion
    assert t.shape[0] == 78
    assert int(m.sum()) == 4
