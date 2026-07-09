import torch

from pepfraggnn.config import GraphConfig
from pepfraggnn.data.features import NUM_AA_CHANNELS, UNK_IDX, aa_index
from pepfraggnn.data.graph import build_peptide_graph, node_feature_dim


def test_single_residue_no_crash():
    # The prototype divided by (L-1) and produced a malformed edge_index here.
    g = build_peptide_graph("A", GraphConfig())
    assert g.x.shape[0] == 1
    assert g.edge_index.shape == (2, 0)
    assert g.cleave_index.shape == (2, 0)


def test_feature_dim_matches_config():
    cfg = GraphConfig(use_physchem=True, use_position=True, use_terminal_flags=True)
    g = build_peptide_graph("PEPTIDE", cfg)
    assert g.x.shape[1] == node_feature_dim(cfg)


def test_unknown_residue_maps_to_unk_channel():
    assert aa_index("X") == UNK_IDX
    g = build_peptide_graph("AXA", GraphConfig(use_physchem=False,
                                               use_position=False,
                                               use_terminal_flags=False))
    # middle residue is UNK -> one-hot fires the UNK channel
    assert g.x[1, UNK_IDX] == 1.0
    assert g.x.shape[1] == NUM_AA_CHANNELS


def test_cleavage_sites_and_backbone_edges():
    g = build_peptide_graph("PEPTIDE", GraphConfig(add_charge_carrier_edges=False))
    L = 7
    assert g.cleave_index.shape == (2, L - 1)
    # backbone: 2*(L-1) directed edges
    assert g.edge_index.shape[1] == 2 * (L - 1)
    assert torch.equal(g.cleave_index[0], torch.arange(0, L - 1))


def test_charge_carrier_edges_added_for_basic_residues():
    no_skip = build_peptide_graph("AAKAA", GraphConfig(add_charge_carrier_edges=False))
    skip = build_peptide_graph("AAKAA", GraphConfig(add_charge_carrier_edges=True))
    assert skip.edge_index.shape[1] > no_skip.edge_index.shape[1]
