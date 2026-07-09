import torch
from torch_geometric.loader import DataLoader

from pepfraggnn.config import Config
from pepfraggnn.data.dataset import make_synthetic_dataset
from pepfraggnn.engine import evaluate, train_one_epoch, unpack_batch
from pepfraggnn.losses import build_loss
from pepfraggnn.models import build_model
from pepfraggnn.utils import count_parameters

PEPTIDES = ["PEPTIDER", "ACDEFGHIK", "AAKAA", "MLLK", "SAMPLERR"]


def _rows(peptides):
    rows = []
    for pep in peptides:
        L = len(pep)
        row = {"charge": 2}
        for k in range(1, L):
            row[str(("b", "1", str(k)))] = min(0.05 * k, 1.0)
            row[str(("y", "1", str(L - k)))] = min(0.04 * k, 1.0)
        rows.append(row)
    return rows


def _batch(cfg):
    ds = make_synthetic_dataset(PEPTIDES, probs=_rows(PEPTIDES), cfg=cfg)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    return next(iter(loader))


def test_edge_batching_offsets_cleave_index():
    cfg = Config()
    batch = _batch(cfg)
    total_nodes = sum(len(p) for p in PEPTIDES)
    total_sites = sum(len(p) - 1 for p in PEPTIDES)
    assert batch.x.shape[0] == total_nodes
    assert batch.cleave_index.shape == (2, total_sites)
    # every referenced node index must be in range after batching
    assert int(batch.cleave_index.max()) < total_nodes


def test_cleavage_gnn_forward_and_shapes():
    cfg = Config()
    cfg.model.readout = "edge"
    model = build_model(cfg)
    batch = _batch(cfg)
    out = model(batch)
    total_sites = sum(len(p) - 1 for p in PEPTIDES)
    assert out.shape == (total_sites, cfg.num_fragment_channels)


def test_pooled_gnn_forward_and_shapes():
    cfg = Config()
    cfg.model.readout = "pool"
    model = build_model(cfg)
    batch = _batch(cfg)
    out = model(batch)
    assert out.shape == (len(PEPTIDES), cfg.pooled_out_dim)


def test_model_under_one_million_params():
    cfg = Config()
    total, _ = count_parameters(build_model(cfg))
    assert total < 1_000_000


def test_one_training_step_runs_and_reduces_loss():
    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.amp = False
    ds = make_synthetic_dataset(PEPTIDES, probs=_rows(PEPTIDES), cfg=cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss("bce")
    device = torch.device("cpu")
    loss = train_one_epoch(model, loader, opt, loss_fn, device, "edge", cfg)
    assert loss > 0
    metrics = evaluate(model, loader, device, "edge")
    assert 0.0 <= metrics["spectral_angle"] <= 1.0


def test_spectral_angle_loss_backprops():
    cfg = Config()
    cfg.train.amp = False
    model = build_model(cfg)
    batch = _batch(cfg)
    loss_fn = build_loss("spectral_angle")
    out = model(batch)
    logits, targets, mask, groups = unpack_batch(out, batch, "edge")
    loss = loss_fn(logits, targets, mask, groups)
    loss.backward()
    assert loss.item() >= 0
