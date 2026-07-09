"""Model definitions: GNN backbones and edge-level / pooled readouts."""
from ..config import Config
from .cleavage_gnn import CleavageGNN
from .pooled_gnn import PooledGNN


def build_model(cfg: Config):
    """Instantiate the model selected by ``cfg.model.readout``."""
    if cfg.model.readout == "edge":
        return CleavageGNN(cfg)
    if cfg.model.readout == "pool":
        return PooledGNN(cfg)
    raise ValueError(f"Unknown readout '{cfg.model.readout}' (use 'edge' or 'pool').")


__all__ = ["CleavageGNN", "PooledGNN", "build_model"]
