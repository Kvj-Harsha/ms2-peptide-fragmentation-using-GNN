"""Single source of truth for all hyperparameters and run configuration.

Every script (train / evaluate / baselines / predict) builds a :class:`Config`
from the same defaults and the same argparse/YAML overrides, so the
train/eval architecture mismatch that plagued the original prototype
(train used 128/4, eval used 64/3) cannot recur.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml

# --------------------------------------------------------------------------- #
# Fragment / ion vocabulary
# --------------------------------------------------------------------------- #
ION_TYPES: List[str] = ["b", "y"]
# Fragment charges we model. Charge "1" reproduces the original scope; extend to
# ["1", "2"] to move toward the full Pep2Prob channel space.
FRAG_CHARGES: List[str] = ["1"]
# Maximum cleavage-site index (peptide length - 1) supported by the fixed-width
# pooled model. The edge-level model does not use this cap.
MAX_CLEAVAGE_SITES: int = 39


@dataclass
class DataConfig:
    """Dataset selection, splitting and subsampling."""

    hf_name: str = "bandeiralab/Pep2Prob"
    peptide_column: str = "peptide"
    charge_column: str = "charge"  # precursor charge column (best-effort)
    # Seeded uniform-random subsample sizes (None = use the whole split).
    train_rows: Optional[int] = 100_000
    val_rows: Optional[int] = 20_000
    test_rows: Optional[int] = 20_000
    # If the HF dataset only exposes a single split, we carve val/test out of it
    # deterministically using `seed`.
    val_fraction: float = 0.1
    cache_dir: Optional[str] = None


@dataclass
class GraphConfig:
    """Graph construction / node-edge featurisation switches."""

    use_physchem: bool = True          # append physicochemical scalars
    use_position: bool = True          # normalised residue position
    use_terminal_flags: bool = True    # is-N-term / is-C-term
    add_charge_carrier_edges: bool = True  # skip edges to basic residues R/K/H


@dataclass
class ModelConfig:
    """Backbone + readout hyperparameters (the ONLY place they are defined)."""

    readout: str = "edge"          # "edge" (CleavageGNN) or "pool" (ablation)
    backbone: str = "gcn"          # "gcn" | "gat" | "gin"
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    use_charge_embedding: bool = True
    max_precursor_charge: int = 8  # Pep2Prob spans 1..8


@dataclass
class TrainConfig:
    """Optimisation schedule."""

    epochs: int = 20
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    loss: str = "bce"             # "bce" (BCEWithLogits) | "mse" | "spectral_angle"
    grad_accum_steps: int = 1
    amp: bool = True              # mixed precision on CUDA
    early_stop_patience: int = 5  # epochs of no val-SA improvement
    num_workers: int = 0


@dataclass
class Config:
    """Top-level config aggregating every sub-config plus run bookkeeping."""

    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    out_dir: str = "runs/default"
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- derived / convenience ------------------------------------------- #
    @property
    def num_fragment_channels(self) -> int:
        """Fragments predicted per cleavage site (ions x fragment charges)."""
        return len(ION_TYPES) * len(FRAG_CHARGES)

    @property
    def pooled_out_dim(self) -> int:
        """Flat output width for the fixed-position pooled model."""
        return len(ION_TYPES) * len(FRAG_CHARGES) * MAX_CLEAVAGE_SITES

    # ---- (de)serialisation ------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
            out_dir=d.get("out_dir", "runs/default"),
            data=DataConfig(**d.get("data", {})),
            graph=GraphConfig(**d.get("graph", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        text = Path(path).read_text(encoding="utf-8")
        d = yaml.safe_load(text)
        return cls.from_dict(d)


def _flat_fields(prefix: str, dc) -> dict:
    """Return {dotted_key: (type, default)} for a dataclass instance."""
    out = {}
    for f in dataclasses.fields(dc):
        key = f"{prefix}{f.name}" if prefix else f.name
        out[key] = (f.type, getattr(dc, f.name))
    return out


def add_config_args(parser) -> None:
    """Register --data.train_rows, --model.hidden_dim, ... on an argparse parser.

    Nested dataclass fields become dotted flags. Booleans accept true/false.
    A --config PATH flag loads a YAML/JSON base that CLI flags then override.
    """
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML/JSON config to use as the base.")
    base = Config()
    groups = {
        "": base,
        "data.": base.data,
        "graph.": base.graph,
        "model.": base.model,
        "train.": base.train,
    }
    for prefix, dc in groups.items():
        for key, (_typ, default) in _flat_fields(prefix, dc).items():
            if dataclasses.is_dataclass(default):
                continue  # skip the sub-config containers on the root
            argname = f"--{key}"
            if isinstance(default, bool):
                parser.add_argument(argname, type=_str2bool, default=None)
            elif default is None:
                parser.add_argument(argname, type=str, default=None)
            elif isinstance(default, int):
                parser.add_argument(argname, type=int, default=None)
            elif isinstance(default, float):
                parser.add_argument(argname, type=float, default=None)
            else:
                parser.add_argument(argname, type=str, default=None)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def config_from_args(args) -> Config:
    """Build a Config from parsed args: optional YAML base + CLI overrides."""
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    sub = {"data": cfg.data, "graph": cfg.graph, "model": cfg.model, "train": cfg.train}
    for dest, value in vars(args).items():
        if dest == "config" or value is None:
            continue
        if "." in dest:
            group, name = dest.split(".", 1)
            if group in sub and hasattr(sub[group], name):
                setattr(sub[group], name, value)
        elif hasattr(cfg, dest):
            setattr(cfg, dest, value)
    return cfg
