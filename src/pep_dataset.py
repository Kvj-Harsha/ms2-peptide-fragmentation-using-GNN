import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.dataset import load_pep2prob
from src.graph_builder import build_peptide_graph
from src.targets import extract_targets


class PepDataset(Dataset):

    def __init__(self, split="train", num_rows=None):
        """
        split: "train", "val", or "test"
        num_rows: limit number of samples for faster debugging
        """
        self.ds = load_pep2prob(split=split)
        self.num_rows = num_rows

    def __len__(self):
        if self.num_rows:
            return min(len(self.ds), self.num_rows)
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # --------------------------
        # 1) Build peptide graph
        # --------------------------
        peptide = row["peptide"]
        x, edge_index = build_peptide_graph(peptide)   # x: [L, 21], edges: [2, E]

        # --------------------------
        # 2) Extract targets + mask
        # --------------------------
        targets, mask = extract_targets(row)

        targets = torch.tensor(targets, dtype=torch.float32)  # [78]
        mask = torch.tensor(mask, dtype=torch.bool)           # [78]

        # expand to [1, 78] (batch dimension)
        targets = targets.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # --------------------------
        # 3) Return PyG Data object
        # --------------------------
        return Data(
            x=x,
            edge_index=edge_index,
            y=targets,       # [1, 78]
            mask=mask        # [1, 78]
        )
