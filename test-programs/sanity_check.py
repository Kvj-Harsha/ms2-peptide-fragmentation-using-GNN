import torch

from src.dataset import load_pep2prob
from src.targets import extract_targets
from src.graph_builder import build_peptide_graph
from src.model import PepFragGNN

def main():
    ds = load_pep2prob("train")
    row = ds[0]

    # 1) Targets
    target, mask = extract_targets(row)
    print("Targets OK:", len(target), len(mask))

    # 2) Graph
    peptide = row["peptide"]
    node_features, edge_index = build_peptide_graph(peptide)

    print("Peptide:", peptide)
    print("Node features shape:", node_features.shape)
    print("Edge index shape:", edge_index.shape)

    # 3) Prepare batch tensor (only 1 graph → batch = zeros)
    batch = torch.zeros(node_features.shape[0], dtype=torch.long)

    # 4) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PepFragGNN().to(device)

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    pred = model(node_features, edge_index, batch)  # [1, 78]
    print("Prediction shape:", pred.shape)

if __name__ == "__main__":
    main()
