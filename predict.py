import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
from src.graph_builder import build_peptide_graph
from src.model import PepFragGNN


# -------------------------------
# Load Model
# -------------------------------
def load_model(device):
    model = PepFragGNN(
        in_dim=21,
        hidden_dim=64,
        num_layers=3,
        out_dim=78
    ).to(device)

    model.load_state_dict(torch.load("fragment_gnn.pt", map_location=device))
    model.eval()
    print("Loaded model: fragment_gnn.pt")
    return model


# -------------------------------
# Run prediction
# -------------------------------
def predict(peptide, model, device):
    print(f"\nPredicting for peptide: {peptide}")

    # Build graph from peptide
    x, edge_index = build_peptide_graph(peptide)

    x = x.to(device)
    edge_index = edge_index.to(device)

    batch = torch.zeros(x.shape[0], dtype=torch.long).to(device)

    with torch.no_grad():
        pred = model(x, edge_index, batch)   # shape: [1, 78]

    pred = pred.squeeze(0).cpu().tolist()
    return pred


# -------------------------------
# Correct decoding for 78 outputs
# -------------------------------
def decode_outputs(pred, pep_len):
    """
    Correct 78-dim mapping:
        b1: positions 1..39   → pred[0:39]
        y1: positions 1..39   → pred[39:78]

    Real valid positions = peptide_length - 1
    """

    max_len = pep_len - 1  # e.g., peptide length 11 → 10 positions

    # slice and limit to actual peptide length
    b1 = pred[0:39][:max_len]
    y1 = pred[39:78][:max_len]

    return {
        "b1": b1,
        "y1": y1
    }


# -------------------------------
# Main Program
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(device)

    peptide = input("Enter peptide sequence: ").strip().upper()

    pred = predict(peptide, model, device)

    print("\n=== RAW 78 OUTPUTS ===")
    print(pred)

    decoded = decode_outputs(pred, len(peptide))

    print("\n=== FINAL B/Y Frag Probabilities ===")
    for k, v in decoded.items():
        print(f"{k}: {v}")
