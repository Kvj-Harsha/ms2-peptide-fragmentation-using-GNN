import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
from src.graph_builder import build_peptide_graph
from src.model import PepFragGNN


# ------------------------------------
# Load Model (Updated to match checkpoint)
# ------------------------------------
def load_model(device):

    # ✔ Match checkpoint architecture EXACTLY
    model = PepFragGNN(
        in_dim=21,
        hidden_dim=128,
        num_layers=4,
        out_dim=78
    ).to(device)

    try:
        state = torch.load("fragment_gnn.pt", map_location=device)
        model.load_state_dict(state, strict=True)
        print("\n✔ Model loaded successfully: fragment_gnn.pt")

    except Exception as e:
        print("\n❌ ERROR loading checkpoint!")
        print("Reason:", e)
        print("\nMake sure the model architecture matches the checkpoint.")
        raise e

    model.eval()
    return model


# ------------------------------------
# Run prediction
# ------------------------------------
def predict(peptide, model, device):
    print(f"\nPredicting for peptide: {peptide}")

    x, edge_index = build_peptide_graph(peptide)
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        pred = model(x, edge_index, batch)

    return pred.squeeze(0).cpu().tolist()


# ------------------------------------
# Decode 78 outputs → b1 / y1
# ------------------------------------
def decode_outputs(pred, pep_len):
    max_len = pep_len - 1  # Fragment positions

    b1 = pred[0:39][:max_len]
    y1 = pred[39:78][:max_len]

    return {"b1": b1, "y1": y1}


# ------------------------------------
# Main Program
# ------------------------------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load trained model
    model = load_model(device)

    # User input
    peptide = input("\nEnter peptide sequence: ").strip().upper()

    # Amino acid validation
    VALID = set("ACDEFGHIKLMNPQRSTVWY")
    if any(aa not in VALID for aa in peptide):
        raise ValueError(f"❌ Invalid amino acid found in: {peptide}")

    # Make prediction
    pred = predict(peptide, model, device)

    print("\n=== RAW 78 MODEL OUTPUTS ===")
    print(pred)

    decoded = decode_outputs(pred, len(peptide))

    print("\n=== FINAL B/Y Fragmentation Probabilities ===")
    for ion_type, values in decoded.items():
        print(f"{ion_type}: {values}")
