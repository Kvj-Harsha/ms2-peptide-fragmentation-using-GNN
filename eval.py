import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from src.pep_dataset import PepDataset
from src.model import PepFragGNN


# -------------------------------
# Compute Pearson correlation
# -------------------------------
def batch_pearson(pred, target, mask):
    pred = pred[mask].cpu().numpy()
    target = target[mask].cpu().numpy()

    if len(pred) < 3:
        return 0.0  # not enough data

    try:
        r, _ = pearsonr(pred, target)
        return r
    except:
        return 0.0


# -------------------------------
# Evaluate model on dataset
# -------------------------------
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    pearsons = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y
            masks = batch.mask

            safe_targets = targets.clone()
            safe_targets[~masks] = 0.0

            loss_matrix = criterion(pred, safe_targets)
            loss = (loss_matrix * masks).sum() / masks.sum()
            total_loss += loss.item()

            # Compute Pearson correlation for each graph
            for i in range(pred.size(0)):
                pearsons.append(batch_pearson(pred[i], targets[i], masks[i]))

    return total_loss / len(loader), np.mean(pearsons)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load test dataset
    test_ds = PepDataset(split="test", num_rows=5000)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load model
    model = PepFragGNN(in_dim=21, hidden_dim=64, num_layers=3, out_dim=78).to(device)
    model.load_state_dict(torch.load("fragment_gnn.pt", map_location=device))

    criterion = nn.BCELoss(reduction="none")

    print("\nEvaluating model on test split...\n")

    test_loss, pearson = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Pearson Correlation: {pearson:.4f}")
