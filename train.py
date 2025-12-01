import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.pep_dataset import PepDataset
from src.model import PepFragGNN


# -----------------------------------------------------------
# One Epoch of Training
# -----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)

        x = batch.x
        edge_index = batch.edge_index
        graph_batch = batch.batch

        # batch.y is [batch, 1, 78] → squeeze to [batch, 78]
        targets = batch.y.squeeze(1)
        masks = batch.mask.squeeze(1)

        safe_targets = targets.clone()
        safe_targets[~masks] = 0.0  # replace invalid entries

        pred = model(x, edge_index, graph_batch)  # [batch, 78]

        loss_matrix = criterion(pred, safe_targets)  # per-element loss

        loss = (loss_matrix * masks).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ------------------------------
    # Load dataset
    # ------------------------------
    train_ds = PepDataset(split="train", num_rows=50000)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # ------------------------------
    # Model
    # ------------------------------
    model = PepFragGNN(
        in_dim=21,
        hidden_dim=64,
        num_layers=3,
        out_dim=78
    ).to(device)

    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------
    # Training Loop
    # ------------------------------
    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {loss:.4f}")

    # ------------------------------
    # Save model
    # ------------------------------
    torch.save(model.state_dict(), "fragment_gnn.pt")
    print("\nModel saved as fragment_gnn.pt")


if __name__ == "__main__":
    main()
