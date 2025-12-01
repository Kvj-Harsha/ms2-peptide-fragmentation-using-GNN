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

        targets = batch.y.squeeze(1)
        masks = batch.mask.squeeze(1)

        safe_targets = targets.clone()
        safe_targets[~masks] = 0.0

        pred = model(x, edge_index, graph_batch)

        loss_matrix = criterion(pred, safe_targets)
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
    # Load TRAIN dataset (full)
    # ------------------------------
    full_train = PepDataset(split="train", num_rows=150000)

    # ------------------------------
    # Manual train/val split
    # ------------------------------
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # ------------------------------
    # Model
    # ------------------------------
    model = PepFragGNN(
        in_dim=21,
        hidden_dim=128,
        num_layers=4,
        out_dim=78
    ).to(device)

    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=1e-5
    )

    # ------------------------------
    # Train
    # ------------------------------
    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {loss:.4f}")

    # ------------------------------
    # Save
    # ------------------------------
    torch.save(model.state_dict(), "fragment_gnn.pt")
    print("\nModel saved as fragment_gnn.pt")


if __name__ == "__main__":
    main()
