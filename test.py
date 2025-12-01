from src.pep_dataset import PepDataset
from torch_geometric.loader import DataLoader

ds = PepDataset(num_rows=4)
loader = DataLoader(ds, batch_size=2)

batch = next(iter(loader))
print("Batch y shape:", batch.y.shape)
print("Batch mask shape:", batch.mask.shape)