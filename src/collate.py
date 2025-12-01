import torch
from torch_geometric.data import Batch

def collate_fn(batch):
    node_features_list = []
    edge_index_list = []
    batch_index_list = []

    targets = []
    masks = []

    node_offset = 0

    for graph_id, (node_features, edge_index, target, mask) in enumerate(batch):
        node_features_list.append(node_features)

        # Shift edges for batching
        ei = edge_index + node_offset
        edge_index_list.append(ei)

        num_nodes = node_features.shape[0]
        batch_index_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        targets.append(target)
        masks.append(mask)

        node_offset += num_nodes

    # Merge all
    x = torch.cat(node_features_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    batch_index = torch.cat(batch_index_list, dim=0)

    targets = torch.stack(targets)
    masks = torch.stack(masks)

    return x, edge_index, batch_index, targets, masks
