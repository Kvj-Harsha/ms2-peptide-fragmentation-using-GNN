import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def aa_one_hot(aa):
    vec = torch.zeros(len(AMINO_ACIDS))
    if aa in AA_TO_IDX:
        vec[AA_TO_IDX[aa]] = 1.0
    return vec

def build_peptide_graph(peptide):
    """
    Input: peptide string, e.g. "AAAAAAAAAAR"
    Returns:
        node_features: tensor [L, 21]
        edge_index: tensor [2, E]
    """
    L = len(peptide)

    # Node features: [AA one-hot (20), position (1)]
    node_features = []
    for i, aa in enumerate(peptide):
        one_hot = aa_one_hot(aa)
        pos_norm = torch.tensor([i / (L - 1)])  # normalized position 0..1
        node_features.append(torch.cat([one_hot, pos_norm], dim=0))

    node_features = torch.stack(node_features)   # shape [L, 21]

    # Edges: connect i -> i+1 and i+1 -> i (undirected graph)
    edges = []
    for i in range(L - 1):
        edges.append([i, i+1])
        edges.append([i+1, i])

    edge_index = torch.tensor(edges).t().long()  # shape [2, E]

    return node_features, edge_index
