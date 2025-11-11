"""SkipGNN training and evaluation utilities."""

import copy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

try:
    import sys
    import scipy.sparse as sp

    skipgnn_path = Path(__file__).parent.parent.parent.parent / "SkipGNN" / "SkipGNN"
    sys.path.insert(0, str(skipgnn_path))
    from models import SkipGNN
    from utils import normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor

    skipgnn_available = True
except ImportError:
    skipgnn_available = False


def prepare_skipgnn_data(nodes: list, train_edges: list, seed: int):
    """Prepare adjacency matrices and features for SkipGNN.

    Args:
        nodes: List of node IDs
        train_edges: List of training edges (tuples)
        seed: Random seed

    Returns:
        Tuple of (adj, adj2, features, idx_map)
    """
    idx_map = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    features = np.eye(n_nodes, dtype=np.float32)

    edge_indices = np.array([[idx_map[u], idx_map[v]] for u, v in train_edges])
    adj = sp.coo_matrix(
        (np.ones(len(edge_indices)), (edge_indices[:, 0], edge_indices[:, 1])),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj2 = adj.dot(adj)
    adj2 = adj2.sign()
    adj2 = adj2 + sp.eye(adj2.shape[0])
    adj2 = normalize_adj(adj2)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)

    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    features = torch.FloatTensor(features)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adj2, features, idx_map


def create_skipgnn_dataset(nodes: list, train_edges: list, idx_map: dict, seed: int):
    """Create train/val datasets with negative sampling.

    Args:
        nodes: List of node IDs
        train_edges: List of training edges
        idx_map: Node ID to index mapping
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    g_train = nx.Graph()
    g_train.add_nodes_from(nodes)
    g_train.add_edges_from(train_edges)

    np.random.seed(seed)
    train_edges = list(train_edges)
    np.random.shuffle(train_edges)

    val_split = int(0.9 * len(train_edges))
    train_edges_split = train_edges[:val_split]
    val_edges_split = train_edges[val_split:]

    available_non_edges = list(nx.non_edges(g_train))
    np.random.shuffle(available_non_edges)

    train_neg = available_non_edges[: len(train_edges_split)]
    val_neg = available_non_edges[
        len(train_edges_split) : len(train_edges_split) + len(val_edges_split)
    ]

    train_data = []
    for u, v in train_edges_split:
        train_data.append({"Protein1_ID": u, "Protein2_ID": v, "label": 1})
    for u, v in train_neg:
        train_data.append({"Protein1_ID": u, "Protein2_ID": v, "label": 0})
    train_df = pd.DataFrame(train_data)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    val_data = []
    for u, v in val_edges_split:
        val_data.append({"Protein1_ID": u, "Protein2_ID": v, "label": 1})
    for u, v in val_neg:
        val_data.append({"Protein1_ID": u, "Protein2_ID": v, "label": 0})
    val_df = pd.DataFrame(val_data)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    try:
        from utils import Data_PPI
    except ImportError:
        raise ImportError("SkipGNN Data_PPI not available")

    params = {"batch_size": 128, "shuffle": True, "num_workers": 0, "drop_last": True}

    training_set = Data_PPI(idx_map, train_df.label.values, train_df)
    train_loader = data.DataLoader(training_set, **params)

    validation_set = Data_PPI(idx_map, val_df.label.values, val_df)
    val_loader = data.DataLoader(validation_set, **params)

    return train_loader, val_loader


def train_skipgnn(
    nodes: list,
    train_edges: list,
    rank: int = 64,
    epochs: int = 15,
    seed: int = 42,
    verbose: bool = False,
):
    """Train SkipGNN model directly from splits data.

    Args:
        nodes: List of node IDs
        train_edges: List of training edges (tuples)
        rank: Hidden dimension size
        epochs: Number of training epochs
        seed: Random seed
        verbose: Print training progress

    Returns:
        Tuple of (model, features, adj, adj2, idx_map)
    """
    if not skipgnn_available:
        raise ImportError("SkipGNN not available")

    adj, adj2, features, idx_map = prepare_skipgnn_data(nodes, train_edges, seed)
    train_loader, val_loader = create_skipgnn_dataset(nodes, train_edges, idx_map, seed)

    model = SkipGNN(
        nfeat=features.shape[1],
        nhid1=rank,
        nhid2=32,
        nhid_decode1=16,
        dropout=0.5,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        adj2 = adj2.cuda()

    loss_fct = torch.nn.BCELoss()
    m = torch.nn.Sigmoid()
    max_auc = 0
    model_max = copy.deepcopy(model)

    def test_val(loader):
        model.eval()
        y_pred, y_label = [], []
        for i, (label, inp) in enumerate(loader):
            if device.type == "cuda":
                label = label.cuda()
            output, _ = model(features, adj, adj2, inp)
            n = torch.squeeze(m(output))
            label_ids = label.to("cpu").numpy()
            y_label.extend(label_ids.flatten().tolist())
            y_pred.extend(output.flatten().tolist())
        from sklearn.metrics import roc_auc_score, average_precision_score

        return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred)

    for epoch in range(epochs):
        model.train()
        for i, (label, inp) in enumerate(train_loader):
            if device.type == "cuda":
                label = label.cuda()
            optimizer.zero_grad()
            output, _ = model(features, adj, adj2, inp)
            n = torch.squeeze(m(output))
            loss_train = loss_fct(n, label.float())
            loss_train.backward()
            optimizer.step()

        if verbose:
            roc_val, prc_val = test_val(val_loader)
            print(
                f"  Epoch {epoch+1}/{epochs}: val_auroc={roc_val:.4f}, val_auprc={prc_val:.4f}"
            )
        else:
            roc_val, _ = test_val(val_loader)

        if roc_val > max_auc:
            model_max = copy.deepcopy(model)
            max_auc = roc_val

    return model_max, features, adj, adj2, idx_map


def predict_skipgnn(
    model,
    features,
    adj,
    adj2,
    idx_map: dict,
    nodes: list,
    test_pairs: np.ndarray,
) -> np.ndarray:
    """Generate predictions using trained SkipGNN model.

    Args:
        model: Trained SkipGNN model
        features: Node features tensor
        adj: Adjacency matrix
        adj2: Second-order adjacency matrix
        idx_map: Node ID to index mapping
        nodes: List of node IDs
        test_pairs: Array of (i, j) node index pairs

    Returns:
        Array of prediction scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    scores = np.zeros(len(test_pairs), dtype=np.float32)
    batch_size = 10000

    with torch.no_grad():
        for start in range(0, len(test_pairs), batch_size):
            end = min(start + batch_size, len(test_pairs))
            batch_pairs = test_pairs[start:end]

            batch_indices = []
            valid_mask = []
            for i, j in batch_pairs:
                node_i = nodes[i]
                node_j = nodes[j]
                if node_i in idx_map and node_j in idx_map:
                    batch_indices.append([idx_map[node_i], idx_map[node_j]])
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)

            if batch_indices:
                batch_tensor = torch.LongTensor(batch_indices).t()
                if device.type == "cuda":
                    batch_tensor = batch_tensor.cuda()

                output, _ = model(features, adj, adj2, batch_tensor)
                batch_scores = torch.sigmoid(output).cpu().numpy().flatten()

                score_idx = 0
                for i, is_valid in enumerate(valid_mask):
                    if is_valid:
                        scores[start + i] = batch_scores[score_idx]
                        score_idx += 1

    return scores
