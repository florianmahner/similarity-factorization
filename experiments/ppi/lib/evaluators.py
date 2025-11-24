"""Method evaluators for link prediction."""

import networkx as nx
import numpy as np

from utils.graphs import build_sparse_adjacency
from ..lib.baselines import evaluate_baseline_fast
from ..lib.utils import compute_link_prediction_metrics, build_adjacency_balanced

try:
    from pysrf import SRF

    srf_available = True
except ImportError:
    srf_available = False


def evaluate_srf(
    nodes: list,
    train_edges: np.ndarray,
    test_edges: np.ndarray,
    test_pairs: np.ndarray,
    test_labels: np.ndarray,
    rank: int,
    seed: int,
    verbose: bool = False,
) -> tuple[dict[str, float], np.ndarray]:

    if not srf_available:
        return {"auroc": 0.0, "auprc": 0.0, "p500": 0.0, "ndcg": 0.0}, np.zeros(
            len(test_pairs)
        )

    n_nodes = len(nodes)

    # 1. Build Graphs using INTEGER Nodes
    # Since train_edges contains ints (e.g. 1428), we must ensure
    # the graph knows the universe is 0..N-1, not just the active nodes.
    g_train = nx.Graph()
    g_train.add_nodes_from(range(n_nodes))  # Nodes are 0, 1, ... N
    g_train.add_edges_from(train_edges)

    g_full = nx.Graph()
    g_full.add_nodes_from(range(n_nodes))
    g_full.add_edges_from(train_edges)
    g_full.add_edges_from(test_edges)

    # 2. Build Adjacency (Pass n_nodes int, not list)
    # CHANGE: Calling the fixed function
    adj = build_adjacency_balanced(g_train, g_full, n_nodes, seed=seed)

    verbose = True if seed == 0 else False

    # 3. Run SRF
    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=300,
        max_inner=50,
        tol=1e-5,
        verbose=verbose,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        loss="frobenius",
    )

    w = model.fit_transform(adj)

    # Calculate scores
    # Ensure test_pairs are ints (they should be from NPZ)
    scores = np.sum(w[test_pairs[:, 0]] * w[test_pairs[:, 1]], axis=1)

    metrics = compute_link_prediction_metrics(scores, test_labels)
    return metrics, scores


def evaluate_baselines(
    nodes: list,
    train_edges: np.ndarray,
    test_pairs: np.ndarray,
    test_labels: np.ndarray,
    methods: list[str],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate baseline methods (CN, AA, RA, JC).

    Returns:
        (metrics_dict, predictions_dict) where predictions_dict maps method -> scores_array
    """
    g_train = nx.Graph()
    g_train.add_nodes_from(nodes)
    g_train.add_edges_from(train_edges)

    a_sparse = build_sparse_adjacency(g_train, nodes)

    results = {}
    predictions_dict = {}
    for method in methods:
        # Get both metrics and scores in one call
        metrics, scores = evaluate_baseline_fast(
            a_sparse, test_pairs, test_labels, method, return_scores=True
        )
        results.update({f"{method.lower()}_{k}": v for k, v in metrics.items()})
        predictions_dict[method.lower()] = scores

    return results, predictions_dict


def evaluate_skipgnn(
    nodes: list,
    train_edges: np.ndarray,
    test_pairs: np.ndarray,
    test_labels: np.ndarray,
    rank: int,
    epochs: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate SkipGNN method.

    Returns:
        (metrics_dict, predictions_array)
    """
    from ..lib.skip_gnn import train_skipgnn, predict_skipgnn, skipgnn_available

    if not skipgnn_available:
        return {"auroc": 0.0, "auprc": 0.0, "p500": 0.0, "ndcg": 0.0}, np.zeros(
            len(test_pairs)
        )

    model, features, adj, adj2, idx_map = train_skipgnn(
        nodes, train_edges, rank, epochs, seed
    )

    scores = predict_skipgnn(model, features, adj, adj2, idx_map, nodes, test_pairs)

    metrics = compute_link_prediction_metrics(scores, test_labels)
    return metrics, scores


def evaluate_node2vec_pyg(
    nodes: list,
    train_edges: np.ndarray,
    test_pairs: np.ndarray,
    test_labels: np.ndarray,
    dimensions: int = 128,
    walk_length: int = 80,
    context_size: int = 10,
    walks_per_node: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.01,
    seed: int = 42,
    device: str = "auto",
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate PyTorch Geometric node2vec for link prediction.

    Args:
        nodes: List of all node IDs
        train_edges: Array of training edges
        test_pairs: Array of test node pairs
        test_labels: Binary labels for test pairs
        dimensions: Embedding dimension
        walk_length: Length of random walks
        context_size: Context window size
        walks_per_node: Number of walks per node
        p: Return parameter
        q: In-out parameter
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed
        device: Device to use

    Returns:
        (metrics_dict, predictions_array)
    """
    from ..lib.pyg_methods import train_node2vec_pyg

    embeddings = train_node2vec_pyg(
        nodes=nodes,
        train_edges=train_edges,
        dimensions=dimensions,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
    )

    scores = np.sum(
        embeddings[test_pairs[:, 0]] * embeddings[test_pairs[:, 1]],
        axis=1,
    )

    metrics = compute_link_prediction_metrics(scores, test_labels)
    return metrics, scores


def evaluate_seal(
    nodes: list,
    train_edges: np.ndarray,
    test_pairs: np.ndarray,
    test_labels: np.ndarray,
    num_hops: int = 2,
    hidden_channels: int = 32,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.0001,
    device: str = "auto",
    seed: int = 42,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate SEAL end-to-end link prediction.

    Args:
        nodes: List of all node IDs
        train_edges: Array of training edges
        test_pairs: Array of test node pairs
        test_labels: Binary labels for test pairs
        num_hops: Number of hops for subgraph extraction
        hidden_channels: Hidden channels in GNN
        num_layers: Number of GNN layers
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        seed: Random seed

    Returns:
        (metrics_dict, predictions_array)
    """
    from ..lib.pyg_methods import train_seal

    metrics, scores = train_seal(
        nodes=nodes,
        train_edges=train_edges,
        test_pairs=test_pairs,
        test_labels=test_labels,
        num_hops=num_hops,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    return metrics, scores
