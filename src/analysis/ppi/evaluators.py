"""Method evaluators for link prediction."""

import networkx as nx
import numpy as np

from .graph_utils import build_sparse_adjacency, build_adjacency_with_nan
from .baselines import evaluate_baseline_fast
from .metrics import compute_link_prediction_metrics

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
    """Evaluate SRF method.

    Returns:
        (metrics_dict, predictions_array)
    """
    if not srf_available:
        return {"auroc": 0.0, "auprc": 0.0, "p500": 0.0, "ndcg": 0.0}, np.zeros(
            len(test_pairs)
        )

    g_train = nx.Graph()
    g_train.add_nodes_from(nodes)
    g_train.add_edges_from(train_edges)

    g_full = nx.Graph()
    g_full.add_nodes_from(nodes)
    g_full.add_edges_from(train_edges)
    g_full.add_edges_from(test_edges)

    adj = build_adjacency_with_nan(g_train, g_full, nodes)

    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=30,
        max_inner=20,
        tol=1e-5,
        verbose=1 if verbose else 0,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        loss="frobenius",
    )

    w = model.fit_transform(adj)
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
    from .skipgnn_utils import train_skipgnn, predict_skipgnn, skipgnn_available

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
