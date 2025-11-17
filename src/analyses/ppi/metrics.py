import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_link_prediction_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Compute link prediction metrics.

    Parameters
    ----------
    scores : np.ndarray
        Prediction scores for test pairs
    labels : np.ndarray
        Binary labels (1 for positive, 0 for negative)

    Returns
    -------
    dict[str, float]
        Dictionary with auroc, auprc, p500, ndcg metrics
    """
    idx = np.argsort(scores)[::-1]
    sorted_labels = labels[idx]

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    k = 500
    p_at_500 = sorted_labels[:k].sum() / min(k, len(sorted_labels))

    n_positives = labels.sum()
    if n_positives > 0:
        dcg = np.sum(sorted_labels / np.log2(np.arange(2, len(sorted_labels) + 2)))
        idcg = np.sum(1.0 / np.log2(np.arange(2, n_positives + 2)))
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return {"auroc": auroc, "auprc": auprc, "p500": p_at_500, "ndcg": ndcg}
