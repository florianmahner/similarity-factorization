"""Minimal helpers for bias-aware triplet similarity matrices."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def _count_triplet_pair_wins(
    n_objects: int,
    triplets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    triplets = triplets.astype(np.intp, copy=False)
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    nn = n_objects * n_objects

    wins = np.zeros(nn, dtype=np.float64)
    wins += np.bincount(i * n_objects + j, minlength=nn)
    wins += np.bincount(j * n_objects + i, minlength=nn)
    wins = wins.reshape(n_objects, n_objects)

    trials = np.zeros(nn, dtype=np.float64)
    for a, b in ((i, j), (i, k), (j, k)):
        trials += np.bincount(a * n_objects + b, minlength=nn)
        trials += np.bincount(b * n_objects + a, minlength=nn)
    trials = trials.reshape(n_objects, n_objects)
    return wins, trials


def _laplace_matrix(
    wins: np.ndarray,
    trials: np.ndarray,
    alpha: float,
    fill_value: float,
) -> np.ndarray:
    matrix = np.divide(
        wins + alpha,
        trials + 2.0 * alpha,
        out=np.full_like(wins, np.nan, dtype=np.float64),
        where=trials > 0,
    )
    matrix = np.nan_to_num(matrix, nan=fill_value)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _fit_item_bias_prior(
    matrix: np.ndarray,
    trials: np.ndarray,
    weight_mode: str,
    clip_eps: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    observed = (trials > 0) & ~np.eye(trials.shape[0], dtype=bool)
    clipped = np.clip(matrix, clip_eps, 1.0 - clip_eps)
    logits = logit(clipped)
    logits = np.where(observed, logits, 0.0)

    if weight_mode == "fisher":
        weights = trials * clipped * (1.0 - clipped)
    elif weight_mode == "shown":
        weights = trials.copy()
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    weights = np.where(observed, weights, 0.0)
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        raise ValueError("No observed off-diagonal entries available for prior fitting.")

    intercept = float((weights * logits).sum() / total_weight)
    item_bias = np.zeros(matrix.shape[0], dtype=np.float64)
    denom = weights.sum(axis=1)

    for _ in range(max_iter):
        updated_bias = np.divide(
            (weights * (logits - intercept - item_bias[None, :])).sum(axis=1),
            denom,
            out=np.zeros_like(item_bias),
            where=denom > 0,
        )
        updated_bias -= updated_bias.mean()
        intercept = float(
            (
                weights
                * (logits - updated_bias[:, None] - updated_bias[None, :])
            ).sum()
            / total_weight
        )
        if np.max(np.abs(updated_bias - item_bias)) < tol:
            item_bias = updated_bias
            break
        item_bias = updated_bias

    prior = expit(intercept + item_bias[:, None] + item_bias[None, :])
    np.fill_diagonal(prior, 1.0)
    return prior


def fit_triplet_bias_prior(
    n_objects: int,
    triplets: np.ndarray,
    alpha: float = 1.0,
    weight_mode: str = "fisher",
    baseline_fill_value: float = 0.5,
    clip_eps: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count pair outcomes and fit a simple item-bias prior from the same triplets."""
    wins, trials = _count_triplet_pair_wins(n_objects=n_objects, triplets=triplets)
    baseline = _laplace_matrix(
        wins=wins,
        trials=trials,
        alpha=alpha,
        fill_value=baseline_fill_value,
    )
    prior = _fit_item_bias_prior(
        matrix=baseline,
        trials=trials,
        weight_mode=weight_mode,
        clip_eps=clip_eps,
        max_iter=max_iter,
        tol=tol,
    )
    return wins, trials, prior


def build_bias_aware_triplet_matrix(
    wins: np.ndarray,
    trials: np.ndarray,
    prior: np.ndarray,
    alpha: float = 1.0,
    prior_strength: float = 0.0,
    fill_missing_with_prior: bool = True,
) -> np.ndarray:
    """Shrink pair wins/trials toward a fitted item-bias prior."""
    matrix = np.divide(
        wins + alpha + prior_strength * prior,
        trials + 2.0 * alpha + prior_strength,
        out=np.full_like(wins, np.nan, dtype=np.float64),
        where=trials > 0,
    )
    if fill_missing_with_prior:
        matrix = np.where(trials > 0, matrix, prior)
    np.fill_diagonal(matrix, 1.0)
    return matrix
