"""Bias-aware similarity matrix for THINGS-behavior.

Wraps the lower-level helpers in ``src/similarity/triplet_rsm.py`` with the
fixed paper choices (alpha=1, Fisher weighting, prior_strength=10). See the
README in this directory for the formal definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from src.similarity.triplet_rsm import (
    build_bias_aware_triplet_matrix,
    fit_triplet_bias_prior,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Paper-fixed bias-aware parameters.
ALPHA = 1.0
WEIGHT_MODE = "fisher"
PRIOR_STRENGTH = 10.0


@dataclass(frozen=True, slots=True)
class BiasAwareMatrix:
    similarity: np.ndarray   # (n, n) symmetric in [0, 1], diagonal = 1.0
    prior: np.ndarray        # (n, n) fitted item-baseline matrix
    wins: np.ndarray         # (n, n) c_ij
    trials: np.ndarray       # (n, n) s_ij
    alpha: float
    weight_mode: str
    prior_strength: float


def build_bias_aware_things_matrix(
    triplets: np.ndarray,
    n_objects: int = 1854,
    alpha: float = ALPHA,
    weight_mode: str = WEIGHT_MODE,
    prior_strength: float = PRIOR_STRENGTH,
    verbose: bool = True,
) -> BiasAwareMatrix:
    """Build the bias-aware THINGS-behavior similarity matrix.

    Steps:
      1. Laplace estimate m_ij = (c_ij + alpha) / (s_ij + 2 alpha).
      2. Fit per-item bias b_i and global intercept beta_0 by weighted LS on
         logits, with Fisher weights w_ij = s_ij * m_ij * (1 - m_ij).
      3. Shrink each pair toward the fitted baseline p_ij = sigmoid(beta_0 +
         b_i + b_j) with prior strength lambda.
      4. Unobserved pairs imputed by p_ij; diagonal = 1.

    Parameters
    ----------
    triplets : (n_triplets, 3) array
        Each row [i, j, k] is a triplet where (i, j) is the chosen similar
        pair and k the odd-one-out.
    """
    t0 = time.time()
    wins, trials, prior = fit_triplet_bias_prior(
        n_objects=n_objects,
        triplets=triplets,
        alpha=alpha,
        weight_mode=weight_mode,
    )
    similarity = build_bias_aware_triplet_matrix(
        wins=wins,
        trials=trials,
        prior=prior,
        alpha=alpha,
        prior_strength=prior_strength,
        fill_missing_with_prior=True,
    )
    if verbose:
        print(
            f"  built bias-aware matrix in {time.time() - t0:.1f}s: "
            f"shape={similarity.shape}, weight_mode={weight_mode}, "
            f"alpha={alpha}, lambda={prior_strength}, "
            f"nan_frac={float(np.mean(np.isnan(similarity))):.2e}"
        )
    return BiasAwareMatrix(
        similarity=similarity,
        prior=prior,
        wins=wins,
        trials=trials,
        alpha=alpha,
        weight_mode=weight_mode,
        prior_strength=prior_strength,
    )


def load_things_train_triplets() -> np.ndarray:
    """Load the canonical THINGS 4.7M trainset.txt (1854 objects)."""
    path = PROJECT_ROOT / "data" / "things" / "triplets_47" / "trainset.txt"
    return np.loadtxt(path, dtype=np.int32)
