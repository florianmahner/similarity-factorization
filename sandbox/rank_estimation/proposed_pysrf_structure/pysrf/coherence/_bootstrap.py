"""Masked-bootstrap engine for eigenspace coherence."""

from __future__ import annotations

import numpy as np


def symmetrize(s: np.ndarray) -> np.ndarray:
    """Symmetrize while preserving NaN semantics."""
    raise NotImplementedError


def observation_mask(s_sym: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (s_filled, mask, observed_rate)."""
    raise NotImplementedError


def reference_eigenpairs(
    s_filled: np.ndarray,
    mask: np.ndarray,
    observed_rate: float,
    k_max: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k_max`` eigenpairs of the reference matrix."""
    raise NotImplementedError


def bootstrap_coherence(
    s_filled: np.ndarray,
    mask: np.ndarray,
    u_ref: np.ndarray,
    k_max: int,
    sampling_grid: np.ndarray,
    n_bootstrap: int,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative overlap and Rayleigh-trace numerator across the grid."""
    raise NotImplementedError
