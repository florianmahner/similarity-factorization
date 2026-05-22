"""Helpers for building similarity matrices from several data sources."""

from .builder import build_similarity, get_rank_grid
from .triplet_rsm import (
    build_bias_aware_triplet_matrix,
    fit_triplet_bias_prior,
)

__all__ = [
    "build_similarity",
    "build_bias_aware_triplet_matrix",
    "fit_triplet_bias_prior",
    "get_rank_grid",
]
