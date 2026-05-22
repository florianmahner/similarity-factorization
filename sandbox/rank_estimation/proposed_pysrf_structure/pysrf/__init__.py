"""pysrf: Similarity-based Representation Factorization."""

from __future__ import annotations

import logging

logging.getLogger("pysrf").addHandler(logging.NullHandler())

from .model import SRF
from .coherence import RankEstimator
from .cross_validation import (
    cross_val_score,
    GridSearchCV,
    EntryKFold,
    fit_and_score,
)
from .consensus import (
    EnsembleFit,
    ClusterConsensus,
    AlignedConsensus,
)

try:
    from importlib.metadata import version

    __version__ = version("pysrf")
except Exception:
    __version__ = "0.1.0"


__all__: list[str] = [
    "SRF",
    "RankEstimator",
    "cross_val_score",
    "GridSearchCV",
    "EntryKFold",
    "fit_and_score",
    "EnsembleFit",
    "ClusterConsensus",
    "AlignedConsensus",
]
