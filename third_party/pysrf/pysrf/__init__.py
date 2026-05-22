"""pysrf: Similarity-based Representation Factorization."""

from __future__ import annotations

import logging

logging.getLogger("pysrf").addHandler(logging.NullHandler())

from .coherence import RankEstimate, estimate_rank
from .consensus import AlignedConsensus, ClusterConsensus, EnsembleFit
from .cross_validation import cross_val_score
from .model import SRF

try:
    from importlib.metadata import version

    __version__ = version("pysrf")
except Exception:
    __version__ = "0.1.0"


__all__: list[str] = [
    "SRF",
    "RankEstimate",
    "estimate_rank",
    "cross_val_score",
    "EnsembleFit",
    "ClusterConsensus",
    "AlignedConsensus",
]
