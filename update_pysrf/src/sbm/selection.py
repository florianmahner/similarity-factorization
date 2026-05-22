"""Argmin and 1-SE rank selection helpers for SBM CV curves.

Mirrors ``argmin_safe`` / ``argmin_1se_safe`` in ``_common.py``, but lifted out
so the sbm package does not need to import from the top-level Recipe-K modules.
"""
from __future__ import annotations
import numpy as np


def argmin_rank(curve, ranks):
    curve = np.asarray(curve, dtype=float)
    if not np.isfinite(curve).any():
        return -1
    i = int(np.nanargmin(curve))
    return int(ranks[i])


def one_se_rank(curve, ranks, sem):
    """Smallest rank whose CV loss is within one SE of the minimum.

    Returns the chosen rank, or -1 if everything is NaN, or argmin if sem is
    None / all NaN.
    """
    curve = np.asarray(curve, dtype=float)
    if not np.isfinite(curve).any():
        return -1
    if sem is None:
        return argmin_rank(curve, ranks)
    sem = np.asarray(sem, dtype=float)
    finite = np.where(np.isfinite(curve) & np.isfinite(sem))[0]
    if finite.size == 0:
        return argmin_rank(curve, ranks)
    am = int(finite[np.argmin(curve[finite])])
    thresh = float(curve[am] + sem[am])
    for i in finite:
        if curve[i] <= thresh:
            return int(ranks[i])
    return int(ranks[am])
