"""Cluster-count and partition recovery metrics (plan §10).

ARI / NMI are computed manually so this package has no scikit-learn
dependency beyond what RECIPE_K already requires.
"""
from __future__ import annotations
import numpy as np


def adjusted_rand_index(labels_true, labels_pred):
    if labels_true is None or labels_pred is None:
        return float("nan")
    a = np.asarray(labels_true).ravel()
    b = np.asarray(labels_pred).ravel()
    if a.size == 0 or a.size != b.size:
        return float("nan")
    # contingency
    classes_a, ia = np.unique(a, return_inverse=True)
    classes_b, ib = np.unique(b, return_inverse=True)
    n = a.size
    K, L = classes_a.size, classes_b.size
    C = np.zeros((K, L), dtype=np.int64)
    np.add.at(C, (ia, ib), 1)
    sum_comb_c = sum(_comb2(x) for x in C.ravel())
    sum_comb_k = sum(_comb2(x) for x in C.sum(axis=1))
    sum_comb_l = sum(_comb2(x) for x in C.sum(axis=0))
    comb_n = _comb2(n)
    if comb_n == 0:
        return float("nan")
    expected = sum_comb_k * sum_comb_l / comb_n
    max_ = 0.5 * (sum_comb_k + sum_comb_l)
    denom = max_ - expected
    if denom == 0:
        return 1.0 if sum_comb_c == max_ else 0.0
    return float((sum_comb_c - expected) / denom)


def normalized_mutual_info(labels_true, labels_pred):
    if labels_true is None or labels_pred is None:
        return float("nan")
    a = np.asarray(labels_true).ravel()
    b = np.asarray(labels_pred).ravel()
    if a.size == 0 or a.size != b.size:
        return float("nan")
    classes_a, ia = np.unique(a, return_inverse=True)
    classes_b, ib = np.unique(b, return_inverse=True)
    n = a.size
    C = np.zeros((classes_a.size, classes_b.size), dtype=np.int64)
    np.add.at(C, (ia, ib), 1)
    Pab = C / n
    Pa = Pab.sum(axis=1, keepdims=True)
    Pb = Pab.sum(axis=0, keepdims=True)
    nz = Pab > 0
    mi = np.sum(Pab[nz] * np.log(Pab[nz] / (Pa.repeat(Pb.size, axis=1)[nz]
                                              * Pb.repeat(Pa.size, axis=0)[nz])))
    Ha = -np.sum(Pa[Pa > 0] * np.log(Pa[Pa > 0]))
    Hb = -np.sum(Pb[Pb > 0] * np.log(Pb[Pb > 0]))
    denom = np.sqrt(max(Ha * Hb, 1e-30))
    if denom == 0:
        return 0.0
    return float(mi / denom)


def _comb2(x):
    x = int(x)
    return x * (x - 1) // 2


def count_metrics(K_hat, K_true):
    if K_true is None:
        return dict(hit=None, abs_err=None, rel_err=None, within1=None)
    if K_hat is None or K_hat < 0:
        return dict(hit=0, abs_err=None, rel_err=None, within1=None)
    K_hat = int(K_hat)
    K_true = int(K_true)
    return dict(
        hit=int(K_hat == K_true),
        abs_err=int(abs(K_hat - K_true)),
        rel_err=float(abs(K_hat - K_true) / max(K_true, 1)),
        within1=int(abs(K_hat - K_true) <= 1),
    )
