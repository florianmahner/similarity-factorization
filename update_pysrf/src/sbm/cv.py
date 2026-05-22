"""Cross-validation wrappers for SBM cluster-count selection.

Three protocols (plan §13.5, §16):

  - ``plain_cv_sbm``       : ordinary k-fold CV with the full off-diagonal as
                              the outer pool. Reference protocol.
  - ``nested_cv_sbm``      : Recipe-K pre-masked CV — keep p_outer fraction of
                              off-diagonal entries in the outer pool, then
                              split the pool into k_inner folds. With
                              p_outer = p_cv, this realises the §13.5 design.
  - ``single_mask_cv_sbm`` : single hold-out mask at p_star, ``n_reps`` repeats.

All three reuse the existing Recipe-K masking primitives from
``symmnmf.cross_validation`` and ``_common`` (``mask_missing_entries`` and
``split_omega_into_folds``), so the masking statistics match exactly the rest
of the Recipe-K experiment suite.
"""
from __future__ import annotations
import numpy as np
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from .fitters import fit_sbm
from .losses import score_sbm_predictions, training_sigma2
from .selection import argmin_rank, one_se_rank


# ----------------------------------------------------------------------
# Mask helpers (kept local so this package is import-safe even when the rest
# of RECIPE_K/src is not on sys.path).
# ----------------------------------------------------------------------

def _mask_missing(S, observed_fraction, rng):
    """Pre-mask off-diagonal: True where MISSING (held out from outer pool).

    Diagonal is always kept (False). Identical convention to
    ``symmnmf.cross_validation.mask_missing_entries``.
    """
    n = S.shape[0]
    observed = ~np.isnan(S)
    iu = np.triu_indices(n, k=1)
    triu_observed = observed[iu]
    valid = np.where(triu_observed)[0]
    n_keep = int(observed_fraction * len(valid))
    miss = np.ones((n, n), dtype=bool)
    if n_keep > 0 and valid.size > 0:
        keep_pos = rng.choice(valid, size=min(n_keep, valid.size), replace=False)
        ii = iu[0][keep_pos]
        jj = iu[1][keep_pos]
        miss[ii, jj] = False
        miss[jj, ii] = False
    np.fill_diagonal(miss, False)
    return miss


def _split_omega_into_folds(M_outer, k_inner, rng):
    """Partition observed off-diagonal pairs (~M_outer) into k_inner symmetric
    folds. Returns list of bool validation masks (True = held out for that
    fold), or None if there are not enough pairs.
    """
    n = M_outer.shape[0]
    iu = np.triu_indices(n, k=1)
    observed_pos = ~M_outer[iu]
    valid = np.where(observed_pos)[0]
    if len(valid) < k_inner:
        return None
    perm = rng.permutation(len(valid))
    fold_groups = np.array_split(perm, k_inner)
    val_masks = []
    for fg in fold_groups:
        m = np.zeros_like(M_outer, dtype=bool)
        idxs = valid[fg]
        ii = iu[0][idxs]
        jj = iu[1][idxs]
        m[ii, jj] = True
        m[jj, ii] = True
        val_masks.append(m)
    return val_masks


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

def _fit_and_score(S, train_mask, val_mask, rank, loss, degree_corrected, seed,
                   loss_extra=None, init="spectral", max_label_iter=10):
    try:
        out = fit_sbm(S, rank=rank, train_mask=train_mask, loss=loss,
                       degree_corrected=degree_corrected, init=init,
                       seed=seed, max_label_iter=max_label_iter)
        S_hat = out["S_hat"]
        extra = dict(loss_extra or {})
        if loss == "gaussian_nll" and "sigma2" not in extra:
            extra["sigma2"] = training_sigma2(S, S_hat, train_mask)
        scored = score_sbm_predictions(S, S_hat, val_mask, loss=loss,
                                       extra_params=extra)
        return {
            "loss_mean": scored["loss_mean"],
            "n_scored": scored["n_scored"],
            "labels": out["labels"],
            "fit_status": out["fit_status"],
            "n_iter": out["n_iter"],
            "train_loss": out["train_loss"],
        }
    except Exception as exc:
        return {
            "loss_mean": float("nan"),
            "n_scored": 0,
            "labels": None,
            "fit_status": f"error:{type(exc).__name__}",
            "n_iter": 0,
            "train_loss": float("nan"),
        }


# ----------------------------------------------------------------------
# Protocol implementations
# ----------------------------------------------------------------------

def _generic_kfold(S, ranks, loss, k_inner, n_reps, degree_corrected, seed,
                   n_jobs, p_outer, loss_extra=None, init="spectral",
                   max_label_iter=10):
    """Common k-fold engine used by both plain CV (p_outer=1.0) and Recipe-K
    pre-masked CV (p_outer < 1.0)."""
    n = S.shape[0]
    rng = check_random_state(int(seed))
    already_nan = np.isnan(S)
    off_diag = ~np.eye(n, dtype=bool)
    valid_mask = off_diag & ~already_nan

    cv_loss = np.full((len(ranks), n_reps, k_inner), np.nan)
    fit_status_grid = np.empty((len(ranks), n_reps, k_inner), dtype=object)
    fit_status_grid.fill("not_run")
    labels_by_rank = {int(r): None for r in ranks}

    for rep in range(n_reps):
        M_outer = _mask_missing(S, float(p_outer), rng)
        val_masks = _split_omega_into_folds(M_outer, k_inner, rng)
        if val_masks is None:
            continue

        # Build the (fold, rank) work list. Use threading because fit_sbm is
        # numpy-heavy; threading avoids the joblib pickling overhead for
        # closures and large numpy arrays.
        jobs = []
        for fold_idx, V in enumerate(val_masks):
            holdout = V | M_outer
            train_mask = (~holdout) & valid_mask
            for ri, r in enumerate(ranks):
                job_seed = int(seed + 7919 * (rep + 1) + 101 * fold_idx + ri)
                jobs.append((ri, fold_idx, train_mask, V & valid_mask,
                              int(r), job_seed))

        if n_jobs in (None, 1):
            results = [_fit_and_score(S, tm, vm, r, loss, degree_corrected,
                                       sd, loss_extra, init, max_label_iter)
                       for (_, _, tm, vm, r, sd) in jobs]
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                delayed(_fit_and_score)(S, tm, vm, r, loss, degree_corrected,
                                         sd, loss_extra, init, max_label_iter)
                for (_, _, tm, vm, r, sd) in jobs
            )

        for (ri, fold_idx, _, _, r, _), res in zip(jobs, results):
            cv_loss[ri, rep, fold_idx] = res["loss_mean"]
            fit_status_grid[ri, rep, fold_idx] = res["fit_status"]
            if rep == 0 and fold_idx == 0 and res["labels"] is not None:
                labels_by_rank[int(r)] = np.asarray(res["labels"])

    return _summarise(ranks, cv_loss, fit_status_grid, labels_by_rank,
                      p_outer=p_outer, k_inner=k_inner, n_reps=n_reps,
                      loss=loss, degree_corrected=degree_corrected)


def _summarise(ranks, cv_loss, fit_status, labels_by_rank, **diagnostics):
    R = len(ranks)
    flat = cv_loss.reshape(R, -1)
    mean = np.nanmean(flat, axis=1)
    std = np.nanstd(flat, axis=1)
    counts = np.sum(np.isfinite(flat), axis=1)
    sem = std / np.maximum(np.sqrt(counts), 1)

    am = argmin_rank(mean, list(ranks))
    one = one_se_rank(mean, list(ranks), sem)
    return {
        "ranks": list(ranks),
        "cv_loss": mean,
        "cv_sem": sem,
        "cv_raw": cv_loss,
        "argmin": am,
        "one_se": one,
        "labels_by_rank": labels_by_rank,
        "fit_status_by_rank": fit_status,
        "diagnostics": diagnostics,
    }


def nested_cv_sbm(S, p_outer, ranks, loss="gaussian_mse", k_inner=5, n_reps=1,
                  degree_corrected=False, seed=0, n_jobs=1, loss_extra=None,
                  init="spectral", max_label_iter=10):
    """Recipe-K pre-masked SBM cross-validation (plan §13.5).

    With ``p_outer = p_cv`` from Recipe K, the post-CV-split training fraction
    equals ``p_star`` in expectation (uncapped) or ``p_train_eff`` (when the
    cap binds).
    """
    return _generic_kfold(S, ranks, loss, k_inner, n_reps, degree_corrected,
                           seed, n_jobs, p_outer=float(p_outer),
                           loss_extra=loss_extra, init=init,
                           max_label_iter=max_label_iter)


def plain_cv_sbm(S, ranks, loss="gaussian_mse", k_inner=5, n_reps=1,
                 degree_corrected=False, seed=0, n_jobs=1, loss_extra=None,
                 init="spectral", max_label_iter=10):
    """Plain k-fold CV: no Recipe-K pre-mask; outer pool = all off-diagonal."""
    return _generic_kfold(S, ranks, loss, k_inner, n_reps, degree_corrected,
                           seed, n_jobs, p_outer=1.0,
                           loss_extra=loss_extra, init=init,
                           max_label_iter=max_label_iter)


def single_mask_cv_sbm(S, p_star, ranks, loss="gaussian_mse", n_reps=5,
                       degree_corrected=False, seed=0, n_jobs=1,
                       loss_extra=None, init="spectral", max_label_iter=10):
    """Single hold-out mask at p_star, ``n_reps`` independent masks."""
    n = S.shape[0]
    rng = check_random_state(int(seed))
    already_nan = np.isnan(S)
    off_diag = ~np.eye(n, dtype=bool)
    valid_mask = off_diag & ~already_nan

    cv_loss = np.full((len(ranks), n_reps, 1), np.nan)
    fit_status_grid = np.empty((len(ranks), n_reps, 1), dtype=object)
    fit_status_grid.fill("not_run")
    labels_by_rank = {int(r): None for r in ranks}

    for rep in range(n_reps):
        M = _mask_missing(S, float(p_star), rng)
        train_mask = (~M) & valid_mask
        val_mask = M & valid_mask
        jobs = []
        for ri, r in enumerate(ranks):
            job_seed = int(seed + 99991 * (rep + 1) + ri)
            jobs.append((ri, int(r), job_seed))
        if n_jobs in (None, 1):
            results = [_fit_and_score(S, train_mask, val_mask, r, loss,
                                       degree_corrected, sd, loss_extra, init,
                                       max_label_iter)
                       for (_, r, sd) in jobs]
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                delayed(_fit_and_score)(S, train_mask, val_mask, r, loss,
                                         degree_corrected, sd, loss_extra, init,
                                         max_label_iter)
                for (_, r, sd) in jobs
            )
        for (ri, r, _), res in zip(jobs, results):
            cv_loss[ri, rep, 0] = res["loss_mean"]
            fit_status_grid[ri, rep, 0] = res["fit_status"]
            if rep == 0 and res["labels"] is not None:
                labels_by_rank[int(r)] = np.asarray(res["labels"])

    return _summarise(ranks, cv_loss, fit_status_grid, labels_by_rank,
                      p_outer=float(p_star), k_inner=1, n_reps=n_reps,
                      loss=loss, degree_corrected=degree_corrected,
                      protocol="single_mask")
