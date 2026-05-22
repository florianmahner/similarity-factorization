"""
Shared utilities for the experiments suite.

Provides:
  - detect_cliff: cliff detector with magnitude + salience thresholds
  - estimate_k_smooth: smoothed change-point detector
  - variance_weighted_kappa: weighted kappa over r <= k
  - spectral_pass: wrapper around the existing Iproj computation
  - nested_cv: generalized nested CV with three losses (V/M/U)
  - mean_running_median: smoothing utility
  - data loaders for v3 datasets
"""
import os
import sys
import numpy as np
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

# Self-locate: this file lives at RECIPE_K/src/_common.py. The sibling
# directory `symmnmf/` provides the ADMM fitter and mask_missing_entries
# helper; `_compute_coherence.py` lives next to this file. Putting RECIPE_K/src/
# on sys.path makes both `from _compute_coherence import ...` and
# `from symmnmf.cross_validation import ...` resolvable from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def running_median(x, w):
    if w <= 1:
        return np.asarray(x, float).copy()
    n = len(x); half = w // 2; o = np.zeros(n)
    for i in range(n):
        o[i] = np.median(np.asarray(x, float)[max(0, i - half):min(n, i + half + 1)])
    return o


def estimate_k_smooth(kappa_hat, k_list, smooth_window=10):
    """LEGACY smoothed change-point. Kept for backwards-compat / comparison.

    Use estimate_k_cut_fstat as the new default (no window parameter, principled).
    Returns (k_smooth, i_smooth).
    """
    kappa_sm = running_median(kappa_hat, max(1, smooth_window))
    d = kappa_sm[1:] - kappa_sm[:-1]
    valid = np.where(np.asarray(k_list)[:-1] >= 2)[0]
    if valid.size == 0:
        valid = np.arange(len(k_list) - 1)
    i_star = int(valid[np.argmax(d[valid])])
    return int(k_list[i_star]), i_star


def estimate_k_cut_fstat(kappa_hat, k_list, min_seg=2):
    """Two-regime changepoint detector via F-statistic / min-SSE.

    Mathematically principled — no window parameter. Models kappa as piecewise
    constant: signal regime (small kappa) for r <= r_c, bulk regime (O(1)+) for
    r > r_c. Picks r_c maximizing the F-statistic of the two-mean comparison
    (equivalently, minimizing within-segment SSE).

    Convention: k_cut = k_list[i_star] where i_star is the index of the cut.
    k_cut is the LAST signal rank (rank just before the regime switch).

    Returns (k_cut, F_max, i_star, F_array).
    """
    kappa = np.asarray(kappa_hat, float)
    n = len(kappa)
    if n < 2 * min_seg:
        return None, float("nan"), None, None

    F = np.full(n - 1, -np.inf)
    for i_c in range(min_seg - 1, n - min_seg):
        seg1 = kappa[:i_c + 1]
        seg2 = kappa[i_c + 1:]
        n1, n2 = len(seg1), len(seg2)
        mu1, mu2 = float(np.mean(seg1)), float(np.mean(seg2))
        sse1 = float(np.sum((seg1 - mu1) ** 2))
        sse2 = float(np.sum((seg2 - mu2) ** 2))
        pooled_var = (sse1 + sse2) / max(n1 + n2 - 2, 1)
        if pooled_var <= 1e-12:
            F[i_c] = np.inf
        else:
            F[i_c] = (n1 * n2 / (n1 + n2)) * (mu2 - mu1) ** 2 / pooled_var

    valid = np.where(np.asarray(k_list)[:-1] >= 2)[0]
    if valid.size == 0:
        valid = np.arange(n - 1)
    F_valid = F[valid]
    i_star = int(valid[np.nanargmax(F_valid)])
    return int(k_list[i_star]), float(F[i_star]), i_star, F


def detect_cliff(kappa_hat, k_list, log_jump_min=2.0, ratio_min=30.0):
    """Returns (k_cliff, max_log_jump, salience). k_cliff is None if no cliff."""
    kh = np.asarray(kappa_hat, float)
    eps = 1e-6
    log_d = np.log(np.maximum(kh[1:], eps)) - np.log(np.maximum(kh[:-1], eps))
    valid = np.where(np.asarray(k_list)[:-1] >= 2)[0]
    if valid.size == 0:
        return None, float("nan"), float("nan")
    log_d_v = log_d[valid]
    med = float(np.median(np.abs(log_d_v))) + 1e-12
    max_lj = float(log_d_v.max())
    salience = max_lj / med
    if max_lj > log_jump_min and salience > ratio_min:
        i_star = int(valid[np.argmax(log_d_v)])
        return int(k_list[i_star]), max_lj, salience
    return None, max_lj, salience


def variance_weighted_kappa(kappa_hat, evals_ref, k_list, k_cut):
    k_arr = np.asarray(k_list, int); mask = k_arr <= k_cut
    rs = k_arr[mask]; w = evals_ref[rs - 1]; ks = kappa_hat[mask]
    return float(np.sum(w * ks) / np.sum(w)) if w.sum() > 0 else float("nan")


def detectability_rho(evals_ref, k):
    if k >= len(evals_ref):
        return float("nan")
    lam = evals_ref[k]
    bulk = np.sum(evals_ref[k:] ** 2)
    return float(lam ** 2 / max(bulk, 1e-30))


def power_law_alpha(evals_ref, k, kw=30):
    rs = np.arange(k + 1, min(k + 1 + kw, len(evals_ref)))
    lams = evals_ref[rs - 1]
    pos = lams > 0
    if pos.sum() < 5:
        return float("nan")
    s, _ = np.polyfit(np.log(rs[pos]), np.log(lams[pos]), 1)
    return -s


def detectability_flag(rho, alpha):
    if (rho > 0.1) and (alpha > 1.0):
        return "sharp"
    if (rho < 0.03) or (alpha < 0.5):
        return "smooth"
    return "borderline"


def spectral_pass(S, k_list=None, p_grid=None, B=20, smooth_window=10,
                  cliff_log_jump_min=2.0, cliff_ratio_min=30.0,
                  show_progress=False):
    """Run the Iproj spectral pass and compute k_smooth, k_cliff, kappa, etc.

    Returns dict with: k_list, kappa_hat, evals_ref, k_smooth, k_cliff,
        log_jump, salience, rho, alpha, flag.
    """
    from _compute_coherence import compute_incremental_coherence_multi_k_eig_anisotropic

    n = S.shape[0]
    if k_list is None:
        k_max = min(n // 4, 100)
        k_list = list(range(1, k_max + 1))
    if p_grid is None:
        p_grid = np.linspace(0.05, 0.95, 20)

    res = compute_incremental_coherence_multi_k_eig_anisotropic(
        S, k_list, p_grid, B=B, use_baseline_correction=False,
        plot_tau=False, mark_activation=False, visualize=False,
        show_progress=show_progress,
    )
    Iproj_boot = res["Iproj_boot"]; evals_ref = res["evals_ref"]
    # Rayleigh-trace numerator: trPkS_boot[ki, j, b] = tr(P_k(p_j) S) computed
    # EXACTLY against the full reference S0 (NaN-zeroed) since 2026-05-11 via
    # one S0 @ U_k matvec per (p, b). Earlier releases used the rank-Kmax
    # truncation tr(P_k S_ref^{(Kmax)}). The exact form is paired with
    # tr_S_mode="full" (np.trace(S)) in recipe_K(); the truncated denominator
    # tr_S_mode="truncated" remains as a legacy diagnostic.
    trPkS_boot = res.get("trPkS_boot")
    tr_S_full = float(np.trace(S))
    tr_S_truncated = float(np.sum(evals_ref))

    # kappa_hat
    I_med = np.median(Iproj_boot, axis=2)
    hi = np.where(p_grid >= np.quantile(p_grid, 0.85))[0]
    if hi.size == 0:
        hi = np.array([len(p_grid) - 1])
    scale = p_grid[hi] / np.maximum(1 - p_grid[hi], 1e-3)
    kappa_hat = np.median((1 - I_med[:, hi]) * scale[None, :], axis=1)

    # Primary: F-statistic two-regime changepoint (no smoothing window)
    k_cut, F_max, i_cut, _F_arr = estimate_k_cut_fstat(kappa_hat, k_list)
    # Parallel: cliff detector for sharp single-step jumps
    k_cliff, log_jump, salience = detect_cliff(kappa_hat, k_list,
                                               log_jump_min=cliff_log_jump_min,
                                               ratio_min=cliff_ratio_min)
    # Legacy: smoothed-window detector, kept for comparison/regression-check
    k_smooth_legacy, _ = estimate_k_smooth(kappa_hat, k_list, smooth_window)

    rho = detectability_rho(evals_ref, k_cut)
    alpha = power_law_alpha(evals_ref, k_cut)
    flag = detectability_flag(rho, alpha)

    # I_med: median over bootstraps of I_r^proj(p_j); shape (K, P)
    # trPkS_boot/_med: Rayleigh-trace numerator for proper VE^D in Recipe K.
    trPkS_med = np.median(trPkS_boot, axis=2) if trPkS_boot is not None else None
    return dict(
        n=int(n),
        k_list=k_list, p_grid=p_grid, kappa_hat=kappa_hat, evals_ref=evals_ref,
        k_cut=k_cut, F_max=F_max, k_cliff=k_cliff, k_smooth_legacy=k_smooth_legacy,
        # Backwards compat: keep "k_smooth" as alias for new primary k_cut.
        k_smooth=k_cut,
        cliff_log_jump=log_jump, cliff_salience=salience,
        rho=rho, alpha=alpha, flag=flag,
        smooth_window=smooth_window, B=B,
        Iproj_boot=Iproj_boot,    # full bootstrap stack (K, P, B)
        I_med=I_med,              # median over bootstraps (K, P)
        trPkS_boot=trPkS_boot,    # Rayleigh-trace numerator (K, P, B), Recipe K corrected
        trPkS_med=trPkS_med,      # median over bootstraps (K, P)
        tr_S=tr_S_full,           # full tr(S), proper VE^D denominator
        tr_S_truncated=tr_S_truncated,  # sum of top-Kmax reference eigenvalues
    )


def split_bcv_symmetric(M_outer, k_inner, rng, max_rescue_iters=10):
    """Symmetric block-CV split for symmetric similarity matrices.

    Partition n indices into k_inner groups G_1,...,G_k. For each fold i:
      - V_i = principal submatrix S[G_i, G_i] (off-diagonal entries only).
      - Per-fold leakage mask M_fold starts equal to M_outer; if any row/col
        of the training set (Omega minus V_i) would be completely empty,
        randomly un-mask one entry of M_fold per bad row/col (sampled from
        cols outside G_i) until all rows/cols have >=1 training entry.

    Returns list of (V_mask, M_fold_mask) pairs. Both masks are symmetric.
    """
    n = M_outer.shape[0]
    perm = rng.permutation(n)
    fold_groups = np.array_split(perm, k_inner)

    folds = []
    for g in fold_groups:
        g = np.asarray(g)
        V_mask = np.zeros((n, n), dtype=bool)
        if g.size > 0:
            V_mask[np.ix_(g, g)] = True
        np.fill_diagonal(V_mask, False)

        M_fold = M_outer.copy()
        # avail[i, j] = "entry (i,j) is in training (observed and not in V)"
        avail = (~V_mask) & (~M_fold)
        np.fill_diagonal(avail, False)

        for _ in range(max_rescue_iters):
            row_count = avail.sum(axis=1)
            bad = np.where(row_count == 0)[0]
            if bad.size == 0:
                break
            for r_idx in bad:
                # Cols where M_fold[r,c]=True (currently masked by M) AND
                # NOT in V_mask (so un-masking adds to training, not test).
                cands = np.where(M_fold[r_idx] & ~V_mask[r_idx])[0]
                if cands.size == 0:
                    continue  # no rescuable entry; row stays empty (rare)
                c_idx = int(rng.choice(cands))
                M_fold[r_idx, c_idx] = False
                M_fold[c_idx, r_idx] = False
                avail[r_idx, c_idx] = True
                avail[c_idx, r_idx] = True

        folds.append((V_mask, M_fold))
    return folds


def split_omega_into_folds(M_outer, k_inner, rng):
    """Split observed entries (~M_outer) into k_inner equal-sized symmetric folds.
    Returns list of bool masks (True where held out)."""
    n = M_outer.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    observed_pos = ~M_outer[triu_i, triu_j]
    valid = np.where(observed_pos)[0]
    if len(valid) < k_inner:
        return None
    perm = rng.permutation(len(valid))
    fold_groups = np.array_split(perm, k_inner)
    val_masks = []
    for fg in fold_groups:
        m = np.zeros_like(M_outer, dtype=bool)
        idxs = valid[fg]
        ii = triu_i[idxs]; jj = triu_j[idxs]
        m[ii, jj] = True
        m[jj, ii] = True
        val_masks.append(m)
    return val_masks


def fit_admm_score(S, holdout_mask, V_mask, M_mask, U_mask, rank, bounds, seed):
    """Fit ADMM with NaN at holdout, return MSE on V/M/U."""
    from symmnmf.models.admm import ADMM
    x_copy = S.copy()
    x_copy[holdout_mask] = np.nan
    est = ADMM(rank=rank, missing_values=np.nan, bounds=bounds, random_state=seed)
    try:
        est.fit(x_copy)
        rec = est.reconstruct()
    except Exception:
        return np.nan, np.nan, np.nan
    mse_V = float(np.mean((S[V_mask] - rec[V_mask]) ** 2)) if V_mask.any() else np.nan
    mse_M = float(np.mean((S[M_mask] - rec[M_mask]) ** 2)) if M_mask.any() else np.nan
    mse_U = float(np.mean((S[U_mask] - rec[U_mask]) ** 2)) if U_mask.any() else np.nan
    return mse_V, mse_M, mse_U


def nested_cv(S, p_outer, ranks, k_inner=5, n_reps=1, seed=0, n_jobs=2):
    """Nested CV with V/M/U scoring. Returns dict with curves and per-fold raw."""
    from symmnmf.cross_validation import mask_missing_entries

    n = S.shape[0]
    rng = check_random_state(seed)
    bounds = (float(np.nanmin(S)), float(np.nanmax(S)))
    already_nan = np.isnan(S)

    cv_v = np.full((len(ranks), n_reps, k_inner), np.nan)
    cv_m = np.full((len(ranks), n_reps, k_inner), np.nan)
    cv_u = np.full((len(ranks), n_reps, k_inner), np.nan)

    for rep in range(n_reps):
        M_outer = mask_missing_entries(S, float(p_outer), rng, missing_values=np.nan)
        if (~M_outer).sum() < k_inner * 2:
            continue
        val_masks = split_omega_into_folds(M_outer, k_inner, rng)
        if val_masks is None:
            continue

        jobs = []
        for fold_idx, V_i in enumerate(val_masks):
            holdout = V_i | M_outer
            V_valid = V_i & ~already_nan
            M_valid = M_outer & ~already_nan
            U_valid = holdout & ~already_nan
            for ri, r in enumerate(ranks):
                seed_rfk = int(seed + 1000 * (rep + 1) + 100 * fold_idx + ri)
                jobs.append((ri, fold_idx, holdout, V_valid, M_valid, U_valid, r, seed_rfk))

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fit_admm_score)(S, hm, vm, mm, um, r, bounds, sd)
            for (_, _, hm, vm, mm, um, r, sd) in jobs
        )
        for (ri, fold_idx, *_), (mv, mm_, mu) in zip(jobs, results):
            cv_v[ri, rep, fold_idx] = mv
            cv_m[ri, rep, fold_idx] = mm_
            cv_u[ri, rep, fold_idx] = mu

    mean_v = np.nanmean(cv_v.reshape(len(ranks), -1), axis=1)
    mean_m = np.nanmean(cv_m.reshape(len(ranks), -1), axis=1)
    mean_u = np.nanmean(cv_u.reshape(len(ranks), -1), axis=1)
    sem_v = np.nanstd(cv_v.reshape(len(ranks), -1), axis=1) / max(np.sqrt(cv_v.size / len(ranks)), 1)
    sem_m = np.nanstd(cv_m.reshape(len(ranks), -1), axis=1) / max(np.sqrt(cv_m.size / len(ranks)), 1)
    sem_u = np.nanstd(cv_u.reshape(len(ranks), -1), axis=1) / max(np.sqrt(cv_u.size / len(ranks)), 1)
    return dict(
        ranks=list(ranks),
        cv_v=mean_v, cv_m=mean_m, cv_u=mean_u,
        sem_v=sem_v, sem_m=sem_m, sem_u=sem_u,
        cv_v_raw=cv_v, cv_m_raw=cv_m, cv_u_raw=cv_u,
    )


def nested_cv_bcv(S, p_outer, ranks, k_inner=5, n_reps=1, seed=0, n_jobs=2):
    """Symmetric BCV nested CV with V/M scoring.

    Same outer-mask M_outer ~ Bernoulli(1 - p_outer) as `nested_cv`, but the
    inner CV is block (principal submatrix) instead of entrywise. Returns
    dict with mean V/M curves plus per-fold rescue counts (number of
    M_outer entries un-masked into training to avoid empty rows).
    """
    from symmnmf.cross_validation import mask_missing_entries

    n = S.shape[0]
    rng = check_random_state(seed)
    bounds = (float(np.nanmin(S)), float(np.nanmax(S)))
    already_nan = np.isnan(S)

    cv_v = np.full((len(ranks), n_reps, k_inner), np.nan)
    cv_m = np.full((len(ranks), n_reps, k_inner), np.nan)
    rescue_counts = []

    for rep in range(n_reps):
        M_outer = mask_missing_entries(S, float(p_outer), rng, missing_values=np.nan)
        if (~M_outer).sum() < k_inner * 2:
            continue
        bcv_folds = split_bcv_symmetric(M_outer, k_inner, rng)
        if not bcv_folds:
            continue

        for V_i, M_i in bcv_folds:
            # Each rescued entry is symmetric — count upper-triangle only.
            rescue = int(((M_outer & ~M_i) & np.triu(np.ones_like(M_outer, bool), k=1)).sum())
            rescue_counts.append(rescue)

        jobs = []
        for fold_idx, (V_i, M_i) in enumerate(bcv_folds):
            holdout = V_i | M_i
            V_valid = V_i & ~already_nan
            M_valid = M_i & ~already_nan
            U_valid = holdout & ~already_nan
            for ri, r in enumerate(ranks):
                seed_rfk = int(seed + 1000 * (rep + 1) + 100 * fold_idx + ri)
                jobs.append((ri, fold_idx, holdout, V_valid, M_valid, U_valid, r, seed_rfk))

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fit_admm_score)(S, hm, vm, mm, um, r, bounds, sd)
            for (_, _, hm, vm, mm, um, r, sd) in jobs
        )
        for (ri, fold_idx, *_), (mv, mm_, _) in zip(jobs, results):
            cv_v[ri, rep, fold_idx] = mv
            cv_m[ri, rep, fold_idx] = mm_

    mean_v = np.nanmean(cv_v.reshape(len(ranks), -1), axis=1)
    mean_m = np.nanmean(cv_m.reshape(len(ranks), -1), axis=1)
    return dict(
        ranks=list(ranks),
        cv_v=mean_v, cv_m=mean_m,
        rescue_counts=rescue_counts,
    )


def argmin_safe(curve, ranks):
    if not np.isfinite(curve).any():
        return -1
    i = int(np.nanargmin(curve))
    return int(ranks[i])


def argmin_1se_safe(curve, ranks, sem=None):
    """1-SE companion to argmin_safe (2026-05-11, round-1).

    Returns the smallest rank whose CV-MSE is within one standard error of the
    minimum — the standard "1-SE rule" for parsimonious rank selection. When
    the CV curve is sharp (curvature high), `argmin_1se` equals `argmin_safe`;
    when it is flat (the regime where argmin is unstable across k_cv),
    `argmin_1se` collapses toward smaller ranks, exposing the ambiguity.

    Parameters
    ----------
    curve : array-like
        Mean CV-MSE at each rank in `ranks`.
    ranks : list of int
        Rank values corresponding to `curve` entries.
    sem : array-like or None
        Standard error of the mean at each rank. If None, the function falls
        back to argmin_safe (no SE → no 1-SE rule; emit a warning flag via -2).

    Returns
    -------
    int : the chosen rank, or -1 if all entries are NaN, or -2 if `sem` is None.
    """
    curve = np.asarray(curve, dtype=float)
    if not np.isfinite(curve).any():
        return -1
    if sem is None:
        return -2
    sem = np.asarray(sem, dtype=float)
    finite = np.where(np.isfinite(curve) & np.isfinite(sem))[0]
    if finite.size == 0:
        return argmin_safe(curve, ranks)
    am = int(finite[np.argmin(curve[finite])])
    threshold = float(curve[am] + sem[am])
    for i in finite:
        if curve[i] <= threshold:
            return int(ranks[i])
    return int(ranks[am])


def shape_label(curve):
    if not np.isfinite(curve).any():
        return "NaN"
    diffs = np.diff(curve[np.isfinite(curve)])
    if (diffs >= -1e-12).all():
        return "INC"
    if (diffs <= 1e-12).all():
        return "DEC"
    return "elbow"


# ============================================================
# Standard dataset loaders (v3 datasets)
# ============================================================

def load_v3_dataset(label):
    """Returns (S, true_rank). Same datasets as v3."""
    from _generate_examples import (
        _rbf_kernel, simulation, make_similarity_dcsbm,
        make_similarity_multiscale_manifolds, make_similarity_temporal,
        make_correlation_horn_fail_challenging,
    )
    from scipy.spatial.distance import cdist

    if label.startswith("10-block n="):
        n = int(label.split("=")[1])
        rng = np.random.default_rng(0)
        D = simulation(n, 10, ndict=10) + rng.random((n, 10)) * 0.5
        return _rbf_kernel(D, bw=1.0), 10
    elif label == "DCSBM":
        return make_similarity_dcsbm(n=300, K=5, seed=0)[0], 5
    elif label == "Multiscale":
        return make_similarity_multiscale_manifolds(n1=200, n2=200, seed=1)[0], 2
    elif label == "Temporal":
        return make_similarity_temporal(n=400, regimes=4, seed=2)[0], 4
    elif label == "Horn-fail":
        S, _ = make_correlation_horn_fail_challenging(n=200, r_true=3, n_clusters=20, seed=0)
        return S, 3
    elif label == "CLIP_RBF":
        Df = np.load(os.path.join(_PARENT, "examples/p_max0pt02_model_features/clip_vit_b_32.npy"))
        dist = cdist(Df, Df, metric="euclidean")
        bw = float(np.median(dist))
        return np.exp(-(dist / bw) ** 2), None
    elif label == "THINGS":
        A = np.load(os.path.join(_PARENT, "things_behavior_similarity.npy"))
        return (A + A.T) / 2, None
    else:
        raise ValueError(f"unknown dataset label: {label}")


V3_DATASETS = [
    "10-block n=200", "10-block n=400", "10-block n=800",
    "DCSBM", "Multiscale", "Temporal", "Horn-fail",
    "CLIP_RBF", "THINGS",
]


def make_pure_noise_wishart(n, d=None, seed=0):
    """Wishart-like null: S = X X^T / d, X ~ N(0,1)."""
    if d is None:
        d = n  # default to n
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    S = X @ X.T / d
    return S


def make_rank_one_spike(n, snr=2.0, seed=0):
    """S = lambda u u^T + Wishart(n)/n. lambda chosen for given SNR.

    BBP threshold for rank-1 spike + GOE-like noise: lambda > 1 detectable.
    SNR := lambda / 1 (simple).
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n); u /= np.linalg.norm(u)
    W = rng.standard_normal((n, n))
    W = (W + W.T) / 2 / np.sqrt(2 * n)
    S = snr * np.outer(u, u) + W
    return S


def env_info():
    """Capture library versions for reproducibility."""
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ["numpy", "scipy", "sklearn", "umap"]:
        try:
            mod = __import__(pkg.replace("sklearn", "sklearn"))
            info[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            info[pkg] = "not installed"
    return info


# ============================================================
# Recipe J: eta-parameterized k_cv-invariant CV
# ============================================================

def estimate_incoherence(S, k_max):
    """Compute matrix-completion incoherence parameter mu_hat from top-k_max eigenvectors.

    mu_hat = (n / k_max) * max_i sum_{r <= k_max} u_{r,i}^2

    mu_hat in [1, n/k_max]. Smaller = more incoherent = easier matrix completion.

    Returns (mu_hat, row_norms_sq).
    """
    n = S.shape[0]
    eigvals, eigvecs = np.linalg.eigh(S)
    U = eigvecs[:, -k_max:]
    row_norms_sq = np.sum(U ** 2, axis=1)
    mu_hat = float((n / max(k_max, 1)) * np.max(row_norms_sq))
    return mu_hat, row_norms_sq


def _pav_nonincreasing(y):
    """Pool-adjacent-violators isotonic regression for a non-increasing fit.

    Returns an array y_iso of the same shape such that y_iso is non-increasing
    and minimizes sum (y_iso - y)^2 (with uniform weights).
    Standard L2 PAV: O(n).
    """
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n <= 1:
        return y
    # Equal-weight pool: store (sum, count) per active block.
    sums = list(y)
    counts = [1] * n
    starts = list(range(n))   # original-index span tracking not strictly needed for values
    i = 0
    # Walk blocks; merge while non-increasing constraint is violated.
    # Use a stack of (sum, count) blocks.
    stack_sum = []
    stack_cnt = []
    for j in range(n):
        s = y[j]
        c = 1
        # If new block mean > previous block mean, merge (violates non-increasing).
        while stack_sum and stack_sum[-1] / stack_cnt[-1] < s / c:
            s += stack_sum.pop()
            c += stack_cnt.pop()
        stack_sum.append(s)
        stack_cnt.append(c)
    # Expand back into y_iso.
    out = np.empty(n, dtype=float)
    pos = 0
    for s, c in zip(stack_sum, stack_cnt):
        out[pos:pos + c] = s / c
        pos += c
    return out


def recipe_K(spectral_out, delta=0.1, k_cv=5, p_max=None, p_floor=0.5,
             p_floor_mode="adaptive",
             M_min=2000, p_max_floor=0.95,
             use_rayleigh_trace=True, apply_pav=True,
             tr_S_mode="full"):
    """Recipe K (corrected): direct inversion of empirical VE^D curve.

    Corrections vs. the original Recipe K (2026-05-10 audit + 2026-05-11
    round-1 rebuttal):

    1. **Rayleigh-trace VE^D.** The original recipe used the per-rank-projector
       proxy
           VE^D_old(k*, p) = sum_{r<=k*} lambda_r * ||P_r(p) u_r^ref||^2 / tr(S),
       which sums different projectors per term and is *not* the Rayleigh trace.
       The corrected definition is
           VE^D(k*, p) = tr(P_{k*}(p) S) / tr(S),
       a single projector applied to the *full* S. Since 2026-05-11 the spectral
       pass computes `trPkS_boot` via the EXACT identity
           tr(P_k(p) S) = sum_{j' <= k} <U_k[:,j'], S U_k[:,j']>
       (one S0 @ U_k matvec per (p, b)) — not via the rank-Kmax truncated
       reference S_ref^{(Kmax)} = sum_{a<=Kmax} lambda_a u_a u_a^T as in earlier
       releases. The matching denominator is `tr_S_mode="full"` (np.trace(S)).
       `tr_S_mode="truncated"` is retained as a legacy/diagnostic option but is
       no longer self-consistent with the exact numerator and should not be
       used for new analyses. `use_rayleigh_trace=False` falls back to the
       per-rank-projector legacy proxy.

    2. **PAV monotonization.** Empirical delta_emp(p_j) is not guaranteed to be
       monotone. We apply pool-adjacent-violators isotonic regression to enforce
       non-increasing delta_emp(p_j) before inverting, and use the generalized
       inverse p_raw(delta) = inf{p_j : delta_iso(p_j) <= delta}.

    3. **Effective training fraction.** When the cap p_max binds we report
       p_train_eff = p_cv * (k_cv-1)/k_cv (which is < p_target) and the
       evaluated leakage at p_train_eff (`delta_eff`); the user has not actually
       achieved the requested delta when the cap binds.

    4. **Adaptive Wigner-proxy operator-norm safety floor (2026-05-11).**
       `p_floor_mode="adaptive"` (now the **default** in RECIPE_K) uses
       p_floor = x/(1+x) with x = (lambda_{k+1}/lambda_k)^2, which is the
       correct threshold under the (1-p)/p mask-noise variance model. This is
       the canonical Recipe-K floor (manuscript §4.4). The pre-2026-05-11
       formula p_floor = x is available as `p_floor_mode="adaptive_legacy"`
       for reproducibility, and `p_floor_mode="constant"` (with `p_floor`
       supplied) bypasses the data-driven floor entirely.

       NOTE: the canonical library at `src_coherence/experiments/_common.py`
       defaults to `"constant"` for backward compat. RECIPE_K flips this to
       `"adaptive"` so the manuscript-recommended behavior is the default.
       Every shipped experiment script passes `p_floor_mode='adaptive'`
       explicitly, so this default change is transparent for them.

    5. **Diagnostics added (2026-05-11).** The return dict now reports
       `k_cv_min_unclipped` (smallest k_cv at which the cap does not bind),
       bulk-edge probes `lambda_k`, `lambda_kp1`, `lambda_kp5`, `gap_ratio`,
       `bulk_flatness`, `bulk_edge_plausible`, and the alias `p_pool=p_cv`
       (clarifying that p_cv is the observed-pool/training fraction, not the
       missing-probability of M_outer).

    Parameters
    ----------
    delta : float in (0, 1)
        Target spectral fidelity tolerance.
    k_cv : int
        Inner CV folds (used in inflation; variance-only knob).
    p_max : float or None
        Upper safety cap on p_cv. If None, computed from n as
            p_max = max(p_max_floor, 1 - M_min / N_pairs).
    p_floor : float
        Hard floor on p_star (default 0.5).
    M_min : int
        Minimum held-out count for the M-curve diagnostic. Default 2000.
    p_max_floor : float
        Lower bound on auto-computed p_max. Default 0.95.
    use_rayleigh_trace : bool
        If True (default), use the corrected Rayleigh-trace VE^D from
        spectral_out["trPkS_med"]. If False or unavailable, fall back to the
        legacy per-rank-projector proxy (and emit a warning flag).
    apply_pav : bool
        If True (default), apply pool-adjacent-violators monotonization to
        delta_emp(p) before inversion.
    tr_S_mode : "full" or "truncated"
        Denominator for VE^D. "full" uses np.trace(S) (proper); "truncated"
        uses sum of top-Kmax reference eigenvalues (legacy). Both make the
        ratio VE^D / VE^ref consistent — VE^ref also uses the same denominator.
    """
    p_grid = np.asarray(spectral_out["p_grid"])
    evals_ref = np.asarray(spectral_out["evals_ref"])
    k_cut = spectral_out["k_cut"] if "k_cut" in spectral_out else spectral_out["k_smooth"]
    n = int(spectral_out.get("n", len(evals_ref)))

    # Single-signal commitment status (round-2 audit, issue #7). Diagnostic
    # only: p_star is still returned so callers can decide for themselves.
    # "rejected_smooth" callers should not treat k_cut/p_star as committed.
    _flag = spectral_out.get("flag", None)
    if _flag == "sharp":
        status = "accepted"
    elif _flag == "smooth":
        status = "rejected_smooth"
    elif _flag == "borderline":
        status = "borderline"
    else:
        status = "unknown"

    if p_max is None:
        N_pairs = n * (n - 1) / 2
        p_max = float(max(p_max_floor, 1.0 - M_min / max(N_pairs, 1.0)))

    # ---- denominator selection ----
    if tr_S_mode == "full":
        tr_S = float(spectral_out.get("tr_S", float(np.sum(evals_ref))))
    elif tr_S_mode == "truncated":
        tr_S = float(np.sum(evals_ref))
    else:
        raise ValueError(f"tr_S_mode must be 'full' or 'truncated', got {tr_S_mode!r}")

    # ---- VE^D(k*, p_j): proper Rayleigh trace (preferred) or legacy proxy ----
    P = len(p_grid)
    trPkS_med = spectral_out.get("trPkS_med")
    used_rayleigh = bool(use_rayleigh_trace and trPkS_med is not None)
    VE_D = np.zeros(P)
    if used_rayleigh:
        # k_cut here is a rank value; the spectral pass uses k_list = 1..Kmax,
        # so the row index is k_cut - 1.
        ki = max(int(k_cut) - 1, 0)
        VE_D = np.asarray(trPkS_med[ki, :], dtype=float) / tr_S
    else:
        # Legacy proxy: sum_r lambda_r * I_r^proj(p_j)
        I_med = spectral_out.get("I_med")
        if I_med is None:
            Iproj_boot = spectral_out.get("Iproj_boot")
            if Iproj_boot is None:
                raise ValueError("spectral_out must contain trPkS_med, I_med, or Iproj_boot")
            I_med = np.median(Iproj_boot, axis=2)
        lam_top = evals_ref[:k_cut]
        for j in range(P):
            VE_D[j] = float(np.sum(lam_top * I_med[:k_cut, j])) / tr_S

    # VE^ref(k*) under the same denominator for consistency.
    sum_top = float(np.sum(evals_ref[:k_cut]))
    VE_ref_k = sum_top / tr_S

    # ---- empirical deficit ----
    delta_emp_raw = 1.0 - VE_D / max(VE_ref_k, 1e-15)

    # ---- PAV monotonization (non-increasing in p) ----
    idx_sort = np.argsort(p_grid)
    p_sorted = p_grid[idx_sort]
    d_sorted_raw = delta_emp_raw[idx_sort]
    if apply_pav:
        d_sorted_iso = _pav_nonincreasing(d_sorted_raw)
        n_violations = int(np.sum(np.diff(d_sorted_raw) > 1e-12))
    else:
        d_sorted_iso = d_sorted_raw.copy()
        n_violations = int(np.sum(np.diff(d_sorted_raw) > 1e-12))

    # ---- generalized inverse: p_raw(delta) = inf{p_j : d_iso(p_j) <= delta} ----
    feasible_above = False
    if d_sorted_iso[-1] >= delta:
        # All p give deficit >= target → would need p above grid max.
        p_star_raw = float(p_sorted[-1])
    elif d_sorted_iso[0] <= delta:
        # Even smallest p satisfies tolerance → recipe wants p smaller than grid min.
        p_star_raw = float(p_sorted[0])
    else:
        # Bracket the first index k where d_iso[k] <= delta (sorted ascending in p,
        # d_iso non-increasing). With non-increasing PAV, scanning from small p:
        # find first k with d_iso[k] <= delta. Linearly interpolate within the
        # bracket [k-1, k] to refine to the equality crossing.
        k_first = int(np.searchsorted(-d_sorted_iso, -delta, side="left"))
        k_first = max(min(k_first, len(d_sorted_iso) - 1), 1)
        d_lo, d_hi = d_sorted_iso[k_first - 1], d_sorted_iso[k_first]
        p_lo, p_hi = p_sorted[k_first - 1], p_sorted[k_first]
        if d_lo == d_hi:
            p_star_raw = float(p_lo)
        else:
            # delta lies between d_lo (>delta) and d_hi (<=delta); linear interp.
            t = (d_lo - delta) / (d_lo - d_hi)
            p_star_raw = float(p_lo + t * (p_hi - p_lo))
        feasible_above = True

    # ---- floor ----
    # Bulk-edge diagnostics (issue #5, 2026-05-11): always populated so the
    # caller can decide whether the lambda_{k+1} estimate is plausibly a bulk
    # edge (rather than another spike or a deterministic mode).
    bulk_top_idx = min(k_cut, len(evals_ref) - 1)
    lam_k = float(evals_ref[k_cut - 1]) if k_cut >= 1 else float(evals_ref[0])
    lam_kp1 = float(evals_ref[bulk_top_idx])
    bulk_probe_idx = min(k_cut + 5, len(evals_ref) - 1)
    lam_kp5 = float(evals_ref[bulk_probe_idx])
    gap_ratio = float(lam_k / max(lam_kp1, 1e-30))
    bulk_flatness = float(lam_kp5 / max(lam_kp1, 1e-30))
    bulk_edge_plausible = bool((gap_ratio > 2.0) and (bulk_flatness > 0.5))

    if p_floor_mode == "adaptive":
        # Corrected BBP floor (issue #4, 2026-05-11). Under the masked-noise
        # model with conditional variance Var(W_ij | S) = S_ij^2 (1-p)/p,
        # BBP detectability requires lambda^2 > 4 sigma^2(p) n. With
        # sigma_hat = lambda_{k+1}/(2 sqrt(n)), let
        #   x := 4 sigma_hat^2 n / lambda_k^2 = (lambda_{k+1}/lambda_k)^2.
        # Solving 1 > x/p_floor (i.e. (1-p)/p version) yields
        #   p_floor = x / (1 + x).
        # Earlier releases used p_floor = x, which is the threshold under a
        # different model where the *effective* noise variance scales like
        # sigma^2/p (no (1-p) factor). Pass p_floor_mode="adaptive_legacy" to
        # reproduce that behaviour.
        sigma_hat = lam_kp1 / (2.0 * np.sqrt(max(n, 1)))
        x = (4.0 * sigma_hat ** 2 * n) / max(lam_k ** 2, 1e-12)
        p_floor_used = float(x / (1.0 + x))
        p_floor_used = float(min(max(p_floor_used, 0.0), 1.0))
    elif p_floor_mode == "adaptive_legacy":
        # Pre-2026-05-11 formula. Kept for reproducibility.
        sigma_hat = lam_kp1 / (2.0 * np.sqrt(max(n, 1)))
        p_floor_used = (4.0 * sigma_hat ** 2 * n) / max(lam_k ** 2, 1e-12)
        p_floor_used = float(min(max(p_floor_used, 0.0), 1.0))
    elif p_floor_mode == "constant":
        sigma_hat = None
        p_floor_used = float(p_floor)
    else:
        raise ValueError(f"Unknown p_floor_mode={p_floor_mode!r}")

    floor_binding = p_star_raw < p_floor_used
    p_target = max(p_star_raw, p_floor_used)

    # ---- Recipe-I inflation, capped ----
    p_cv_unclipped = p_target * k_cv / max(k_cv - 1, 1)
    p_cv = float(min(p_cv_unclipped, p_max))
    cap_binding = bool(p_cv_unclipped > p_max)
    feasible_inflation = not cap_binding
    p_train_eff = float(p_cv * (k_cv - 1) / k_cv)

    # Minimum k_cv needed to avoid cap binding at this p_target (issue #11).
    # Derived from p_cv_unclipped = p_target * k / (k - 1) <= p_max:
    #   k >= p_max / (p_max - p_target).
    # Clip to >= 2: k_cv = 1 is not a valid CV count (round-2 audit, issue #13).
    if (p_max > p_target) and (p_max - p_target > 1e-12):
        k_cv_min_unclipped = max(2, int(np.ceil(p_max / (p_max - p_target))))
    else:
        k_cv_min_unclipped = None   # infeasible: would need k_cv -> infinity

    # ---- evaluate leakage at the *effective* training density ----
    # Interpolate d_iso at p_train_eff (sorted ascending by p).
    if p_train_eff <= p_sorted[0]:
        delta_eff = float(d_sorted_iso[0])
    elif p_train_eff >= p_sorted[-1]:
        delta_eff = float(d_sorted_iso[-1])
    else:
        delta_eff = float(np.interp(p_train_eff, p_sorted, d_sorted_iso))

    return dict(
        recipe="K",
        status=str(status),   # round-2: diagnostic commitment label
        delta=float(delta), k_cv=int(k_cv),
        # Operating points: p_target is post-floor, pre-inflation; p_train_eff is
        # the actual post-CV-split training density (= p_target iff cap not binding).
        p_target=float(p_target),
        p_star_raw=float(p_star_raw),
        p_star=float(p_target),       # alias for backwards compat
        p_floor=float(p_floor_used), p_floor_input=float(p_floor),
        p_floor_mode=str(p_floor_mode), sigma_hat=sigma_hat,
        floor_binding=bool(floor_binding),
        p_cv=p_cv, p_cv_unclipped=float(p_cv_unclipped),
        p_pool=p_cv,                  # alias: p_cv is the observed-pool fraction,
                                      # not the missing-probability of M_outer
        p_train_eff=p_train_eff,
        p_train=p_train_eff,          # alias for backwards compat
        delta_eff=float(delta_eff),   # leakage actually achieved at p_train_eff
        cap_binding=cap_binding,
        k_cv_min_unclipped=k_cv_min_unclipped,   # min k_cv to avoid cap binding
        p_max=float(p_max), M_min=int(M_min), n=int(n),
        feasible_inversion=bool(feasible_above),
        feasible_inflation=bool(feasible_inflation),
        # Bulk-edge diagnostics for the BBP floor (issue #5):
        lambda_k=lam_k, lambda_kp1=lam_kp1, lambda_kp5=lam_kp5,
        gap_ratio=gap_ratio, bulk_flatness=bulk_flatness,
        bulk_edge_plausible=bulk_edge_plausible,
        # Curve diagnostics:
        delta_emp_raw=d_sorted_raw.tolist(),
        delta_emp=d_sorted_iso.tolist(),   # PAV-monotonized; backwards-compat name
        delta_emp_iso=d_sorted_iso.tolist(),
        p_grid=p_sorted.tolist(),
        n_monotonicity_violations=int(n_violations),
        used_rayleigh_trace=used_rayleigh,
        tr_S_mode=str(tr_S_mode), tr_S=float(tr_S),
        VE_ref_k=float(VE_ref_k), k_cut=int(k_cut),
    )


def recipe_J(S, spectral_out, eta=0.05, k_cv=5, k_max=None,
             alpha_0=0.028, p_min=0.5, p_max=0.95):
    """eta-parameterized Recipe J operating point.

    Phase-transition formula:
        1 - p_star = alpha(eta) * mu_hat^2 * k_max * log^2(n) / n
        alpha(eta) = alpha_0 * log(1/eta) / log(20)

    Then Recipe I inflation gives p_cv = p_star * k_cv / (k_cv - 1) so the
    POST-CV-split training fraction is exactly p_star (k_cv-invariant in expectation).

    alpha_0 is empirically-calibrated for typical n ~ 200-1000 to land in the
    discriminative regime; default 0.028 gives ~20% masking at eta=0.05 for n=400.

    Returns dict with mu_hat, eta, p_star, p_cv, p_train, k_max, ...
    """
    n = S.shape[0]
    k_cut = spectral_out["k_cut"] if "k_cut" in spectral_out else spectral_out["k_smooth"]
    if k_max is None:
        k_max = min(int(2 * k_cut + 1), n // 4, 100)
        k_max = max(k_max, k_cut)

    mu_hat, _ = estimate_incoherence(S, k_max)

    alpha_eta = alpha_0 * np.log(1.0 / eta) / np.log(20.0)
    log_n_sq = np.log(n) ** 2
    one_minus_p = alpha_eta * (mu_hat ** 2) * k_max * log_n_sq / n
    one_minus_p = float(min(max(one_minus_p, 1 - p_max), 1 - p_min))
    p_star = 1.0 - one_minus_p

    p_cv_raw = p_star * k_cv / (k_cv - 1)
    p_cv = min(p_cv_raw, p_max)
    feasible = p_cv_raw <= p_max
    p_train = p_cv * (k_cv - 1) / k_cv

    return dict(
        recipe="J", eta=float(eta), k_cv=int(k_cv), k_max=int(k_max),
        mu_hat=float(mu_hat), alpha_eta=float(alpha_eta),
        p_star=float(p_star), p_cv=float(p_cv), p_train=float(p_train),
        feasible=bool(feasible), n=int(n), k_cut=int(k_cut),
    )
