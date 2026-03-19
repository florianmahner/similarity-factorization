"""Subspace coherence for rank selection.

Measures how well the top-k eigensubspace of a similarity matrix survives
random subsampling. Uses the block Frobenius norm of the cross-Gram matrix
(mean squared canonical correlation) instead of per-eigenvector projected
coherence, making it robust to eigenvector swapping in continuous spectra.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.linalg as la
from joblib import Parallel, delayed
from tqdm import tqdm

# Import bootstrap helpers from Ka Chun's v2
_V2_DIR = str(Path(__file__).resolve().parent.parent / "kachun" / "v2")
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from _compute_coherence import (
    _symmetrize_with_nan,
    _prepare_observation_mask,
    _masked_unbiased_spd_missing_from_uniform,
    _topk_eigenvectors_exact_warm,
    _orthonormalize_columns,
    _randomized_topk_eigenspace_symmetric,
)


# =============================================================================
# Core math: S_k and null computation
# =============================================================================


def _compute_subspace_and_iproj(g2, k_idx):
    """Extract incremental subspace coherence and I^proj_k from G^2.

    Parameters
    ----------
    g2 : (Kmax, Kmax) array
        Element-wise squared cross-Gram matrix: G[i,j]^2 = (u_i^boot . u_j^ref)^2
    k_idx : (K_count,) int array
        Zero-based indices into k_list (i.e., k_list - 1).

    Returns
    -------
    delta_k : (K_count,) array
        Incremental subspace coherence:
        Delta_k = ||G[:k,:k]||_F^2 - ||G[:k-1,:k-1]||_F^2
        This is the marginal improvement from adding the k-th dimension.
        Includes the k-th row of G[:k,k-1] and k-th column G[k-1,:k],
        capturing eigenvector rotation (unlike per-eigenvector I^proj).
    iproj_k : (K_count,) array
        Per-eigenvector coherence: I^proj_k = sum_{i<=k} G[i,k]^2
    """
    kmax = g2.shape[0]

    # I^proj: diagonal of row-cumsum (same as v2)
    iproj_all = np.cumsum(g2, axis=0)[np.arange(kmax), np.arange(kmax)]

    # Block Frobenius norms via 2D cumsum
    block_sum = np.cumsum(np.cumsum(g2, axis=0), axis=1)
    block_fro = block_sum[np.arange(kmax), np.arange(kmax)]

    # Incremental: Delta_k = ||G[:k,:k]||_F^2 - ||G[:k-1,:k-1]||_F^2
    delta_all = np.empty(kmax, dtype=float)
    delta_all[0] = block_fro[0]
    delta_all[1:] = block_fro[1:] - block_fro[:-1]

    return (
        np.clip(delta_all[k_idx], 0.0, None),
        np.clip(iproj_all[k_idx], 0.0, 1.0),
    )


def _compute_null_subspace(u_boot, k_idx, b_null, rng):
    """Compute null distribution for incremental subspace coherence.

    Parameters
    ----------
    u_boot : (n, Kmax) array
        Top-Kmax bootstrap eigenvectors.
    k_idx : (K_count,) int array
        Zero-based indices into k_list.
    b_null : int
        Number of null samples.
    rng : numpy Generator
        Random state.

    Returns
    -------
    null_delta : (K_count, b_null) array
        Null Delta_k values for each tested k.
    """
    n, kmax = u_boot.shape
    k_count = len(k_idx)
    null_delta = np.zeros((k_count, b_null), dtype=float)

    for m in range(b_null):
        v = rng.standard_normal((n, kmax))
        v /= np.maximum(la.norm(v, axis=0, keepdims=True), 1e-12)

        g_null = u_boot.T @ v
        g2_null = g_null * g_null

        block_sum = np.cumsum(np.cumsum(g2_null, axis=0), axis=1)
        block_fro = block_sum[np.arange(kmax), np.arange(kmax)]

        delta_null_all = np.empty(kmax, dtype=float)
        delta_null_all[0] = block_fro[0]
        delta_null_all[1:] = block_fro[1:] - block_fro[:-1]

        null_delta[:, m] = np.clip(delta_null_all[k_idx], 0.0, None)

    return null_delta


# =============================================================================
# Bootstrap worker (one replicate)
# =============================================================================


def _subspace_worker_one_boot(args):
    """Compute S_k, I^proj_k, and null S_k for one bootstrap replicate."""
    (
        b, p_list, s0, w, u_ref_k, k_idx,
        random_seed_base, keep_diag,
        solver, solver_tol, solver_maxiter, residual_tol,
        use_nested_masks, b_null,
    ) = args

    n = s0.shape[0]
    kmax = u_ref_k.shape[1]
    k_count = len(k_idx)
    p_list = np.asarray(p_list, float)
    p_count = len(p_list)
    iu = np.triu_indices(n, k=1)

    rng = np.random.default_rng((int(random_seed_base) + 9176 * int(b)) & 0xFFFFFFFF)
    edge_u = rng.random(iu[0].size) if use_nested_masks else None

    s_boot_b = np.zeros((k_count, p_count), float)
    iproj_boot_b = np.zeros((k_count, p_count), float)
    null_s_boot_b = np.zeros((k_count, p_count, b_null), float) if b_null > 0 else None

    x_init = np.array(u_ref_k, copy=True)

    for j, p in enumerate(p_list):
        if use_nested_masks:
            eu = edge_u
        else:
            eu = rng.random(iu[0].size)

        a = _masked_unbiased_spd_missing_from_uniform(
            s0, w, float(p), eu, iu, keep_diag=keep_diag,
        )

        _, u_k, info = _topk_eigenvectors_exact_warm(
            a, kmax, X_init=x_init,
            solver=solver, tol=solver_tol,
            maxiter=solver_maxiter, residual_tol=residual_tol,
        )

        x_init = np.array(u_k, copy=True)

        g = u_k.T @ u_ref_k
        g2 = g * g

        s_boot_b[:, j], iproj_boot_b[:, j] = _compute_subspace_and_iproj(g2, k_idx)

        if null_s_boot_b is not None:
            null_rng = np.random.default_rng(
                (int(random_seed_base) + 7919 * int(b) + 6271 * int(j)) & 0xFFFFFFFF
            )
            null_s_boot_b[:, j, :] = _compute_null_subspace(
                u_k, k_idx, b_null, null_rng,
            )

    return b, s_boot_b, iproj_boot_b, null_s_boot_b


# =============================================================================
# Main orchestrator
# =============================================================================


def compute_subspace_coherence(
    s,
    k_list,
    p_list,
    b=100,
    random_state=0,
    keep_diag=True,
    solver="auto",
    solver_tol=1e-10,
    solver_maxiter=200,
    residual_tol=1e-8,
    use_nested_masks=True,
    b_null=20,
    alpha_tau=0.95,
    ci_level=0.95,
    use_baseline_correction=True,
    n_jobs=None,
    show_progress=True,
    ref_oversample=10,
    ref_n_iter=2,
):
    """Compute subspace coherence across multiple k and masking fractions p.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix, may contain NaN.
    k_list : list[int]
        Candidate dimensions to test.
    p_list : array-like
        Sampling fraction grid, sorted increasing in (0, 1].
    b : int
        Bootstrap replicates.
    random_state : int
        Seed for reproducibility.
    b_null : int
        Null samples per (bootstrap, p) cell. Pooled total = b * b_null.
    alpha_tau : float
        Quantile for null threshold (0.95 = 95th percentile).
    ci_level : float
        Confidence level for bootstrap CIs.
    use_baseline_correction : bool
        Apply (S_k - k/n) / (1 - k/n) correction.
    n_jobs : int or None
        Parallel workers. None = cpu_count - 1.

    Returns
    -------
    result : dict
    """
    s_sym = _symmetrize_with_nan(s)
    n = s_sym.shape[0]

    k_list = np.asarray(sorted(set(k_list)), int)
    p_list = np.asarray(p_list, float)
    kmax = int(k_list.max())
    k_count = len(k_list)
    p_count = len(p_list)
    k_idx = k_list - 1

    cpu = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs_eff = max(1, cpu - 1)
    else:
        n_jobs_eff = int(max(1, min(int(n_jobs), max(1, cpu - 1))))

    s0, w, q = _prepare_observation_mask(s_sym, keep_diag=keep_diag)

    # Reference eigenspace
    has_missing = q < (1.0 - 1e-15)
    if has_missing:
        s_ref = w * s0
        if keep_diag:
            np.fill_diagonal(s_ref, np.diag(s0))
        s_ref = 0.5 * (s_ref + s_ref.T)
    else:
        s_ref = 0.5 * (s0 + s0.T)

    evals_all, evecs_all = la.eigh(s_ref)
    idx_ref = np.argsort(evals_all)[::-1][:kmax]
    evals_ref = evals_all[idx_ref].astype(float)
    u_ref_k = _orthonormalize_columns(evecs_all[:, idx_ref].astype(float))

    # Dispatch workers
    tasks = [
        (
            boot, p_list, s0, w, u_ref_k, k_idx,
            int(random_state or 0), bool(keep_diag),
            str(solver), float(solver_tol), int(solver_maxiter), float(residual_tol),
            bool(use_nested_masks), int(b_null),
        )
        for boot in range(b)
    ]

    if n_jobs_eff == 1 or b <= 1:
        iterator = tqdm(tasks, desc="subspace coherence") if show_progress else tasks
        results = [_subspace_worker_one_boot(t) for t in iterator]
    else:
        if show_progress:
            results = Parallel(n_jobs=n_jobs_eff, backend="loky", prefer="processes")(
                delayed(_subspace_worker_one_boot)(t)
                for t in tqdm(tasks, desc=f"subspace coherence (n_jobs={n_jobs_eff})")
            )
        else:
            results = Parallel(n_jobs=n_jobs_eff, backend="loky", prefer="processes")(
                delayed(_subspace_worker_one_boot)(t) for t in tasks
            )

    results.sort(key=lambda x: x[0])

    # Aggregate
    delta_boot = np.zeros((k_count, p_count, b), float)
    iproj_boot = np.zeros((k_count, p_count, b), float)
    null_delta_boot = np.zeros((k_count, p_count, b, b_null), float) if b_null > 0 else None

    for boot_idx, delta_b, ip_b, null_b in results:
        delta_boot[:, :, boot_idx] = delta_b
        iproj_boot[:, :, boot_idx] = ip_b
        if null_delta_boot is not None and null_b is not None:
            null_delta_boot[:, :, boot_idx, :] = null_b

    # No analytical baseline correction for Delta_k -- we test directly
    # against the empirical null distribution. Just pass through raw values.

    # Summary statistics on delta
    ci_lo_q = (1.0 - float(ci_level)) / 2.0
    ci_hi_q = 1.0 - ci_lo_q
    delta_mean = delta_boot.mean(axis=2)
    delta_ci_lo = np.quantile(delta_boot, ci_lo_q, axis=2)
    delta_ci_hi = np.quantile(delta_boot, ci_hi_q, axis=2)

    # Null thresholds from empirical null
    tau_kp = None
    if null_delta_boot is not None:
        null_flat = null_delta_boot.reshape(k_count, p_count, -1)
        tau_kp = np.quantile(null_flat, float(alpha_tau), axis=2)

    return {
        "delta_boot": delta_boot,
        "Iproj_boot": iproj_boot,
        "null_delta_boot": null_delta_boot,
        "delta_mean": delta_mean,
        "delta_ci_lo": delta_ci_lo,
        "delta_ci_hi": delta_ci_hi,
        "tau_kp": tau_kp,
        "evals_ref": evals_ref,
        "U_ref_K": u_ref_k,
        "p": p_list,
        "k_list": k_list,
        "n": n,
        "q_offdiag_obs_rate": float(q),
        "B": b,
        "B_null": b_null,
        "alpha_tau": alpha_tau,
        "ci_level": ci_level,
    }


# =============================================================================
# Rank selection with FDR
# =============================================================================


@dataclass
class RankResult:
    k_star: int
    p_star: float | None
    pvalues: np.ndarray
    significant: np.ndarray
    kappa_hat: np.ndarray
    kappa_k_cut: int
    signal_mask: np.ndarray
    noise_ref_curve: np.ndarray | None
    signal_ref_curve: np.ndarray | None


def _bh_fdr(pvalues, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean rejection mask."""
    pv = np.asarray(pvalues, float)
    m = pv.size
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pv)
    sorted_pv = pv[order]
    thresholds = np.arange(1, m + 1) / m * q
    reject_sorted = np.zeros(m, dtype=bool)
    candidates = np.where(sorted_pv <= thresholds)[0]
    if len(candidates) > 0:
        cutoff = candidates[-1]
        reject_sorted[: cutoff + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def _estimate_kappa(s_mean, p_list, hi_band_quantile=0.85):
    """Kappa from subspace coherence: ell_k(p) = (1 - S_k(p)) * p/(1-p)."""
    p_list = np.asarray(p_list, float)
    p_hi = float(np.quantile(p_list, hi_band_quantile))
    hi_idx = np.where(p_list >= p_hi)[0]
    if hi_idx.size == 0:
        hi_idx = np.array([len(p_list) - 1], int)
    scale = p_list[hi_idx] / np.maximum(1.0 - p_list[hi_idx], 1e-12)
    ell = (1.0 - s_mean[:, hi_idx]) * scale[None, :]
    kappa_hat = np.median(ell, axis=1)
    return kappa_hat


def _kappa_changepoint(kappa_hat, k_list, smooth_window=3, min_k=2):
    """Find k_cut via largest jump in smoothed kappa."""
    k = kappa_hat.size
    if k < 3:
        return int(k_list[-1])
    sm = np.copy(kappa_hat)
    half = smooth_window // 2
    for i in range(k):
        a, b_end = max(0, i - half), min(k, i + half + 1)
        sm[i] = np.median(kappa_hat[a:b_end])
    d = sm[1:] - sm[:-1]
    valid = np.where(np.asarray(k_list[:-1]) >= min_k)[0]
    if valid.size == 0:
        valid = np.arange(k - 1)
    i_star = int(valid[np.argmax(d[valid])])
    return int(k_list[i_star])


def select_rank(
    result,
    fdr_q=0.05,
    noise_quantile=0.90,
    noise_floor_multiplier=3.0,
    lift_margin=0.0,
    hi_band_quantile=0.85,
    smooth_window=3,
):
    """Select rank k* and sampling fraction p* from subspace coherence results.

    Two-stage approach:
    1. Kappa changepoint on Delta_k gives a conservative k_cut (where the
       steepest spectral transition is). Dimensions k > noise_floor_multiplier * k_cut
       are treated as "definitely noise."
    2. For each dimension k, compute p-value against the empirical noise
       distribution (Delta_k values of the noise dimensions at p_max).
       Apply BH-FDR. k* = last significant dimension.

    This avoids the random-direction null (which is too weak because noise
    eigenvectors of S still project significantly onto the bootstrap eigenspace).

    Parameters
    ----------
    result : dict
        Output of compute_subspace_coherence.
    fdr_q : float
        FDR level for BH correction.
    noise_floor_multiplier : float
        Dimensions with k > multiplier * kappa_k_cut form the empirical null.
    """
    k_list = result["k_list"]
    p_list = result["p"]
    delta_boot = result["delta_boot"]
    delta_ci_lo = result["delta_ci_lo"]
    delta_ci_hi = result["delta_ci_hi"]
    delta_mean = result["delta_mean"]
    k_count = len(k_list)

    # Stage 1: Kappa changepoint for conservative noise floor
    kappa_hat = _estimate_kappa(delta_mean, p_list, hi_band_quantile)
    kappa_k_cut = _kappa_changepoint(kappa_hat, k_list, smooth_window)

    # Define noise dimensions: k > noise_floor_multiplier * kappa_k_cut
    noise_threshold = noise_floor_multiplier * kappa_k_cut
    noise_indices = np.where(k_list > noise_threshold)[0]

    if len(noise_indices) < 3:
        # Fallback: use top third of k_list as noise
        noise_indices = np.arange(k_count * 2 // 3, k_count)
        warnings.warn(
            f"Too few noise dimensions above {noise_threshold}. "
            f"Using top third of k_list as fallback."
        )

    # Stage 2: Empirical p-values against noise distribution
    # At p_max: pool Delta_k across all noise dimensions and all bootstrap replicates
    # This gives us the distribution of "what Delta_k looks like for noise eigenvectors"
    noise_delta_at_pmax = delta_boot[noise_indices, -1, :].ravel()

    # For each k, compare its median Delta_k against the noise distribution
    obs_at_pmax = np.median(delta_boot[:, -1, :], axis=1)
    pvalues = np.array([
        float(np.mean(noise_delta_at_pmax >= obs_at_pmax[kk]))
        for kk in range(k_count)
    ])

    significant = _bh_fdr(pvalues, q=fdr_q)

    if np.any(significant):
        k_star = int(k_list[np.where(significant)[0][-1]])
    else:
        k_star = 0
        warnings.warn("No dimension passed the FDR threshold. Returning k*=0.")

    # Signal/noise classification based on k*
    signal_mask = k_list <= k_star if k_star > 0 else np.zeros(k_count, dtype=bool)
    noise_mask = ~signal_mask

    # p* via signal-noise liftoff
    p_star = None
    noise_ref_curve = None
    signal_ref_curve = None

    if k_star > 0 and np.any(noise_mask):
        noise_pool = delta_ci_hi[noise_mask]
        noise_ref_curve = np.quantile(noise_pool, noise_quantile, axis=0)

        boundary_idx = np.where(signal_mask)[0][-1]
        signal_ref_curve = delta_ci_lo[boundary_idx]

        gap = signal_ref_curve - noise_ref_curve
        above = gap >= lift_margin
        idxs = np.where(above)[0]
        if len(idxs) > 0:
            p_star = float(p_list[idxs[0]])

    return RankResult(
        k_star=k_star,
        p_star=p_star,
        pvalues=pvalues,
        significant=significant,
        kappa_hat=kappa_hat,
        kappa_k_cut=kappa_k_cut,
        signal_mask=signal_mask,
        noise_ref_curve=noise_ref_curve,
        signal_ref_curve=signal_ref_curve,
    )


# =============================================================================
# Plotting
# =============================================================================


def plot_rank_selection(result, rank_result, output_dir):
    """Generate all diagnostic plots and save to output_dir."""
    import matplotlib.pyplot as plt

    _src_dir = str(Path(__file__).resolve().parents[3])
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT, GRAY_DARK
    from src.utils.figure_theme import create_figure, despine

    output_dir = Path(output_dir)
    k_list = result["k_list"]
    p_list = result["p"]
    delta_mean = result["delta_mean"]
    tau_kp = result["tau_kp"]
    k_star = rank_result.k_star
    saved = []

    # 1. Heatmap of Delta_k(p)
    fig, ax = create_figure("wide")
    im = ax.imshow(
        delta_mean, aspect="auto", origin="lower", interpolation="nearest",
        extent=[p_list[0], p_list[-1], k_list[0] - 2.5, k_list[-1] + 2.5],
        cmap="magma",
    )
    if k_star > 0:
        ax.axhline(k_star + 2.5, color="white", linewidth=1.5, linestyle="--")
        ax.text(p_list[1], k_star + 4, f"k* = {k_star}", color="white", fontsize=8)
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Dimension $k$")
    ax.set_title("Incremental subspace coherence $\\Delta_k(p)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = output_dir / "subspace_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 2. Delta_k at p_max with noise floor
    delta_boot = result["delta_boot"]
    d_at_pmax = np.median(delta_boot[:, -1, :], axis=1)
    d_lo_pmax = np.quantile(delta_boot[:, -1, :], 0.05, axis=1)
    d_hi_pmax = np.quantile(delta_boot[:, -1, :], 0.95, axis=1)

    # Compute empirical noise floor from noise dimensions
    noise_threshold_k = 3.0 * rank_result.kappa_k_cut
    noise_idx = np.where(k_list > noise_threshold_k)[0]
    if len(noise_idx) >= 3:
        noise_floor_median = float(np.median(d_at_pmax[noise_idx]))
        noise_floor_95 = float(np.quantile(
            delta_boot[noise_idx, -1, :].ravel(), 0.95
        ))
    else:
        noise_floor_median = None
        noise_floor_95 = None

    fig, ax = create_figure("wide")
    ax.fill_between(k_list, d_lo_pmax, d_hi_pmax, color=TEAL, alpha=0.15)
    ax.plot(k_list, d_at_pmax, marker="o", markersize=4, linewidth=1.5,
            color=TEAL, label="$\\Delta_k(p_{\\max})$")
    if noise_floor_95 is not None:
        ax.axhline(noise_floor_95, linestyle="--", linewidth=1.5,
                   color=ROSE, label=f"Noise 95th pctl = {noise_floor_95:.3f}")
    if noise_floor_median is not None:
        ax.axhline(noise_floor_median, linestyle=":", linewidth=1,
                   color=GRAY, label=f"Noise median = {noise_floor_median:.3f}")
    if k_star > 0:
        ax.axvline(k_star, linestyle=":", linewidth=1, color=GRAY_DARK,
                   label=f"$k^*$ = {k_star}")
    ax.axvline(rank_result.kappa_k_cut, linestyle=":", linewidth=1, color=GRAY_LIGHT,
               label=f"Kappa $k_{{cut}}$ = {rank_result.kappa_k_cut}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel(f"Incremental subspace coherence at $p = {p_list[-1]:.2f}$")
    ax.set_title("Subspace rank selection")
    ax.legend(fontsize=6)
    despine(ax)
    path = output_dir / "subspace_rank_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 3. Kappa curve
    fig, ax = create_figure("wide")
    ax.plot(k_list, rank_result.kappa_hat, marker=".", markersize=5,
            linewidth=1, color=GRAY_DARK)
    ax.scatter(k_list[rank_result.signal_mask],
              rank_result.kappa_hat[rank_result.signal_mask],
              s=30, color=TEAL, zorder=5, label="Signal")
    ax.scatter(k_list[~rank_result.signal_mask],
              rank_result.kappa_hat[~rank_result.signal_mask],
              s=15, color=GRAY_LIGHT, zorder=4, label="Noise")
    ax.axvline(rank_result.kappa_k_cut + 2.5, linestyle="--", linewidth=1.5,
               color=ROSE, label=f"Kappa $k_{{cut}}$ = {rank_result.kappa_k_cut}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("$\\hat{\\kappa}_k$")
    ax.set_title("Kappa diagnostic (secondary)")
    ax.legend(fontsize=7)
    despine(ax)
    path = output_dir / "kappa_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 4. Comparison: Delta_k vs I^proj at p_max
    iproj_at_pmax = np.median(result["Iproj_boot"][:, -1, :], axis=1)

    fig, axes = create_figure("full_width", ncols=2)
    ax1, ax2 = axes

    ax1.plot(k_list, d_at_pmax, marker="o", markersize=3, linewidth=1.2,
             color=TEAL, label="$\\Delta_k$ (incremental subspace)")
    if noise_floor_95 is not None:
        ax1.axhline(noise_floor_95, linestyle="--", linewidth=1, color=ROSE,
                    label=f"Noise 95th pctl")
    if k_star > 0:
        ax1.axvline(k_star, linestyle=":", color=GRAY_DARK, linewidth=0.8)
    ax1.set_xlabel("Dimension $k$")
    ax1.set_ylabel("Incremental subspace coherence")
    ax1.set_title("$\\Delta_k$ (this work)")
    ax1.legend(fontsize=5)
    despine(ax1)

    ax2.plot(k_list, iproj_at_pmax, marker="o", markersize=3, linewidth=1.2,
             color=CYAN, label="$I^{proj}_k$ (per-eigenvector)")
    ax2.set_xlabel("Dimension $k$")
    ax2.set_ylabel("Per-eigenvector coherence")
    ax2.set_title("$I^{proj}_k$ (Ka Chun v2)")
    ax2.legend(fontsize=5)
    despine(ax2)

    path = output_dir / "subspace_vs_iproj.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 5. Signal-noise liftoff for p*
    if rank_result.signal_ref_curve is not None and rank_result.noise_ref_curve is not None:
        fig, ax = create_figure("wide")
        ax.plot(p_list, rank_result.signal_ref_curve, linewidth=1.5, color=TEAL,
                label=f"Signal ref ($k$={k_star})")
        ax.plot(p_list, rank_result.noise_ref_curve, linewidth=1.5, linestyle="--",
                color=ROSE, label="Noise ref")
        gap = rank_result.signal_ref_curve - rank_result.noise_ref_curve
        ax.plot(p_list, gap, linewidth=1, linestyle=":", color=GRAY_DARK, label="Gap")
        ax.axhline(0, color=GRAY_LIGHT, linewidth=0.5)
        if rank_result.p_star is not None:
            ax.axvline(rank_result.p_star, color="k", linewidth=1, linestyle="--",
                       label=f"$p^*$ = {rank_result.p_star:.3f}")
        ax.set_xlabel("Sampling fraction $p$")
        ax.set_ylabel("Coherence")
        ax.set_title(f"Signal-noise liftoff ($k^*$={k_star})")
        ax.legend(fontsize=6)
        despine(ax)
        path = output_dir / "signal_noise_liftoff.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(path)

    return saved
