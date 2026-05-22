"""
Rank-selection benchmarks for Experiment A.

Each method takes a symmetric similarity matrix S and returns an integer k_hat
(estimated rank/dimension), plus diagnostic info.

Methods implemented:
  bench_kneedle           : Kneedle elbow on log-eigenvalue scree
  bench_eigengap          : eigengap heuristic (von Luxburg 2007)
  bench_donoho_gavish     : Donoho-Gavish 4/sqrt(3) optimal hard threshold (2014)
  bench_screenot          : ScreeNOT (Donoho-Gavish-Romanov 2023)  [pip:ScreeNOT]
  bench_horn_pa           : Horn's parallel analysis (1965; Dinno 2009 implementation)
  bench_recipe_K_argmin   : Recipe K's k_cut from spectral pass (uses ours)

Each returns dict with at least: k_hat (int), runtime_s, name, diagnostics.
Note: Recipe K's full pipeline (spectral pass + δ_emp inversion + CV argmin) is
called separately in the experiment script; here `bench_recipe_K_kcut`
returns only the spectral k_cut for spectrum-only comparison.
"""
import time
import numpy as np
from scipy import linalg as sla


# ============================================================
# Eigenvalue-based benchmarks
# ============================================================

def _topk_eigvals(S, kmax=None):
    eigs = np.linalg.eigvalsh(S)[::-1]
    if kmax is not None:
        eigs = eigs[:kmax]
    return eigs


def bench_kneedle(S, kmax=None, **_):
    """Kneedle (Satopaa 2011) elbow on log-eigenvalue scree."""
    t0 = time.time()
    try:
        from kneed import KneeLocator
    except ImportError:
        return dict(name="kneedle", k_hat=-1, runtime_s=0.0, error="kneed not installed")
    eigs = _topk_eigvals(S, kmax or min(S.shape[0] // 2, 100))
    eigs_pos = np.maximum(eigs, 1e-12)
    log_eigs = np.log(eigs_pos)
    x = np.arange(1, len(log_eigs) + 1)
    try:
        kl = KneeLocator(x, log_eigs, curve="convex", direction="decreasing")
        k_hat = int(kl.knee) if kl.knee is not None else -1
    except Exception:
        k_hat = -1
    return dict(name="kneedle", k_hat=k_hat, runtime_s=time.time() - t0,
                eigvals=eigs.tolist())


def bench_eigengap(S, kmax=None, k_floor=1, **_):
    """Eigengap heuristic: k_hat = argmax_r (lambda_r - lambda_{r+1}).
    Uses the LARGEST gap among the top-kmax eigenvalues.
    """
    t0 = time.time()
    eigs = _topk_eigvals(S, kmax or min(S.shape[0] // 2, 100))
    diffs = eigs[:-1] - eigs[1:]
    valid = np.arange(k_floor, len(diffs))  # gap at index r is lambda_r - lambda_{r+1}
    if valid.size == 0:
        return dict(name="eigengap", k_hat=-1, runtime_s=time.time() - t0)
    i_star = int(valid[np.argmax(diffs[valid])])
    return dict(name="eigengap", k_hat=i_star + 1, runtime_s=time.time() - t0,
                gap_size=float(diffs[i_star]))


def bench_donoho_gavish(S, **_):
    """Donoho-Gavish (2014) optimal hard threshold for known/unknown noise.

    For square matrices (n = m), threshold is omega(beta=1) * median(sigma)
    where omega(1) = 4/sqrt(3) approx 2.858.
    Counts singular values exceeding the threshold.
    """
    t0 = time.time()
    s = sla.svdvals(S)
    n = S.shape[0]
    omega = 4.0 / np.sqrt(3.0)  # for square (beta=1) with unknown noise
    thresh = omega * np.median(s)
    k_hat = int(np.sum(s > thresh))
    return dict(name="donoho_gavish", k_hat=k_hat, runtime_s=time.time() - t0,
                threshold=float(thresh))


def bench_screenot(S, k_max=None, **_):
    """ScreeNOT (Donoho-Gavish-Romanov 2023). Pip-installed package."""
    t0 = time.time()
    try:
        import screenot
    except ImportError:
        return dict(name="screenot", k_hat=-1, runtime_s=0.0,
                    error="ScreeNOT not installed")
    n = S.shape[0]
    if k_max is None:
        k_max = min(n // 4, 100)
    try:
        # adaptiveHardThresholding(Y, k, strategy='i') returns (Xest, Topt, r).
        _, _, r_hat = screenot.adaptiveHardThresholding(S, k_max)
        k_hat = int(r_hat)
    except Exception as e:
        return dict(name="screenot", k_hat=-1, runtime_s=time.time() - t0,
                    error=str(e)[:200])
    return dict(name="screenot", k_hat=k_hat, runtime_s=time.time() - t0,
                k_max_param=int(k_max))


def bench_horn_pa(S, n_perm=10, alpha=0.05, kmax=None, seed=0, **_):
    """Horn's parallel analysis (1965). Compare each observed eigenvalue to
    the upper alpha-quantile from random Gaussian symmetric matrices scaled
    to match S's noise floor.

    Noise scale is estimated as MAD of off-diagonal entries (robust to signal
    contributions which are O(lambda/sqrt(n)) per entry). Then null Z has
    entries N(0, sigma_z^2) symmetric; its Wigner-semicircle bulk has edge
    ~2*sigma_z*sqrt(n).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    eigs = _topk_eigvals(S, kmax or min(n, 100))

    # Noise scale: MAD of off-diagonal entries (sigma estimate).
    off = S[np.triu_indices(n, k=1)]
    sigma_z = float(1.4826 * np.median(np.abs(off - np.median(off))))
    if sigma_z <= 0:
        sigma_z = float(np.std(off)) + 1e-12

    null_eigs = np.zeros((n_perm, len(eigs)))
    for b in range(n_perm):
        Z = rng.standard_normal((n, n))
        Z = (Z + Z.T) / np.sqrt(2)        # symmetric entries N(0, 1)
        Z *= sigma_z                        # match S's noise scale
        e = np.linalg.eigvalsh(Z)[::-1]
        null_eigs[b] = e[:len(eigs)]
    cutoff = np.quantile(null_eigs, 1 - alpha, axis=0)

    # k_hat = first rank where observed drops below cutoff (run of trues).
    above = eigs > cutoff
    k_hat = 0
    for r in range(len(eigs)):
        if above[r]:
            k_hat = r + 1
        else:
            break
    return dict(name="horn_pa", k_hat=int(k_hat), runtime_s=time.time() - t0,
                sigma_z=sigma_z, cutoff=cutoff.tolist())


# ============================================================
# Intrinsic-dimension estimators (apply to top-k embedded points)
# ============================================================

def _embed_from_kernel(S, top_k=30):
    """Recover Euclidean coordinates X from a positive-semi-definite-ish S
    via top-k eigenvectors weighted by sqrt(eigvals). Used as input to
    intrinsic-dim estimators (TwoNN, MLE, DANCo)."""
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1][:top_k]
    return eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0.0) + 1e-9)[None, :]


def _intrinsic_dim_runner(name, est_factory, top_k=30):
    """Factory for an intrinsic-dim benchmark."""
    def runner(S, **_):
        t0 = time.time()
        try:
            import skdim
        except ImportError:
            return dict(name=name, k_hat=-1, runtime_s=0.0,
                         error="scikit-dimension not installed")
        try:
            X = _embed_from_kernel(S, top_k=top_k)
            est = est_factory()
            d = float(est.fit_transform(X))
            k_hat = max(1, int(round(d)))
        except Exception as e:
            return dict(name=name, k_hat=-1, runtime_s=time.time() - t0,
                         error=str(e)[:200])
        return dict(name=name, k_hat=k_hat, runtime_s=time.time() - t0,
                     d_continuous=d, top_k_embed=int(top_k))
    return runner


def _make_twonn():
    import skdim
    return skdim.id.TwoNN()

def _make_mle():
    import skdim
    return skdim.id.MLE()

def _make_danco():
    import skdim
    return skdim.id.DANCo()


bench_twonn = _intrinsic_dim_runner("twonn", _make_twonn, top_k=30)
bench_mle   = _intrinsic_dim_runner("mle",   _make_mle,   top_k=30)
bench_danco = _intrinsic_dim_runner("danco", _make_danco, top_k=30)


# ============================================================
# Bootstrap rank stability (Hennig 2007 / Fischer-Buhmann 2003)
# ============================================================

def bench_bootstrap_stability(S, B=5, k_max=None, subsample_frac=0.7,
                                stability_threshold=0.7, seed=0, **_):
    """Bootstrap subspace stability — pick the largest rank r for which the
    bootstrap-resampled top-r eigenspace remains stable.

    For each bootstrap b (B=10): subsample subsample_frac of indices with
    replacement to get S_b; compute top-K eigenvectors U_b. Map U_b back to
    the original index set (zero-padded), then compute average pairwise
    canonical-correlation similarity (sum of squared singular values of
    U_b^T U_b' / r) across pairs (b, b'). Pick the largest r where mean
    stability >= stability_threshold.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    if k_max is None:
        k_max = min(n // 4, 100)

    # Generate B bootstrap subsamples
    boot_eigvecs = []
    for b in range(B):
        idx = rng.choice(n, size=int(subsample_frac * n), replace=True)
        idx_uniq = np.unique(idx)
        if len(idx_uniq) < k_max + 5:
            continue
        S_b = S[np.ix_(idx_uniq, idx_uniq)]
        S_b = (S_b + S_b.T) / 2
        try:
            ev_b, U_b = np.linalg.eigh(S_b)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(ev_b)[::-1][:k_max]
        U_b = U_b[:, order]  # shape (len(idx_uniq), k_max)
        # Map back to full index space
        U_full = np.zeros((n, k_max))
        U_full[idx_uniq, :] = U_b
        boot_eigvecs.append(U_full)

    if len(boot_eigvecs) < 2:
        return dict(name="bootstrap_stability", k_hat=-1,
                     runtime_s=time.time() - t0, error="too few bootstraps")

    # For each candidate r, compute mean pairwise stability
    stability = np.zeros(k_max)
    for r in range(1, k_max + 1):
        sims = []
        for i in range(len(boot_eigvecs)):
            for j in range(i + 1, len(boot_eigvecs)):
                Ui = boot_eigvecs[i][:, :r]
                Uj = boot_eigvecs[j][:, :r]
                # Re-orthonormalize (zero-rows from index restriction)
                try:
                    Qi, _ = np.linalg.qr(Ui)
                    Qj, _ = np.linalg.qr(Uj)
                    M = Qi.T @ Qj
                    sims.append(float(np.linalg.norm(M, "fro") ** 2 / r))
                except np.linalg.LinAlgError:
                    continue
        stability[r - 1] = float(np.mean(sims)) if sims else 0.0

    # Find largest r where stability >= threshold
    above = stability >= stability_threshold
    if not above.any():
        k_hat = 1
    else:
        # k_hat = largest r in the prefix where stability stays above threshold
        # (i.e., last r where stability >= threshold AND all r' <= r are above)
        k_hat = int(np.where(above)[0][-1] + 1)
        # Tighten: largest contiguous prefix above threshold
        below = np.where(~above)[0]
        if below.size > 0:
            k_hat = int(below[0])
            if k_hat == 0:
                k_hat = 1

    return dict(name="bootstrap_stability", k_hat=k_hat,
                 runtime_s=time.time() - t0,
                 stability_curve=stability.tolist(), n_boot=len(boot_eigvecs))


# ============================================================
# Nakajima EVBMF (variational Bayes matrix factorization, 2013)
# ============================================================
def bench_evbmf(S, k_max=None, **_):
    """Empirical-Bayes VBMF (Nakajima et al 2013) — closed-form analytical
    rank selection. Returns the number of non-zero singular values in the
    rank-truncated solution.
    """
    t0 = time.time()
    try:
        from _vbmf_third_party import EVBMF
    except ImportError:
        return dict(name="evbmf", k_hat=-1, runtime_s=0.0,
                     error="VBMF module not found")
    try:
        _, S_truncated, _, _ = EVBMF(S, sigma2=None, H=k_max)
        # S_truncated is diagonal of nonzero singular values
        if S_truncated.ndim == 2:
            nonzero = np.diag(S_truncated)
        else:
            nonzero = S_truncated
        k_hat = int(np.sum(nonzero > 1e-10))
    except Exception as e:
        return dict(name="evbmf", k_hat=-1, runtime_s=time.time() - t0,
                     error=str(e)[:200])
    return dict(name="evbmf", k_hat=k_hat, runtime_s=time.time() - t0)


# ============================================================
# Owen-Perry Bi-Cross-Validation (2009) — symmetric variant
# ============================================================
def bench_owen_perry_bicv(S, k_max=None, n_folds_row=2, n_folds_col=2,
                            seed=0, **_):
    """Bi-cross-validation (Owen & Perry 2009, AOAS) for symmetric similarity.

    Partition rows into r row-folds and columns into c col-folds. For each
    (i, j) fold pair, hold out S[I, J], fit rank-r SVD on S[I^c, J^c], and
    predict held-out via the Owen-Perry formula:
        S_hat[I, J] = S[I, J^c] V_r Sigma_r^{-1} U_r^T S[I^c, J].
    Score MSE over all (i, j) folds; pick rank minimizing total CV MSE.

    Returns the rank that minimizes total bi-CV MSE.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    if k_max is None:
        k_max = min(n // 4, 50)

    row_perm = rng.permutation(n)
    col_perm = rng.permutation(n)
    row_folds = np.array_split(row_perm, n_folds_row)
    col_folds = np.array_split(col_perm, n_folds_col)
    K = max(1, k_max)
    cv_mse = np.zeros(K)

    for I in row_folds:
        Ic = np.setdiff1d(np.arange(n), I)
        for J in col_folds:
            Jc = np.setdiff1d(np.arange(n), J)
            S_train = S[np.ix_(Ic, Jc)]
            S_held = S[np.ix_(I, J)]
            S_top = S[np.ix_(I, Jc)]      # left predictor
            S_left = S[np.ix_(Ic, J)]     # right predictor
            try:
                U, sv, Vt = sla.svd(S_train, full_matrices=False)
            except Exception:
                continue
            for r in range(1, min(K, len(sv)) + 1):
                Ur = U[:, :r]; svr = sv[:r]; Vtr = Vt[:r, :]
                # Owen-Perry prediction: S_top @ V_r @ diag(1/sv_r) @ U_r^T @ S_left
                pred = S_top @ Vtr.T @ np.diag(1.0 / np.maximum(svr, 1e-12)) @ Ur.T @ S_left
                mse = float(np.mean((S_held - pred) ** 2))
                cv_mse[r - 1] += mse

    if not np.any(cv_mse > 0):
        return dict(name="owen_perry_bicv", k_hat=-1,
                     runtime_s=time.time() - t0, error="all folds failed")
    k_hat = int(np.argmin(cv_mse) + 1)
    return dict(name="owen_perry_bicv", k_hat=k_hat,
                 runtime_s=time.time() - t0,
                 cv_curve=cv_mse.tolist())


# ============================================================
# Recipe K's spectral k_cut (delegated; kept after intrinsic-dim block)
# ============================================================

def bench_recipe_K_kcut(S, spec=None, **_):
    """Recipe K's spectral k_cut (from F-stat changepoint).

    If `spec` is provided (cached spectral pass), reuses it; else runs a fresh
    spectral pass.
    """
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    from _common import spectral_pass

    t0 = time.time()
    if spec is None:
        spec = spectral_pass(S, B=20, smooth_window=10)
    k_cut = int(spec["k_smooth"])
    return dict(name="recipe_K_kcut", k_hat=k_cut, runtime_s=time.time() - t0,
                k_cliff=spec.get("k_cliff"), tk=None)


# ============================================================
# Registry
# ============================================================

BENCHMARKS = [
    ("kneedle",        bench_kneedle),
    ("eigengap",       bench_eigengap),
    ("donoho_gavish",  bench_donoho_gavish),
    ("screenot",       bench_screenot),
    ("horn_pa",        bench_horn_pa),
    ("recipe_K_kcut",  bench_recipe_K_kcut),
    ("twonn",          bench_twonn),
    ("mle",            bench_mle),
    ("danco",          bench_danco),
    ("evbmf",          bench_evbmf),
    ("owen_perry_bicv", bench_owen_perry_bicv),
    # Methods kept in module but excluded from default suite:
    #   bootstrap_stab (too slow at high n; broken decision rule).
    # SOTA peers NOT included (and why):
    #   Wang-Stephens flashier (2021) — R-only via rpy2; CRAN compile failed
    #     (mixsqp/ashr deps need build env we lack). Document omission.
    #   Onatski ED (2010), Bai-Ng IC (2002), Ahn-Horenstein ER (2013)
    #     — all assume MP-bulk + iid factor-model noise. Same fundamental
    #     assumption as ScreeNOT (already included as the most recent and
    #     most-cited representative of that family). Including them would
    #     replicate ScreeNOT's failure pattern without adding methodological
    #     diversity. See EXPERIMENT_A_FULL_REPORT.md §6 for the rationale.
]
