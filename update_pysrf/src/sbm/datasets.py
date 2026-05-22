"""Synthetic similarity-matrix builders for the Recipe-K + SBM experiment
suite (plan_SBM_clustering.md §9 and §16).

Every builder returns a dictionary conforming to §13.1:

    {
        "S":               (n, n) ndarray similarity / adjacency,
        "K_true":          int or None (None for negative-control / overlapping),
        "labels_true":     (n,) ndarray of integer labels, or None,
        "matrix_for_recipe_k": (n, n) PSD-or-symmetric matrix used by the
                                spectral pass (often == S, but may be a PSD
                                derivative for raw adjacency),
        "matrix_for_sbm":  (n, n) matrix used by the SBM fitter (often == S),
        "loss_family":     str — one of the LOSS_FAMILIES,
        "metadata":        dict.
    }

For real-data inputs the existing Recipe-K loaders in ``_common.load_v3_dataset``
remain the canonical source — they are not duplicated here.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def _symmetrize_offdiag(M):
    out = 0.5 * (M + M.T)
    return out


def _rbf_kernel(X, bw):
    D2 = cdist(X, X, metric="sqeuclidean")
    return np.exp(-D2 / (2.0 * bw ** 2))


def _diffusion_kernel(A, alpha=0.5):
    """Return a PSD similarity from a symmetric (possibly non-PSD) adjacency:

        K = (I + alpha * A_sym)^{T} (I + alpha * A_sym)  (always PSD).

    Used as ``matrix_for_recipe_k`` when the SBM is fit on raw binary edges and
    the Rayleigh-trace assumption in Recipe K calls for a PSD matrix (plan §1.4).
    """
    n = A.shape[0]
    A_sym = 0.5 * (A + A.T)
    M = np.eye(n) + alpha * A_sym
    K = M.T @ M
    return _symmetrize_offdiag(K)


def _equal_sizes(n, K):
    sizes = [n // K + (1 if i < n % K else 0) for i in range(K)]
    return sizes


def _labels_from_sizes(sizes):
    return np.concatenate([np.full(s, k, dtype=int) for k, s in enumerate(sizes)])


# ----------------------------------------------------------------------
# §9.1 Clean balanced weighted SBM
# ----------------------------------------------------------------------

def clean_weighted_sbm(n=300, K=5, mu_in=1.0, mu_out=0.0, sigma=0.3, seed=0):
    """S_ij = B*_{z_i, z_j} + sigma * xi_ij, xi_ij ~ N(0, 1).

    SNR := (mu_in - mu_out) / sigma.
    """
    rng = np.random.default_rng(int(seed))
    sizes = _equal_sizes(n, K)
    z = _labels_from_sizes(sizes)
    Bstar = np.full((K, K), float(mu_out))
    np.fill_diagonal(Bstar, float(mu_in))
    mean = Bstar[z][:, z]
    iu = np.triu_indices(n, k=1)
    noise = np.zeros((n, n))
    noise[iu] = sigma * rng.standard_normal(len(iu[0]))
    noise = noise + noise.T
    S = mean + noise
    # Diagonal = same-block expected value (mu_in). Recipe-K's Rayleigh-trace
    # denominator tr(S) needs a non-degenerate diagonal; the SBM fitter never
    # uses diag(S) so this only affects the spectral pass.
    np.fill_diagonal(S, float(mu_in))
    S = _symmetrize_offdiag(S)
    snr = (mu_in - mu_out) / max(sigma, 1e-12)
    meta = dict(model="clean_weighted_sbm", K=K, sigma=sigma, mu_in=mu_in,
                 mu_out=mu_out, SNR=float(snr), seed=int(seed))
    return dict(S=S, K_true=int(K), labels_true=z,
                matrix_for_recipe_k=S, matrix_for_sbm=S,
                loss_family="gaussian_mse", metadata=meta)


# ----------------------------------------------------------------------
# §9.2 Binary assortative SBM
# ----------------------------------------------------------------------

def binary_assortative_sbm(n=300, K=5, p_in=0.30, p_out=0.05, seed=0):
    """A_ij ~ Bernoulli(p_in if z_i==z_j else p_out)."""
    rng = np.random.default_rng(int(seed))
    sizes = _equal_sizes(n, K)
    z = _labels_from_sizes(sizes)
    Pstar = np.full((K, K), float(p_out))
    np.fill_diagonal(Pstar, float(p_in))
    P = Pstar[z][:, z]
    iu = np.triu_indices(n, k=1)
    A = np.zeros((n, n))
    A[iu] = (rng.random(len(iu[0])) < P[iu]).astype(float)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    threshold_excess = ((p_in - p_out) * n) ** 2 - 2.0 * (p_in + p_out) * n
    meta = dict(model="binary_assortative_sbm", K=K, p_in=p_in, p_out=p_out,
                 threshold_excess=float(threshold_excess), seed=int(seed))
    # SBM is fit directly on the binary edges (Bernoulli NLL); spectral pass
    # uses a PSD diffusion kernel for the Rayleigh-trace step (§1.4).
    return dict(S=A, K_true=int(K), labels_true=z,
                matrix_for_recipe_k=_diffusion_kernel(A),
                matrix_for_sbm=A, loss_family="bernoulli", metadata=meta)


# ----------------------------------------------------------------------
# §9.3 Degree-corrected SBM with hubs
# ----------------------------------------------------------------------

def dcsbm_with_hubs(n=300, K=5, p_in=0.30, p_out=0.05, tail_strength=2.0,
                    seed=0):
    """E[A_ij | z, theta] = theta_i theta_j P*_{z_i, z_j}, then Bernoulli.

    theta_i drawn from a Pareto-like distribution, normalised so the average
    propensity within each block is 1.
    """
    rng = np.random.default_rng(int(seed))
    sizes = _equal_sizes(n, K)
    z = _labels_from_sizes(sizes)
    theta = 1.0 + rng.pareto(tail_strength, size=n)
    # block-wise normalisation
    for c in range(K):
        idx = np.where(z == c)[0]
        theta[idx] /= max(theta[idx].mean(), 1e-12)
    Pstar = np.full((K, K), float(p_out))
    np.fill_diagonal(Pstar, float(p_in))
    mean = (theta[:, None] * Pstar[z][:, z]) * theta[None, :]
    mean = np.clip(mean, 0.0, 1.0)
    iu = np.triu_indices(n, k=1)
    A = np.zeros((n, n))
    A[iu] = (rng.random(len(iu[0])) < mean[iu]).astype(float)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    meta = dict(model="dcsbm_with_hubs", K=K, p_in=p_in, p_out=p_out,
                 tail_strength=tail_strength, seed=int(seed))
    return dict(S=A, K_true=int(K), labels_true=z,
                matrix_for_recipe_k=_diffusion_kernel(A),
                matrix_for_sbm=A, loss_family="bernoulli", metadata=meta)


# ----------------------------------------------------------------------
# §9.4 RBF Gaussian blocks
# ----------------------------------------------------------------------

def rbf_gaussian_blocks(n=300, K=5, sep=4.0, sigma=0.5, dim=2, bw_scale=1.0,
                        seed=0):
    """K Gaussian blobs in R^dim + RBF kernel at median-distance bandwidth."""
    rng = np.random.default_rng(int(seed))
    sizes = _equal_sizes(n, K)
    z = _labels_from_sizes(sizes)
    centers = rng.standard_normal((K, dim))
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True),
                                   1e-12) * (sep / 2.0)
    X = np.vstack([centers[k] + sigma * rng.standard_normal((sz, dim))
                   for k, sz in enumerate(sizes)])
    D = cdist(X, X)
    bw = float(np.median(D[D > 0])) * float(bw_scale)
    S = np.exp(-(D ** 2) / (2.0 * bw ** 2))
    # RBF self-similarity = 1; keep it (Recipe-K Rayleigh trace needs tr(S) > 0).
    np.fill_diagonal(S, 1.0)
    S = _symmetrize_offdiag(S)
    meta = dict(model="rbf_gaussian_blocks", K=K, sep=sep, sigma=sigma,
                 dim=dim, bw_scale=bw_scale, bw=bw, seed=int(seed))
    return dict(S=S, K_true=int(K), labels_true=z,
                matrix_for_recipe_k=S, matrix_for_sbm=S,
                loss_family="gaussian_mse", metadata=meta)


# ----------------------------------------------------------------------
# §9.5 Hierarchical blocks
# ----------------------------------------------------------------------

def hierarchical_blocks(n=400, K_coarse=4, K_fine=16, mu0=0.0, mu1=0.5,
                        mu2=0.8, sigma=0.4, seed=0):
    """B*_ij = mu0 + mu1 * 1{g_i = g_j} + mu2 * 1{h_i = h_j}.

    Coarse labels are g ∈ {0, .., K_coarse-1}; fine labels h ∈ {0, .., K_fine-1};
    each coarse block contains K_fine / K_coarse fine blocks.
    """
    if K_fine % K_coarse != 0:
        raise ValueError("K_fine must be a multiple of K_coarse")
    rng = np.random.default_rng(int(seed))
    sizes_fine = _equal_sizes(n, K_fine)
    h = _labels_from_sizes(sizes_fine)
    g = h // (K_fine // K_coarse)
    same_g = (g[:, None] == g[None, :])
    same_h = (h[:, None] == h[None, :])
    mean = mu0 + mu1 * same_g + mu2 * same_h
    iu = np.triu_indices(n, k=1)
    noise = np.zeros((n, n))
    noise[iu] = sigma * rng.standard_normal(len(iu[0]))
    noise = noise + noise.T
    S = mean + noise
    # Diagonal = within-fine-block expected affinity = mu0 + mu1 + mu2.
    np.fill_diagonal(S, float(mu0 + mu1 + mu2))
    S = _symmetrize_offdiag(S)
    meta = dict(model="hierarchical_blocks", K_coarse=K_coarse, K_fine=K_fine,
                 mu0=mu0, mu1=mu1, mu2=mu2, sigma=sigma, seed=int(seed))
    return dict(S=S, K_true=int(K_fine), labels_true=h,
                matrix_for_recipe_k=S, matrix_for_sbm=S,
                loss_family="gaussian_mse", metadata=meta,
                K_alt=int(K_coarse), labels_alt=g)


# ----------------------------------------------------------------------
# §9.8 Smooth manifold negative controls
# ----------------------------------------------------------------------

def smooth_manifold(n=300, kind="circle", bw_scale=0.5, seed=0):
    """Single connected manifold + RBF kernel. No discrete cluster structure."""
    rng = np.random.default_rng(int(seed))
    if kind == "circle":
        t = np.sort(rng.uniform(0, 2 * np.pi, size=n))
        X = np.c_[np.cos(t), np.sin(t)]
    elif kind == "swiss_roll":
        t = 1.5 * np.pi * (1 + 2.0 * rng.random(n))
        X = np.c_[t * np.cos(t), 10.0 * rng.random(n), t * np.sin(t)]
    elif kind == "trajectory":
        t = np.linspace(0, 1, n) + 0.001 * rng.standard_normal(n)
        X = np.c_[t, np.sin(2 * np.pi * t)]
    elif kind == "torus":
        u = 2 * np.pi * rng.random(n)
        v = 2 * np.pi * rng.random(n)
        R, r = 2.0, 0.7
        X = np.c_[(R + r * np.cos(v)) * np.cos(u),
                  (R + r * np.cos(v)) * np.sin(u),
                  r * np.sin(v)]
    else:
        raise ValueError(f"unknown manifold kind {kind!r}")
    D = cdist(X, X)
    bw = float(np.median(D[D > 0])) * float(bw_scale)
    S = np.exp(-(D ** 2) / (2.0 * bw ** 2))
    np.fill_diagonal(S, 1.0)
    S = _symmetrize_offdiag(S)
    meta = dict(model="smooth_manifold", kind=kind, bw_scale=bw_scale,
                 bw=bw, seed=int(seed))
    # K_true=None: expected output is "no discrete cluster count" (§9.8).
    return dict(S=S, K_true=None, labels_true=None,
                matrix_for_recipe_k=S, matrix_for_sbm=S,
                loss_family="gaussian_mse", metadata=meta)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

DATASET_BUILDERS = {
    "clean_weighted_sbm": clean_weighted_sbm,
    "binary_assortative_sbm": binary_assortative_sbm,
    "dcsbm_with_hubs": dcsbm_with_hubs,
    "rbf_gaussian_blocks": rbf_gaussian_blocks,
    "hierarchical_blocks": hierarchical_blocks,
    "smooth_manifold": smooth_manifold,
}


def build_dataset(name, **kwargs):
    if name not in DATASET_BUILDERS:
        raise ValueError(f"unknown dataset {name!r}; available: "
                         f"{sorted(DATASET_BUILDERS)}")
    return DATASET_BUILDERS[name](**kwargs)
