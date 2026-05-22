"""
Synthetic generators for Experiment A (adversarial benchmark stress-tests) and
Experiment B (per-representation generative models with well-defined true rank).

EXPERIMENT A — adversarial datasets designed to break specific benchmarks.
  A1 power_law:        slow eigenvalue decay (no scree elbow, no eigengap)
  A2 t3_noise:         rank-r signal + Student-t(df=3) noise (breaks PA Gaussian assumption)
  A3 anisotropic:      rank-r + AR(1) row-correlated noise (breaks Donoho-Gavish/ScreeNOT MP-bulk)
  A4 coherent:         rank-r with localized (high mu) eigenvectors (stress matrix completion)
  A5 bbp_twin:         two spikes straddling the BBP threshold (one inside, one outside)
  A6 mixed_scale:      large and small spikes mixed (breaks ratio/elbow methods)
  A7 unequal_blocks:   block-diagonal with unequal sizes (breaks eigengap/modularity)
  A8 shifted_offset:   rank-1 constant offset plus low-rank signal (breaks non-centered methods)

EXPERIMENT B — generative models tied to a specific representation method's natural form.
  ppca_synth:          S = Lambda Lambda^T + sigma^2 I  (Tipping & Bishop 1999)
  softimp_synth:       S = U_r Sigma_r U_r^T + sigma G  (low-rank + Gaussian noise)
  nmf_synth:           S = W W^T + small nonneg noise   (W >= 0)
  rpca_synth:          S = L_r + S_sparse + sigma G    (low-rank + sparse outliers)
  manifold_swissroll:  intrinsic-dim 2 manifold (for intrinsic-dim track)
  manifold_torus:      intrinsic-dim 2
  manifold_helix:      intrinsic-dim 1

All generators return (S, true_rank). For Experiment B, "true_rank" is the
parameter of the generative process that the corresponding method should
recover under no-noise conditions.
"""
import numpy as np
from scipy.spatial.distance import cdist


# ============================================================
# Experiment A — adversarial datasets
# ============================================================

def make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=0):
    """A1: rank-k_signal signal with eigenvalues lambda_r = snr * r^{-alpha}
    (slow decay; alpha=1 gives 5, 2.5, 1.67, 1.25, 1.0, ...). Past the BBP
    threshold the smallest spikes become indistinguishable from noise.

    "true_rank" = k_signal (structural). Note that the *detectable* rank is
    smaller because lambda_r drops below the BBP threshold; we report the
    structural rank so methods that detect only the top few are penalized.
    """
    rng = np.random.default_rng(seed)
    eigvals = snr * (np.arange(1, k_signal + 1)) ** (-alpha)
    U = rng.standard_normal((n, k_signal))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, k_signal


def make_t3_noise(n=300, r=5, snr=4.0, df=3, seed=0):
    """A2: rank-r signal plus heavy-tailed Student-t(df=3) symmetric noise.

    Breaks Horn's PA (assumes Gaussian) and tests robustness of thresholding.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    # Student-t symmetric noise via N(0,1) / sqrt(chi2/df)
    Z = rng.standard_normal((n, n))
    chi2 = rng.chisquare(df, size=(n, n))
    T = Z / np.sqrt(chi2 / df)
    T = (T + T.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_anisotropic(n=300, r=5, snr=4.0, rho=0.7, seed=0):
    """A3: rank-r signal plus AR(1) row-correlated Gaussian noise.

    Noise covariance Sigma_ij = rho^|i-j|; Marchenko-Pastur bulk is non-MP.
    Breaks Donoho-Gavish 4/sqrt(3) (assumes MP bulk).
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    # Sample AR(1) noise: W has rows correlated row-wise.
    Z = rng.standard_normal((n, n))
    # AR(1) along columns
    c = np.zeros((n, n))
    c[:, 0] = Z[:, 0]
    for j in range(1, n):
        c[:, j] = rho * c[:, j - 1] + np.sqrt(1 - rho ** 2) * Z[:, j]
    T = (c + c.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_coherent(n=300, r=5, snr=4.0, frac_active=0.15, seed=0):
    """A4: rank-r signal whose eigenvectors are LOCALIZED on a fraction of
    coordinates — high incoherence parameter mu. Stresses matrix completion.

    Each eigenvector u_k has support on ~frac_active*n indices.
    """
    rng = np.random.default_rng(seed)
    n_active = int(frac_active * n)
    U = np.zeros((n, r))
    for k in range(r):
        idx = rng.choice(n, size=n_active, replace=False)
        v = rng.standard_normal(n_active)
        v /= np.linalg.norm(v)
        U[idx, k] = v
    # Re-orthogonalize roughly (overlapping localized vectors)
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_bbp_twin(n=400, lam_inside=1.6, lam_outside=0.7, seed=0):
    """A5: two rank-1 spikes — one above the BBP threshold (lam_inside > 1)
    and one below (lam_outside < 1). True rank is structurally 2 but only
    1 spike is detectable in the spectrum.

    "True_rank" reported as 2 (the structural rank), so methods that detect
    only 1 are penalized.
    """
    rng = np.random.default_rng(seed)
    u1 = rng.standard_normal(n); u1 /= np.linalg.norm(u1)
    u2 = rng.standard_normal(n); u2 -= np.dot(u1, u2) * u1; u2 /= np.linalg.norm(u2)
    sig = lam_inside * np.outer(u1, u1) + lam_outside * np.outer(u2, u2)
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, 2


def make_mixed_scale(n=300, scales=(15.0, 10.0, 1.0, 0.5), seed=0):
    """A6: spikes with disparate strengths. Methods using ratio gaps may
    pick k=2 (large vs small jump), methods using absolute thresholds may
    pick k=4. True rank = len(scales) = 4.
    """
    rng = np.random.default_rng(seed)
    r = len(scales)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(scales) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_unequal_blocks(n=400, sizes=(0.5, 0.25, 0.125, 0.125),
                          within=0.9, between=0.1, noise=0.05, seed=0):
    """A7: block-diagonal with unequal block sizes. K=4 communities of sizes
    n*sizes. Eigengap may pick wrong k; modularity may underestimate.
    """
    rng = np.random.default_rng(seed)
    block_sizes = [int(n * s) for s in sizes]
    block_sizes[-1] = n - sum(block_sizes[:-1])  # fill to exactly n
    K = len(block_sizes)
    labels = np.concatenate([np.full(b, k) for k, b in enumerate(block_sizes)])
    perm = rng.permutation(n)
    labels = labels[perm]
    S = np.full((n, n), between)
    for k in range(K):
        idx = np.where(labels == k)[0]
        S[np.ix_(idx, idx)] = within
    np.fill_diagonal(S, 1.0)
    Z = rng.standard_normal((n, n))
    T = noise * (Z + Z.T) / np.sqrt(2)
    return S + T, K


def make_linear_decay(n=400, r=20, lam_top=6.0, lam_bot=1.5, seed=0):
    """E1: rank-r signal with LINEARLY-decaying eigenvalues from `lam_top`
    down to `lam_bot`. Largest within-signal gap is the *very first* one,
    so eigengap will pick k_hat=1. Smoothly decaying signal exposes egap's
    "max single gap" assumption.
    """
    rng = np.random.default_rng(seed)
    eigvals = np.linspace(lam_top, lam_bot, r)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_two_tier_spikes(n=400, r_big=5, r_small=20, lam_big=8.0, lam_small=1.5, seed=0):
    """E2: TWO TIERS of spikes — `r_big` large eigenvalues then `r_small`
    smaller ones, then bulk. Total true rank = r_big + r_small.

    Eigengap picks the largest gap, which is between the big-tier and the
    small-tier (k_hat=r_big), missing the smaller spikes entirely.
    """
    rng = np.random.default_rng(seed)
    r = r_big + r_small
    eigvals = np.concatenate([np.full(r_big, lam_big), np.full(r_small, lam_small)])
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_smooth_powerlaw_hi(n=500, k_signal=30, alpha=0.5, snr=8.0, seed=0):
    """E3: rank-30 slow power-law decay (alpha=0.5). No clear gap; egap
    arbitrary. Large true_rank stresses methods that prefer small k.
    """
    rng = np.random.default_rng(seed)
    eigvals = snr * (np.arange(1, k_signal + 1)) ** (-alpha)
    U = rng.standard_normal((n, k_signal))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, k_signal


def make_nested_blocks(n=400, super_K=4, sub_K=4, within_super=0.4,
                         within_sub=0.85, between=0.05, noise=0.05, seed=0):
    """E4: 16 nested communities (super_K * sub_K). Hierarchical block model.

    Eigengap will likely pick super_K=4 (the largest gap, at the
    super-community boundary), missing the sub-communities. True total rank = K.
    """
    rng = np.random.default_rng(seed)
    K_total = super_K * sub_K
    sub_block = n // K_total
    sizes = [sub_block] * K_total
    sizes[-1] += n - sum(sizes)
    labels_sub = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    labels_super = labels_sub // sub_K
    perm = rng.permutation(n)
    labels_sub = labels_sub[perm]
    labels_super = labels_super[perm]

    S = np.full((n, n), between)
    for k in range(super_K):
        idx = np.where(labels_super == k)[0]
        S[np.ix_(idx, idx)] = within_super
    for k in range(K_total):
        idx = np.where(labels_sub == k)[0]
        S[np.ix_(idx, idx)] = within_sub
    np.fill_diagonal(S, 1.0)
    Z = rng.standard_normal((n, n))
    T = noise * (Z + Z.T) / np.sqrt(2)
    return S + T, K_total


# ============================================================
# Recipe-K-adversarial synthetics (R1-R4)
# ============================================================

def make_continuous_spectrum(n=400, k_signal=30, lam_top=8.0, decay=0.08, seed=0):
    """R1: continuous spectrum — eigenvalues decay smoothly from spike to bulk
    with NO clear changepoint between signal and noise. Adversarial against
    Recipe K's F-stat changepoint detector (assumes 2-regime kappa profile).

    lambda_r = lam_top * exp(-decay * (r-1)), for r=1..k_signal.
    With decay=0.08, lam decays from 8 to 8*exp(-0.08*30) ~ 0.73 (just at bulk).

    "true_rank" is the BBP-threshold rank: smallest r where lambda_r drops
    below 1.0 (the bulk edge for symmetric Wigner noise on this scale).
    """
    rng = np.random.default_rng(seed)
    eigvals = lam_top * np.exp(-decay * np.arange(k_signal))
    U = rng.standard_normal((n, k_signal))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    # BBP threshold: lambda_BBP ~ 1 for symmetric noise scale 1/sqrt(2n).
    k_BBP = int(np.sum(eigvals > 1.0))
    return sig + T, k_BBP


def make_heteroscedastic(n=300, r=5, snr=4.0, sigma_lo=0.3, sigma_hi=1.5, seed=0):
    """R2: rank-r signal plus HETEROSCEDASTIC Gaussian noise — row i has
    variance sigma_i^2 = (sigma_lo + (sigma_hi - sigma_lo) * i/n)^2.
    Violates the homogeneous-noise assumption shared by Recipe K, ScreeNOT,
    and Donoho-Gavish.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    sigmas = np.linspace(sigma_lo, sigma_hi, n)
    Z = rng.standard_normal((n, n)) * sigmas[:, None]
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_extreme_coherence(n=300, r=5, snr=5.0, n_active=10, seed=0):
    """R3: rank-r signal where each eigenvector is supported on exactly
    `n_active` indices — extreme localization (mu = n / n_active >> 1).
    Stresses ALL methods that assume eigenvector delocalization.
    """
    rng = np.random.default_rng(seed)
    U = np.zeros((n, r))
    for k in range(r):
        idx = rng.choice(n, size=n_active, replace=False)
        v = rng.standard_normal(n_active)
        v /= np.linalg.norm(v)
        U[idx, k] = v
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_mixed_tail_noise(n=300, r=5, snr=4.0, sparse_frac=0.02, sparse_amp=8.0, seed=0):
    """R4: rank-r signal + Gaussian noise + sparse very-large outliers
    (Bernoulli sparse + Gaussian amplitude). Attacks methods that assume
    noise is light-tailed; tests robustness gradient (e.g., Gaussian → t_3 → mixed).
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    # Add sparse large outliers
    mask = rng.random((n, n)) < sparse_frac
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    Sp = np.zeros((n, n))
    Sp[mask] = rng.standard_normal(int(mask.sum())) * sparse_amp
    Sp = (Sp + Sp.T) / 2
    return sig + T + Sp, r


# ============================================================
# R4 mixed-tail variants — different outlier regimes
# ============================================================

def make_mixed_tail_heavy(n=300, r=5, snr=4.0, sparse_frac=0.02, sparse_amp=20.0, seed=0):
    """R4a: stronger outlier amplitude (sparse_amp=20 vs R4's 8)."""
    return make_mixed_tail_noise(n=n, r=r, snr=snr, sparse_frac=sparse_frac,
                                  sparse_amp=sparse_amp, seed=seed)


def make_mixed_tail_dense(n=300, r=5, snr=4.0, sparse_frac=0.05, sparse_amp=8.0, seed=0):
    """R4b: denser outliers (sparse_frac=0.05 vs R4's 0.02)."""
    return make_mixed_tail_noise(n=n, r=r, snr=snr, sparse_frac=sparse_frac,
                                  sparse_amp=sparse_amp, seed=seed)


def make_cauchy_noise(n=300, r=5, snr=4.0, scale=0.3, seed=0):
    """R4c: rank-r signal + standard Cauchy noise (Student-t df=1, infinite mean
    and variance). Most extreme heavy-tail noise. Standard Wigner / MP theory
    breaks completely.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_cauchy(size=(n, n)) * scale
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_kitchen_sink(n=300, r=5, snr=4.0, sparse_frac=0.03, sparse_amp=15.0,
                      sigma_lo=0.5, sigma_hi=1.5, seed=0):
    """R4d: combined attack — heteroscedastic noise + sparse heavy outliers.
    Tests robustness when multiple assumptions break simultaneously.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    sigmas = np.linspace(sigma_lo, sigma_hi, n)
    Z = rng.standard_normal((n, n)) * sigmas[:, None]
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    mask = rng.random((n, n)) < sparse_frac
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    Sp = np.zeros((n, n))
    Sp[mask] = rng.standard_normal(int(mask.sum())) * sparse_amp
    Sp = (Sp + Sp.T) / 2
    return sig + T + Sp, r


# ============================================================
# High-dimensional datasets — n in 800..2000
# ============================================================

def make_kernel_block_big(n=1000, K=15, bw=1.0, seed=0):
    """HD1: K-block structure with RBF kernel similarity at n=1000.
    Real-data scale; tests whether thresholding methods break on RBF kernels.
    """
    rng = np.random.default_rng(seed)
    block_size = n // K
    sizes = [block_size] * K
    sizes[-1] += n - sum(sizes)
    centers = rng.standard_normal((K, 10)) * 5.0
    X = np.zeros((n, 10))
    pos = 0
    for k in range(K):
        sz = sizes[k]
        X[pos:pos+sz] = centers[k] + rng.standard_normal((sz, 10)) * 0.5
        pos += sz
    perm = rng.permutation(n)
    X = X[perm]
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-D / (bw ** 2 * np.median(D[D > 0]))), K


def make_continuous_big(n=1000, k_signal=40, lam_top=10.0, decay=0.06, seed=0):
    """HD2: high-dim continuous spectrum (n=1000, k_signal=40).

    "true_rank" defined as smallest r such that lambda_r > 1 (BBP threshold for
    standard symmetric Wigner noise on this scale).
    """
    rng = np.random.default_rng(seed)
    eigvals = lam_top * np.exp(-decay * np.arange(k_signal))
    U = rng.standard_normal((n, k_signal))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    k_BBP = int(np.sum(eigvals > 1.0))
    return sig + T, k_BBP


def make_powerlaw_big(n=1500, k_signal=50, alpha=0.5, snr=10.0, seed=0):
    """HD3: high-dim slow-power-law (n=1500, rank-50, alpha=0.5)."""
    rng = np.random.default_rng(seed)
    eigvals = snr * (np.arange(1, k_signal + 1)) ** (-alpha)
    U = rng.standard_normal((n, k_signal))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(eigvals) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, k_signal


def make_nested_big(n=1000, super_K=5, sub_K=5, within_super=0.4, within_sub=0.85,
                     between=0.05, noise=0.05, seed=0):
    """HD4: 25 nested blocks at n=1000."""
    return make_nested_blocks(n=n, super_K=super_K, sub_K=sub_K,
                               within_super=within_super, within_sub=within_sub,
                               between=between, noise=noise, seed=seed)


def make_shifted_offset(n=300, r=5, offset=2.0, snr=2.0, seed=0):
    """A8: rank-1 constant offset (1 1^T scaled by 'offset') plus rank-r signal.
    Spectrum has 1 huge spike (the constant) plus r smaller ones.

    "True_rank" depends on interpretation: structural = r (the meaningful
    signal), naive = r + 1 (counts the constant). We report r + 1 to penalize
    methods that fail to subtract the constant; doublecentering would give r.
    """
    rng = np.random.default_rng(seed)
    one = np.ones((n, 1))
    offset_mat = offset * (one @ one.T) / n
    U = rng.standard_normal((n, r))
    # Make U orthogonal to ones
    U = U - one @ (one.T @ U) / n
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return offset_mat + sig + T, r + 1


# ============================================================
# Experiment B — per-representation generators
# ============================================================

def make_ppca_synth(n=300, r=5, sigma=0.4, seed=0):
    """PPCA generative model: S = Lambda Lambda^T + sigma^2 * I + small symm noise.

    True dim = r (number of latent factors). Suited to factor analysis / PPCA
    which assumes isotropic Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    Lam = rng.standard_normal((n, r))
    sig = Lam @ Lam.T + sigma ** 2 * np.eye(n)
    # Add small symmetric noise to avoid zero off-diagonal
    Z = rng.standard_normal((n, n)) * 0.05
    T = (Z + Z.T) / 2
    return sig + T, r


def make_softimp_synth(n=300, r=8, snr=3.0, seed=0):
    """Low-rank + Gaussian noise: S = U Sigma U^T + sigma G_sym.
    Suited to Soft-Impute SVD truncation.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 2.0, snr, r)) @ U.T
    Z = rng.standard_normal((n, n))
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_nmf_synth(n=300, r=10, noise=0.05, seed=0):
    """Symmetric NMF generative model: S = W W^T + small nonneg noise, W >= 0.

    True dim = r. Suited to weighted NMF or SymmNMF.
    """
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n, r)))
    sig = W @ W.T
    sig /= sig.max()
    nz = noise * np.abs(rng.standard_normal((n, n)))
    nz = (nz + nz.T) / 2
    return sig + nz, r


def make_rpca_synth(n=300, r=5, sparse_frac=0.05, snr=3.0, sigma=0.05, seed=0):
    """Robust PCA generator: S = L_r + S_sparse + sigma * G_sym.

    L_r is rank-r, S_sparse has sparse_frac symmetric outliers ~ N(0, large).
    True rank of L is r. Suited to Robust PCA decomposition.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    L = U @ np.diag(np.linspace(snr * 2.0, snr, r)) @ U.T
    # Sparse outliers
    Sp = np.zeros((n, n))
    n_out = int(sparse_frac * n * (n - 1) / 2)
    idx_i = rng.integers(0, n, size=n_out)
    idx_j = rng.integers(0, n, size=n_out)
    vals = rng.standard_normal(n_out) * 5.0
    for ii, jj, vv in zip(idx_i, idx_j, vals):
        if ii != jj:
            Sp[ii, jj] += vv; Sp[jj, ii] += vv
    Z = rng.standard_normal((n, n))
    T = sigma * (Z + Z.T) / np.sqrt(2)
    return L + Sp + T, r


# ============================================================
# Manifold generators (for intrinsic-dim track in Experiment B)
# ============================================================

def make_swissroll(n=400, sigma=0.0, bw=None, seed=0):
    """Swiss roll embedded in 3D; intrinsic dim = 2.

    Returns RBF kernel similarity matrix. true_rank reflects intrinsic dim.
    """
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    if sigma > 0:
        X += sigma * rng.standard_normal(X.shape)
    D = cdist(X, X)
    if bw is None:
        bw = float(np.median(D))
    S = np.exp(-(D / bw) ** 2)
    return S, 2


def make_torus(n=400, R=3.0, r=1.0, seed=0, bw=None):
    """Torus surface (2D manifold in 3D)."""
    rng = np.random.default_rng(seed)
    u = 2 * np.pi * rng.random(n)
    v = 2 * np.pi * rng.random(n)
    X = np.column_stack([
        (R + r * np.cos(v)) * np.cos(u),
        (R + r * np.cos(v)) * np.sin(u),
        r * np.sin(v),
    ])
    D = cdist(X, X)
    if bw is None:
        bw = float(np.median(D))
    return np.exp(-(D / bw) ** 2), 2


def make_helix(n=400, k=5, seed=0, bw=None):
    """1D helix in 3D — intrinsic dim = 1."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n)) * 4 * np.pi
    X = np.column_stack([np.cos(t), np.sin(t), t / k])
    D = cdist(X, X)
    if bw is None:
        bw = float(np.median(D))
    return np.exp(-(D / bw) ** 2), 1


# ============================================================
# v3 generators — Laplacian Eigenmaps assumed form
#
# Returns (S, true_r, aux_dict). The aux_dict carries TWO non-dimension
# hyperparameters that match the LE-heat-kernel fitter's inputs:
#
#   t_kernel : kernel bandwidth in S = exp(-D^2 / t_kernel). Sets locality
#              of the similarity matrix; bound to the manifold's spatial
#              scale (default median(D)^2).
#   t_heat   : heat-kernel time in the V-objective weight exp(-lambda_k * t_heat).
#              Independent of t_kernel; controls how aggressively the
#              reconstruction down-weights high-frequency Laplacian modes.
#              Normalised Laplacian eigenvalues are bounded in [0, 2], so
#              t_heat is scale-free (default 1.0).
#
# Both are declared by the generator and passed unchanged into
# fit_laplacian_eigenmaps_heat, realising the fixed-auxiliary v3 design
# (see EXPERIMENT_B_AUDIT.md §1, §5.2.1).
# ============================================================

_T_HEAT_DEFAULT = 1.0


def make_le_swissroll(n=400, sigma=0.0, t_kernel=None, t_heat=_T_HEAT_DEFAULT,
                       seed=0):
    """LE generative form: 2D Swiss-roll manifold in R^3,
    S = exp(-D^2 / t_kernel). Returns (S, 2, dict(t_kernel, t_heat, X, D2))."""
    rng = np.random.default_rng(seed)
    th = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21 * rng.random(n)
    X = np.column_stack([th * np.cos(th), h, th * np.sin(th)])
    if sigma > 0:
        X += sigma * rng.standard_normal(X.shape)
    D = cdist(X, X)
    D2 = D ** 2
    if t_kernel is None:
        t_kernel = float(np.median(D)) ** 2
    S = np.exp(-D2 / t_kernel)
    return S, 2, dict(t_kernel=float(t_kernel), t_heat=float(t_heat),
                      X=X, D2=D2)


def make_le_torus(n=400, R=3.0, r=1.0, t_kernel=None, t_heat=_T_HEAT_DEFAULT,
                   seed=0):
    """LE generative form: 2D torus surface in R^3,
    S = exp(-D^2 / t_kernel). Returns (S, 2, dict(t_kernel, t_heat, X, D2))."""
    rng = np.random.default_rng(seed)
    u = 2 * np.pi * rng.random(n)
    v = 2 * np.pi * rng.random(n)
    X = np.column_stack([
        (R + r * np.cos(v)) * np.cos(u),
        (R + r * np.cos(v)) * np.sin(u),
        r * np.sin(v),
    ])
    D = cdist(X, X)
    D2 = D ** 2
    if t_kernel is None:
        t_kernel = float(np.median(D)) ** 2
    S = np.exp(-D2 / t_kernel)
    return S, 2, dict(t_kernel=float(t_kernel), t_heat=float(t_heat),
                      X=X, D2=D2)


def make_le_helix(n=400, k=5, t_kernel=None, t_heat=_T_HEAT_DEFAULT, seed=0):
    """LE generative form: 1D helix in R^3, S = exp(-D^2 / t_kernel).
    Returns (S, 1, dict(t_kernel, t_heat, X, D2))."""
    rng = np.random.default_rng(seed)
    th = np.sort(rng.random(n)) * 4 * np.pi
    X = np.column_stack([np.cos(th), np.sin(th), th / k])
    D = cdist(X, X)
    D2 = D ** 2
    if t_kernel is None:
        t_kernel = float(np.median(D)) ** 2
    S = np.exp(-D2 / t_kernel)
    return S, 1, dict(t_kernel=float(t_kernel), t_heat=float(t_heat),
                      X=X, D2=D2)


def make_le_sphere(n=400, ambient_dim=6, sigma_noise=0.0, t_kernel=None,
                    t_heat=_T_HEAT_DEFAULT, seed=0):
    """LE generative form: uniform points on the unit (ambient_dim - 1)-sphere
    in R^{ambient_dim}, optionally perturbed by isotropic Gaussian noise of
    standard deviation `sigma_noise` per coordinate. S = exp(-D^2/t_kernel)
    is built from the NOISY ambient distances. Intrinsic manifold dim =
    ambient_dim - 1.

    aux dict contains:
      - X (n x ambient_dim)        clean unit-sphere positions
      - X_noisy (n x ambient_dim)  noisy positions actually used for S
      - D2_clean                   pairwise squared dist of CLEAN points
      - D2_noisy                   pairwise squared dist of NOISY points
      - D2 (alias of D2_noisy)     default V-target (the observable)
      - sigma_noise, t_kernel, t_heat, ambient_dim
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, ambient_dim))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = X / norms
    if sigma_noise > 0:
        X_noisy = X + sigma_noise * rng.standard_normal(X.shape)
    else:
        X_noisy = X
    D_clean = cdist(X, X)
    D_noisy = cdist(X_noisy, X_noisy)
    D2_clean = D_clean ** 2
    D2_noisy = D_noisy ** 2
    if t_kernel is None:
        t_kernel = float(np.median(D_noisy)) ** 2
    S = np.exp(-D2_noisy / t_kernel)
    return S, int(ambient_dim - 1), dict(
        t_kernel=float(t_kernel), t_heat=float(t_heat),
        ambient_dim=int(ambient_dim),
        X=X, X_noisy=X_noisy,
        D2=D2_noisy, D2_clean=D2_clean, D2_noisy=D2_noisy,
        sigma_noise=float(sigma_noise),
    )


# ============================================================
# Registry
# ============================================================

ADVERSARIAL_DATASETS = [
    ("A1_power_law",    lambda: make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=0)),
    ("A2_t3_noise",     lambda: make_t3_noise(n=300, r=5, snr=4.0, seed=0)),
    ("A3_anisotropic",  lambda: make_anisotropic(n=300, r=5, snr=4.0, rho=0.7, seed=0)),
    ("A4_coherent",     lambda: make_coherent(n=300, r=5, snr=4.0, frac_active=0.15, seed=0)),
    ("A5_bbp_twin",     lambda: make_bbp_twin(n=400, lam_inside=1.6, lam_outside=0.7, seed=0)),
    ("A6_mixed_scale",  lambda: make_mixed_scale(n=300, seed=0)),
    ("A7_unequal_blk",  lambda: make_unequal_blocks(n=400, seed=0)),
    ("A8_shifted_off",  lambda: make_shifted_offset(n=300, r=5, offset=10.0, snr=3.0, seed=0)),
]

EGAP_ADVERSARIAL = [
    ("E1_linear_decay20", lambda: make_linear_decay(n=400, r=20, lam_top=12.0, lam_bot=1.5, seed=0)),
    ("E2_two_tier_5_20",  lambda: make_two_tier_spikes(n=400, r_big=5, r_small=20, lam_big=8.0, lam_small=1.5, seed=0)),
    ("E3_powerlaw30",     lambda: make_smooth_powerlaw_hi(n=500, k_signal=30, alpha=0.5, snr=8.0, seed=0)),
    ("E4_nested16",       lambda: make_nested_blocks(n=400, super_K=4, sub_K=4, seed=0)),
]

# Multi-seed registries: each entry is (label, build_fn(seed)) where
# build_fn takes a seed argument (so we can run n_seeds replicates).
# Includes harder variants of A4/A7/A8 and Recipe-K-adversarial R1-R4.
ADVERSARIAL_DATASETS_SEEDED = [
    ("A1_power_law",     lambda seed: make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=seed)),
    ("A2_t3_noise",      lambda seed: make_t3_noise(n=300, r=5, snr=4.0, seed=seed)),
    ("A3_anisotropic",   lambda seed: make_anisotropic(n=300, r=5, snr=4.0, rho=0.7, seed=seed)),
    ("A4_coherent",      lambda seed: make_coherent(n=300, r=5, snr=4.0, frac_active=0.15, seed=seed)),
    ("A4h_coherent_05",  lambda seed: make_coherent(n=300, r=5, snr=3.0, frac_active=0.05, seed=seed)),
    ("A5_bbp_twin",      lambda seed: make_bbp_twin(n=400, lam_inside=1.6, lam_outside=0.7, seed=seed)),
    ("A5h_bbp_twin_close", lambda seed: make_bbp_twin(n=400, lam_inside=1.2, lam_outside=0.85, seed=seed)),
    ("A6_mixed_scale",   lambda seed: make_mixed_scale(n=300, seed=seed)),
    ("A7_unequal_blk",   lambda seed: make_unequal_blocks(n=400, seed=seed)),
    ("A7h_unequal_blk6", lambda seed: make_unequal_blocks(n=400, sizes=(0.45, 0.20, 0.15, 0.10, 0.05, 0.05), seed=seed)),
    ("A8_shifted_off",   lambda seed: make_shifted_offset(n=300, r=5, offset=10.0, snr=3.0, seed=seed)),
    ("A8h_shifted_off_strong", lambda seed: make_shifted_offset(n=300, r=5, offset=20.0, snr=1.5, seed=seed)),
]

EGAP_ADVERSARIAL_SEEDED = [
    ("E1_linear_decay20", lambda seed: make_linear_decay(n=400, r=20, lam_top=12.0, lam_bot=1.5, seed=seed)),
    ("E2_two_tier_5_20",  lambda seed: make_two_tier_spikes(n=400, r_big=5, r_small=20, lam_big=8.0, lam_small=1.5, seed=seed)),
    ("E3_powerlaw30",     lambda seed: make_smooth_powerlaw_hi(n=500, k_signal=30, alpha=0.5, snr=8.0, seed=seed)),
    ("E4_nested16",       lambda seed: make_nested_blocks(n=400, super_K=4, sub_K=4, seed=seed)),
]

RECIPE_K_ADVERSARIAL_SEEDED = [
    ("R1_continuous",     lambda seed: make_continuous_spectrum(n=400, k_signal=30, lam_top=8.0, decay=0.08, seed=seed)),
    ("R2_heteroscedastic", lambda seed: make_heteroscedastic(n=300, r=5, snr=4.0, sigma_lo=0.3, sigma_hi=1.5, seed=seed)),
    ("R3_extreme_coh",    lambda seed: make_extreme_coherence(n=300, r=5, snr=5.0, n_active=10, seed=seed)),
    ("R4_mixed_tail",     lambda seed: make_mixed_tail_noise(n=300, r=5, snr=4.0, sparse_frac=0.02, sparse_amp=8.0, seed=seed)),
]

R4_VARIANTS_SEEDED = [
    ("R4a_heavy_outliers", lambda seed: make_mixed_tail_heavy(n=300, r=5, snr=4.0, sparse_frac=0.02, sparse_amp=20.0, seed=seed)),
    ("R4b_dense_outliers", lambda seed: make_mixed_tail_dense(n=300, r=5, snr=4.0, sparse_frac=0.05, sparse_amp=8.0, seed=seed)),
    ("R4c_cauchy",         lambda seed: make_cauchy_noise(n=300, r=5, snr=4.0, scale=0.3, seed=seed)),
    ("R4d_kitchen_sink",   lambda seed: make_kitchen_sink(n=300, r=5, snr=4.0, sparse_frac=0.03, sparse_amp=15.0, sigma_lo=0.5, sigma_hi=1.5, seed=seed)),
]

HIGH_DIM_SEEDED = [
    ("HD1_kernel_block_n2000", lambda seed: make_kernel_block_big(n=2000, K=20, seed=seed)),
    ("HD2_continuous_n2000",   lambda seed: make_continuous_big(n=2000, k_signal=50, lam_top=12.0, decay=0.05, seed=seed)),
    ("HD3_powerlaw70_n2000",   lambda seed: make_powerlaw_big(n=2000, k_signal=70, alpha=0.5, snr=10.0, seed=seed)),
    ("HD4_nested30_n2000",     lambda seed: make_nested_big(n=2000, super_K=5, sub_K=6, seed=seed)),
]


# ============================================================
# X-suite — existing well-known synthetics (seeded for multi-seed runs)
# ============================================================
def _x_10block(n=200, seed=0):
    """RBF kernel of 10-block-simulation data (the original 'easy' synthetic)."""
    from _generate_examples import _rbf_kernel, simulation
    rng = np.random.default_rng(seed)
    D = simulation(n, 10, ndict=10) + rng.random((n, 10)) * 0.5
    return _rbf_kernel(D, bw=1.0), 10


def _x_dcsbm(n=300, K=5, seed=0):
    from _generate_examples import make_similarity_dcsbm
    return make_similarity_dcsbm(n=n, K=K, seed=seed)[:2]


def _x_multiscale(n1=200, n2=200, seed=1):
    from _generate_examples import make_similarity_multiscale_manifolds
    return make_similarity_multiscale_manifolds(n1=n1, n2=n2, seed=seed)[:2]


def _x_temporal(n=400, regimes=4, seed=2):
    from _generate_examples import make_similarity_temporal
    return make_similarity_temporal(n=n, regimes=regimes, seed=seed)[:2]


def _x_hornfail(n=200, r_true=3, n_clusters=20, seed=0):
    from _generate_examples import make_correlation_horn_fail_challenging
    S, _ = make_correlation_horn_fail_challenging(
        n=n, r_true=r_true, n_clusters=n_clusters, seed=seed
    )
    return S, r_true


X_SUITE_SEEDED = [
    ("X_10block_n200", lambda seed: _x_10block(n=200, seed=seed)),
    ("X_DCSBM",        lambda seed: _x_dcsbm(n=300, K=5, seed=seed)),
    ("X_Multiscale",   lambda seed: _x_multiscale(n1=200, n2=200, seed=1 + seed)),
    ("X_Temporal",     lambda seed: _x_temporal(n=400, regimes=4, seed=2 + seed)),
    ("X_Hornfail",     lambda seed: _x_hornfail(n=200, r_true=3, n_clusters=20, seed=seed)),
]


# ============================================================
# S-suite — ScreeNOT-faithful synthetics (Donoho-Gavish-Romanov 2023)
# ============================================================
# These generators replicate (or symmetrize) the experimental panel from
# the ScreeNOT paper to enable fair comparison on its home-turf model class.
# All return (Y_or_S, X_true, r). Y is rectangular for S1-S5; S is symmetric for S6.
# X_true is the population low-rank component, used for Frobenius MSE.

def make_S1_screenot_fig4a(seed=0):
    """ScreeNOT Fig 4(a) replication: rectangular MP-white at gamma=0.5,
    n=1000, p=500, r=5, x=(0.5, 1.0, 1.3, 2.5, 5.2)."""
    rng = np.random.default_rng(seed)
    n, p, r = 1000, 500, 5
    x = np.array([0.5, 1.0, 1.3, 2.5, 5.2])
    A, _ = np.linalg.qr(rng.standard_normal((n, r)))
    B, _ = np.linalg.qr(rng.standard_normal((p, r)))
    X = (A * x) @ B.T
    Z = rng.standard_normal((n, p)) / np.sqrt(n)
    return X + Z, X, r


def make_S2_mix2_bulk(seed=0):
    """ScreeNOT Fig 5 panel (ii): correlated rows, dF_S = 0.5*delta_1 + 0.5*delta_10,
    gamma=0.5. Two-point covariance bulk."""
    rng = np.random.default_rng(seed)
    n, p, r = 1000, 500, 5
    x = np.linspace(2.0, 6.0, r)
    half = p // 2
    s_diag = np.concatenate([np.ones(half), 10.0 * np.ones(p - half)])
    S_root = np.diag(np.sqrt(s_diag))
    W = rng.standard_normal((n, p)) / np.sqrt(n)
    Z = W @ S_root
    A, _ = np.linalg.qr(rng.standard_normal((n, r)))
    B, _ = np.linalg.qr(rng.standard_normal((p, r)))
    X = (A * x) @ B.T
    return X + Z, X, r


def make_S3_unif_bulk(seed=0):
    """ScreeNOT Fig 5 panel (iii): F_S = Unif[1,10] continuous-coloured bulk,
    gamma=0.5."""
    rng = np.random.default_rng(seed)
    n, p, r = 1000, 500, 5
    x = np.linspace(2.0, 6.0, r)
    s_diag = rng.uniform(1.0, 10.0, size=p)
    S_root = np.diag(np.sqrt(s_diag))
    W = rng.standard_normal((n, p)) / np.sqrt(n)
    Z = W @ S_root
    A, _ = np.linalg.qr(rng.standard_normal((n, r)))
    B, _ = np.linalg.qr(rng.standard_normal((p, r)))
    X = (A * x) @ B.T
    return X + Z, X, r


def make_S4_square_white(seed=0):
    """ScreeNOT Fig 4(d) replication: gamma=1.0 square MP-white, n=p=1000,
    r=10, x=(1, 2, ..., 10)."""
    rng = np.random.default_rng(seed)
    n, p, r = 1000, 1000, 10
    x = np.arange(1, r + 1, dtype=float)
    A, _ = np.linalg.qr(rng.standard_normal((n, r)))
    B, _ = np.linalg.qr(rng.standard_normal((p, r)))
    X = (A * x) @ B.T
    Z = rng.standard_normal((n, p)) / np.sqrt(n)
    return X + Z, X, r


def make_S5_mixed_bbp(seed=0):
    """Decisive test: mixed sub/super-BBP spikes. r=6 with x=(0.3, 0.5, 0.8, 1.5, 3.0, 6.0).
    BBP threshold for gamma=0.5 is x_+ ≈ 1, so 3 spikes (0.3, 0.5, 0.8) are SUB-BBP and
    should be CORRECTLY DROPPED by an MSE-optimal rank selector. ScreeNOT-correct k=3,
    structural rank=6.
    """
    rng = np.random.default_rng(seed)
    n, p = 1000, 500
    x = np.array([0.3, 0.5, 0.8, 1.5, 3.0, 6.0])
    r = len(x)
    A, _ = np.linalg.qr(rng.standard_normal((n, r)))
    B, _ = np.linalg.qr(rng.standard_normal((p, r)))
    X = (A * x) @ B.T
    Z = rng.standard_normal((n, p)) / np.sqrt(n)
    return X + Z, X, r  # r=6 structural; ScreeNOT-correct rank ≈ 3


def make_S6_symm_wigner(seed=0):
    """Symmetric Wigner version of S1: square symmetric spiked Wigner.
    Bridges the rectangular-vs-symmetric comparison.
    n=1000, r=5, eigenvalues x=(0.5, 1.0, 1.3, 2.5, 5.2), Wigner noise."""
    rng = np.random.default_rng(seed)
    n, r = 1000, 5
    x = np.array([0.5, 1.0, 1.3, 2.5, 5.2])
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    X = U @ np.diag(x) @ U.T
    # Symmetric Wigner noise with entry variance 1/n
    Z = rng.standard_normal((n, n))
    W = (Z + Z.T) / np.sqrt(2 * n)
    return X + W, X, r


SCREENOT_FAITHFUL_SEEDED = [
    ("S1_screenot_fig4a", lambda seed: make_S1_screenot_fig4a(seed=seed)),
    ("S2_mix2_bulk",      lambda seed: make_S2_mix2_bulk(seed=seed)),
    ("S3_unif_bulk",      lambda seed: make_S3_unif_bulk(seed=seed)),
    ("S4_square_white",   lambda seed: make_S4_square_white(seed=seed)),
    ("S5_mixed_bbp",      lambda seed: make_S5_mixed_bbp(seed=seed)),
    ("S6_symm_wigner",    lambda seed: make_S6_symm_wigner(seed=seed)),
]


# ============================================================
# Group expansions (2026-05-08 round 3): add variety + high-rank cases
# ============================================================

# --- Group 4: anisotropic / heteroscedastic noise ---

def make_AR2_noise(n=300, r=5, snr=4.0, phi1=0.6, phi2=0.3, seed=0):
    """G4-AR2: AR(2) row-correlated noise.
    z_{ij} = phi1 * z_{i,j-1} + phi2 * z_{i,j-2} + eps; (phi1+phi2=0.9, near
    unit-root, broad noise spectrum).
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r)); U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_normal((n, n))
    c = np.zeros((n, n))
    c[:, 0] = Z[:, 0]
    c[:, 1] = phi1 * c[:, 0] + Z[:, 1]
    for j in range(2, n):
        c[:, j] = phi1 * c[:, j - 1] + phi2 * c[:, j - 2] + np.sqrt(1 - phi1**2 - phi2**2) * Z[:, j]
    T = (c + c.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_matern32_noise(n=300, r=5, snr=4.0, length_scale=20, seed=0):
    """G4-Matern32: spatial-correlation noise with Matern nu=3/2 kernel covariance.
    Sigma_{ij} = (1 + sqrt(3)|i-j|/l) * exp(-sqrt(3)|i-j|/l). PSD by construction.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r)); U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    s3 = np.sqrt(3.0)
    Sigma = (1 + s3 * d / length_scale) * np.exp(-s3 * d / length_scale)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))
    Z = rng.standard_normal((n, n))
    c = Z @ L.T
    T = (c + c.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_block_hetero(n=400, r=5, snr=4.0, sigmas=(0.3, 0.7, 1.2, 2.0), seed=0):
    """G4-blockhetero: block-wise heteroscedastic noise. n divided into len(sigmas)
    equal blocks, block k has noise std sigma_k. Single global sigma estimator
    fails."""
    rng = np.random.default_rng(seed)
    K = len(sigmas)
    block = n // K
    sigma_vec = np.zeros(n)
    for k in range(K):
        sigma_vec[k*block:(k+1)*block] = sigmas[k]
    if K * block < n:
        sigma_vec[K*block:] = sigmas[-1]
    U = rng.standard_normal((n, r)); U, _ = np.linalg.qr(U)
    sig = U @ np.diag(np.linspace(snr * 1.5, snr, r)) @ U.T
    Z = rng.standard_normal((n, n)) * sigma_vec[:, None]
    T = (Z + Z.T) / (2 * np.sqrt(2 * n))
    return sig + T, r


def make_strong_aniso(n=300, r=5, snr=4.0, rho=0.9, seed=0):
    """G4-stronganiso: AR(1) noise with rho=0.9 (extreme anisotropy)."""
    return make_anisotropic(n=n, r=r, snr=snr, rho=rho, seed=seed)


# --- Group 3: high-rank heavy-tail variants ---

def make_mixed_tail_r20(n=400, r=20, snr=6.0, sparse_frac=0.02, sparse_amp=8.0, seed=0):
    """R4_r20: rank-20 signal + mixed-tail noise (Gaussian + sparse outliers)."""
    return make_mixed_tail_noise(n=n, r=r, snr=snr, sparse_frac=sparse_frac,
                                   sparse_amp=sparse_amp, seed=seed)


def make_cauchy_r20(n=400, r=20, snr=6.0, scale=0.3, seed=0):
    """R4c_r20: rank-20 signal + Cauchy noise. NOTE: under Cauchy noise, the
    population 'rank' is not well-defined (no finite moments). The label r=20
    is the *generative* rank but the population *detectable* rank is undefined
    (Soshnikov / Auffinger-Ben Arous-Péché regime). We keep this as a probe of
    'does the method abstain or commit'."""
    return make_cauchy_noise(n=n, r=r, snr=snr, scale=scale, seed=seed)


def make_kitchen_sink_r20(n=400, r=20, snr=6.0, sparse_frac=0.03, sparse_amp=15.0,
                            sigma_lo=0.5, sigma_hi=1.5, seed=0):
    """R4d_r20: rank-20 + kitchen sink (heteroscedastic + outliers)."""
    return make_kitchen_sink(n=n, r=r, snr=snr, sparse_frac=sparse_frac,
                               sparse_amp=sparse_amp, sigma_lo=sigma_lo,
                               sigma_hi=sigma_hi, seed=seed)


# --- Group 5: high-rank kernel variants ---

def make_kernel_K30(n=1600, K=30, bw=1.0, seed=0):
    """X_10block_K30 / HD1_kernel_K30: RBF kernel of K=30 Gaussian clusters."""
    return make_kernel_block_big(n=n, K=K, bw=bw, seed=seed)


def make_DCSBM_K20(n=1600, K=20, seed=0):
    """X_DCSBM_K20: degree-corrected SBM with K=20 blocks."""
    from _generate_examples import make_similarity_dcsbm
    return make_similarity_dcsbm(n=n, K=K, seed=seed)[:2]


def make_multiscale_d5(n=1600, d=5, h=None, seed=0):
    """X_Multiscale_d5: RBF kernel of points uniform on [0,1]^d. d=5 intrinsic.
    Effective rank depends on bandwidth h; we report 'effective rank at 1%
    cumulative spectral mass cutoff' as ground truth.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    if h is None:
        h = float(np.sqrt(np.median(D2[D2 > 0])))
    S = np.exp(-D2 / (h ** 2))
    # Compute effective rank (top eigvals capturing 99% of spectrum)
    eigs = np.linalg.eigvalsh(S)[::-1]
    eigs = np.maximum(eigs, 0)
    cum = np.cumsum(eigs) / eigs.sum()
    eff_r = int(np.searchsorted(cum, 0.99) + 1)
    return S, eff_r


GROUP4_EXPANSION_SEEDED = [
    ("G4_AR2",          lambda seed: make_AR2_noise(n=300, r=5, snr=4.0, seed=seed)),
    ("G4_AR2_r20",      lambda seed: make_AR2_noise(n=400, r=20, snr=6.0, seed=seed)),
    ("G4_Matern32",     lambda seed: make_matern32_noise(n=300, r=5, snr=4.0, seed=seed)),
    ("G4_blockhetero",  lambda seed: make_block_hetero(n=400, r=5, snr=4.0, seed=seed)),
    ("G4_blockhetero_r20", lambda seed: make_block_hetero(n=400, r=20, snr=6.0, seed=seed)),
    ("G4_stronganiso",  lambda seed: make_strong_aniso(n=300, r=5, snr=4.0, rho=0.9, seed=seed)),
    ("G4_stronganiso_r20", lambda seed: make_strong_aniso(n=400, r=20, snr=6.0, rho=0.9, seed=seed)),
]

GROUP3_HIGHRANK_SEEDED = [
    ("R4_r20",          lambda seed: make_mixed_tail_r20(n=400, r=20, seed=seed)),
    ("R4c_r20",         lambda seed: make_cauchy_r20(n=400, r=20, seed=seed)),
    ("R4d_r20",         lambda seed: make_kitchen_sink_r20(n=400, r=20, seed=seed)),
]

GROUP5_HIGHRANK_SEEDED = [
    ("X_10block_K30",   lambda seed: make_kernel_K30(n=1600, K=30, seed=seed)),
    ("X_DCSBM_K20",     lambda seed: make_DCSBM_K20(n=1600, K=20, seed=seed)),
    ("HD1_kernel_K30",  lambda seed: make_kernel_K30(n=2000, K=30, seed=seed)),
    ("X_Multiscale_d5", lambda seed: make_multiscale_d5(n=1600, d=5, seed=seed)),
]


def all_seeded_datasets():
    """Combined registry of all unique seeded synthetics across experiments A,
    A_comprehensive, A_egap_targeted, A_harder.
    """
    seen = set()
    out = []
    for reg in [ADVERSARIAL_DATASETS_SEEDED, EGAP_ADVERSARIAL_SEEDED,
                 RECIPE_K_ADVERSARIAL_SEEDED, R4_VARIANTS_SEEDED,
                 HIGH_DIM_SEEDED, X_SUITE_SEEDED]:
        for label, fn in reg:
            if label not in seen:
                seen.add(label)
                out.append((label, fn))
    return out

REPRESENTATION_DATASETS = [
    ("ppca",     lambda: make_ppca_synth(n=300, r=5, sigma=0.4, seed=0)),
    ("softimp",  lambda: make_softimp_synth(n=300, r=8, snr=3.0, seed=0)),
    ("nmf",      lambda: make_nmf_synth(n=300, r=10, seed=0)),
    ("rpca",     lambda: make_rpca_synth(n=300, r=5, seed=0)),
]

MANIFOLD_DATASETS = [
    ("swissroll", lambda: make_swissroll(n=400, seed=0)),
    ("torus",     lambda: make_torus(n=400, seed=0)),
    ("helix",     lambda: make_helix(n=400, seed=0)),
]
