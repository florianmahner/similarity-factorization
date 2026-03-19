"""Quick debug of NSD bounds - use subsampled matrix for fast analysis."""
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.nsd_utils import get_roi, load_nsd_betas
from tools.metrics import gaussian_kernel_similarity
from pysrf.bounds import (
    precompute_matrix_info,
    pmin_bound,
    _p_upper_only_k_ultra,
    lambda_bulk_dyson_raw,
    estimate_sampling_bounds_ultra,
)


def quick_analysis(S: np.ndarray, name: str):
    """Quick analysis without full eigendecomposition."""
    print(f"\n{'='*60}")
    print(f"Quick analysis: {name}")
    print(f"{'='*60}")

    n = S.shape[0]
    print(f"Shape: {S.shape}")
    print(f"Diagonal: min={S.diagonal().min():.6f}, max={S.diagonal().max():.6f}")
    print(f"Off-diagonal range: [{S[~np.eye(n, dtype=bool)].min():.6f}, {S[~np.eye(n, dtype=bool)].max():.6f}]")
    print(f"Off-diagonal mean: {S[~np.eye(n, dtype=bool)].mean():.6f}")
    print(f"Off-diagonal std: {S[~np.eye(n, dtype=bool)].std():.6f}")

    # Check for issues
    print(f"\nNaN count: {np.isnan(S).sum()}")
    print(f"Inf count: {np.isinf(S).sum()}")
    print(f"Is symmetric: {np.allclose(S, S.T)}")

    # Top 20 eigenvalues only (much faster)
    print("\nComputing top 20 eigenvalues...")
    from scipy.sparse.linalg import eigsh
    try:
        top_eigvals = eigsh(S, k=20, which='LM', return_eigenvectors=False)
        top_eigvals = np.sort(top_eigvals)[::-1]
        print(f"Top 20 eigenvalues: {top_eigvals}")
    except Exception as e:
        print(f"eigsh failed: {e}")
        print("Falling back to full eigvalsh on smaller matrix...")

    # Precomputed info
    print("\nComputing precomputed info...")
    info = precompute_matrix_info(S)
    print(f"  s_norm (spectral): {info.s_norm:.6f}")
    print(f"  s2_max: {info.s2_max:.6f}")
    print(f"  fro_norm: {info.fro_norm:.6f}")
    print(f"  eff_dim: {info.eff_dim}")

    return info


def test_bounds_on_subsample(betas: np.ndarray, n_samples: int = 500):
    """Test bounds on a subsampled version."""
    print(f"\n{'='*60}")
    print(f"Testing on {n_samples} subsampled stimuli")
    print(f"{'='*60}")

    # Subsample stimuli
    rng = np.random.RandomState(42)
    idx = rng.choice(betas.shape[0], n_samples, replace=False)
    betas_sub = betas[idx]

    # Compute RSM
    print("Computing gaussian kernel RSM...")
    S = gaussian_kernel_similarity(betas_sub, betas_sub, sigma=None)
    print(f"RSM shape: {S.shape}")

    # Quick analysis
    info = quick_analysis(S, f"NSD subsample (n={n_samples})")

    # Full eigenvalue analysis on subsample
    print("\nFull eigenvalue analysis on subsample...")
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(eigvals)[::-1]
    print(f"Eigenvalue range: [{eigvals.min():.6f}, {eigvals.max():.6f}]")
    print(f"Negative eigenvalues: {(eigvals < 0).sum()}")
    print(f"Near-zero (|λ|<1e-6): {(np.abs(eigvals) < 1e-6).sum()}")

    # Cumulative variance
    pos_eigvals = eigvals[eigvals > 0]
    cumsum = np.cumsum(pos_eigvals) / pos_eigvals.sum()
    n_90 = np.searchsorted(cumsum, 0.9) + 1
    n_99 = np.searchsorted(cumsum, 0.99) + 1
    print(f"Eigenvalues for 90% variance: {n_90}")
    print(f"Eigenvalues for 99% variance: {n_99}")

    # Test bounds
    print("\nTesting estimate_sampling_bounds_ultra...")
    pmin, pmax, _ = estimate_sampling_bounds_ultra(S, random_state=42, verbose=True)
    print(f"\nResult: pmin={pmin:.6f}, pmax={pmax:.6f}")

    # Debug pmax computation
    print("\nDebugging pmax computation...")
    print(f"Using k = eff_dim = {info.eff_dim}")

    # Test lambda_bulk at different p values
    print("\nlambda_bulk_dyson at different p values:")
    for p in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
        edge = lambda_bulk_dyson_raw(S, p, omega=0.8, eta=1e-3, jump_frac=0.1)
        # Count how many eigenvalues are above p*edge
        n_above = np.sum(p * eigvals > edge)
        print(f"  p={p:.2f}: edge={edge:.4f}, n_eig > edge: {n_above}")

    return S, eigvals, info


def main():
    print("Loading NSD subject 1 data...")
    subject_id = 1
    roi_mask = get_roi(subject_id, "nsdgeneral")
    betas, trials = load_nsd_betas(
        subject_id,
        voxel_indices=roi_mask > 0,
        zscore_betas=True,
        max_workers=8,
    )
    print(f"Betas shape: {betas.shape}")

    # Test on subsample first
    S_sub, eigvals_sub, info_sub = test_bounds_on_subsample(betas, n_samples=500)

    # Now test on larger subsample
    print("\n\n" + "="*60)
    print("Testing on larger subsample (n=1000)")
    print("="*60)
    S_1k, eigvals_1k, info_1k = test_bounds_on_subsample(betas, n_samples=1000)

    # Compare with gaussian kernel properties
    print("\n\n" + "="*60)
    print("GAUSSIAN KERNEL ANALYSIS")
    print("="*60)
    print("""
The Gaussian kernel k(x,y) = exp(-||x-y||²/(2σ²)) produces a dense
similarity matrix. Key properties:
- All values in (0, 1]
- Diagonal = 1
- Symmetric positive definite

For the median heuristic σ = median(||x-y||), the matrix typically has:
- Very high effective dimension (all eigenvalues similar)
- Slow eigenvalue decay

This is VERY different from low-rank behavioral data!
""")

    # Check the median sigma used
    print("Checking sigma used in gaussian_kernel...")
    betas_sub = betas[:500]
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(betas_sub)
    median_dist = np.median(dists[np.triu_indices_from(dists, k=1)])
    print(f"Median pairwise distance: {median_dist:.4f}")
    print(f"Implied sigma (median heuristic): {median_dist:.4f}")


if __name__ == "__main__":
    main()
