"""Debug NSD bounds estimation - why is pmax=0?"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.nsd_utils import get_roi, load_nsd_betas
from tools.metrics import gaussian_kernel_similarity
from pysrf.bounds import (
    precompute_matrix_info,
    pmin_bound,
    _p_upper_only_k_ultra,
    lambda_bulk_dyson_raw,
)


def analyze_similarity_matrix(S: np.ndarray, name: str) -> dict:
    """Analyze properties of a similarity matrix."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    n = S.shape[0]
    print(f"Shape: {S.shape}")
    print(f"Range: [{S.min():.6f}, {S.max():.6f}]")
    print(f"Mean (off-diag): {S[~np.eye(n, dtype=bool)].mean():.6f}")
    print(f"Std (off-diag): {S[~np.eye(n, dtype=bool)].std():.6f}")

    # Eigenvalue analysis
    print("\nComputing eigenvalues...")
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(eigvals)[::-1]  # Descending

    print(f"Top 10 eigenvalues: {eigvals[:10]}")
    print(f"Bottom 5 eigenvalues: {eigvals[-5:]}")
    print(f"Number of negative eigenvalues: {(eigvals < 0).sum()}")
    print(f"Number of near-zero eigenvalues (|λ| < 1e-6): {(np.abs(eigvals) < 1e-6).sum()}")

    # Effective dimension
    fro_norm = np.linalg.norm(S, 'fro')
    spectral_norm = np.abs(eigvals[0])
    eff_dim = int(np.ceil((fro_norm / spectral_norm) ** 2))
    print(f"\nFrobenius norm: {fro_norm:.4f}")
    print(f"Spectral norm: {spectral_norm:.4f}")
    print(f"Effective dimension: {eff_dim}")

    # Eigenvalue concentration
    cumsum = np.cumsum(eigvals) / eigvals.sum()
    n_90 = np.searchsorted(cumsum, 0.9) + 1
    n_99 = np.searchsorted(cumsum, 0.99) + 1
    print(f"Eigenvalues for 90% variance: {n_90}")
    print(f"Eigenvalues for 99% variance: {n_99}")

    # Pre-computed info for bounds
    info = precompute_matrix_info(S)
    print(f"\nPrecomputed info:")
    print(f"  s_norm: {info.s_norm:.6f}")
    print(f"  s2_max: {info.s2_max:.6f}")
    print(f"  fro_norm: {info.fro_norm:.6f}")
    print(f"  eff_dim: {info.eff_dim}")

    return {
        "name": name,
        "shape": S.shape,
        "eigvals": eigvals,
        "eff_dim": eff_dim,
        "info": info,
    }


def test_bounds_components(S: np.ndarray, info, name: str):
    """Test individual bounds components."""
    print(f"\n{'='*60}")
    print(f"Testing bounds components: {name}")
    print(f"{'='*60}")

    # Test pmin
    print("\n1. Testing pmin_bound...")
    pmin_result = pmin_bound(S, random_state=42, verbose=True)
    pmin = pmin_result[0]
    print(f"   pmin = {pmin:.6f}")

    # Test lambda_bulk at different p values
    print("\n2. Testing lambda_bulk_dyson at different p values...")
    test_p = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for p in test_p:
        try:
            edge = lambda_bulk_dyson_raw(S, p, omega=0.8, eta=1e-3, jump_frac=0.1)
            print(f"   p={p:.1f}: lambda_bulk = {edge:.6f}")
        except Exception as e:
            print(f"   p={p:.1f}: ERROR - {e}")

    # Test p_upper at different k values
    print("\n3. Testing p_upper_only_k at different k values...")
    for k in [1, 5, 10, info.eff_dim]:
        try:
            pmax = _p_upper_only_k_ultra(
                S, k=k, info=info, tol=1e-4, omega=0.8,
                eta=1e-3, jump_frac=0.1, verbose=True
            )
            print(f"   k={k}: pmax = {pmax:.6f}")
        except Exception as e:
            print(f"   k={k}: ERROR - {e}")

    # Check eigenvalue vs bulk edge
    print("\n4. Eigenvalue vs bulk edge analysis...")
    eigvals = info.eigvals  # Already sorted descending
    for p in [0.1, 0.3, 0.5]:
        edge = lambda_bulk_dyson_raw(S, p, omega=0.8, eta=1e-3, jump_frac=0.1)
        n_above = np.sum(p * eigvals > edge)
        print(f"   p={p:.1f}: edge={edge:.4f}, n_eigenvalues above edge: {n_above}")


def main():
    output_dir = Path(__file__).parent / "outputs" / "debug_bounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NSD subject 1 data
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

    # Compute gaussian kernel RSM
    print("Computing gaussian kernel RSM...")
    S_nsd = gaussian_kernel_similarity(betas, betas, sigma=None)

    # Analyze NSD
    nsd_analysis = analyze_similarity_matrix(S_nsd, "NSD (gaussian kernel)")

    # Test bounds components
    test_bounds_components(S_nsd, nsd_analysis["info"], "NSD")

    # Compare with a synthetic low-rank matrix that works
    print("\n\n" + "="*60)
    print("COMPARISON: Synthetic low-rank matrix")
    print("="*60)
    n = 500
    rank = 10
    rng = np.random.RandomState(42)
    W = rng.rand(n, rank)
    S_synth = W @ W.T
    S_synth = (S_synth + S_synth.T) / 2

    synth_analysis = analyze_similarity_matrix(S_synth, "Synthetic (rank 10)")
    test_bounds_components(S_synth, synth_analysis["info"], "Synthetic")

    # Plot eigenvalue spectra comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(nsd_analysis["eigvals"][:100], 'b-', label='NSD')
    ax.semilogy(synth_analysis["eigvals"][:100], 'r--', label='Synthetic')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Top 100 eigenvalues')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cumsum_nsd = np.cumsum(nsd_analysis["eigvals"]) / nsd_analysis["eigvals"].sum()
    cumsum_synth = np.cumsum(synth_analysis["eigvals"]) / synth_analysis["eigvals"].sum()
    ax.plot(cumsum_nsd[:200], 'b-', label='NSD')
    ax.plot(cumsum_synth[:200], 'r--', label='Synthetic')
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90%')
    ax.axhline(0.99, color='gray', linestyle='--', alpha=0.5, label='99%')
    ax.set_xlabel('Number of eigenvalues')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('Variance explained')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "eigenvalue_comparison.png", dpi=150)
    print(f"\nPlot saved to: {output_dir / 'eigenvalue_comparison.png'}")


if __name__ == "__main__":
    main()
