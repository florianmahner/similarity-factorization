"""Quick targeted debug of pmax=0 issue."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "third_party/pysrf"))

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)


def create_gaussian_kernel_like_matrix(n: int, mean_sim: float = 0.59, std_sim: float = 0.08):
    """Create a matrix similar to NSD gaussian kernel without loading data."""
    print(f"Creating synthetic gaussian-kernel-like matrix (n={n}, mean={mean_sim})...")

    # Start with constant matrix
    S = np.ones((n, n)) * mean_sim

    # Add noise
    rng = np.random.RandomState(42)
    noise = rng.randn(n, n) * std_sim
    noise = (noise + noise.T) / 2  # Symmetrize
    S = S + noise

    # Clip to valid range and set diagonal
    S = np.clip(S, 0.01, 0.99)
    np.fill_diagonal(S, 1.0)

    # Make symmetric
    S = (S + S.T) / 2

    return S


def analyze_and_test_bounds(S: np.ndarray, name: str):
    """Analyze matrix and test bounds."""
    from pysrf.bounds import (
        precompute_matrix_info,
        pmin_bound,
        _p_upper_only_k_ultra,
        lambda_bulk_dyson_raw,
    )

    n = S.shape[0]
    print(f"\n{'='*60}")
    print(f"Analyzing: {name} (n={n})")
    print(f"{'='*60}")

    # Basic stats
    offdiag = S[~np.eye(n, dtype=bool)]
    print(f"Off-diagonal: mean={offdiag.mean():.4f}, std={offdiag.std():.4f}")

    # Precompute info
    print("Computing precomputed info...")
    info = precompute_matrix_info(S)
    print(f"  s_norm (λ1): {info.s_norm:.4f}")
    print(f"  s2_max: {info.s2_max:.4f}")
    print(f"  eff_dim: {info.eff_dim}")

    # Test lambda_bulk at key p values
    print("\nTesting lambda_bulk_dyson...")
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        try:
            edge = lambda_bulk_dyson_raw(S, p, omega=0.8, eta=1e-3, jump_frac=0.1)
            n_above = np.sum(p * info.eigvals > edge)
            print(f"  p={p:.1f}: edge={edge:.4f}, n_above={n_above}")
        except Exception as e:
            print(f"  p={p:.1f}: ERROR - {e}")

    # Test p_upper_only_k
    print(f"\nTesting _p_upper_only_k_ultra (k={info.eff_dim})...")
    try:
        pmax = _p_upper_only_k_ultra(
            S, k=info.eff_dim, info=info, tol=1e-4, omega=0.8,
            eta=1e-3, jump_frac=0.1, verbose=False
        )
        print(f"  pmax = {pmax:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return info


def main():
    print("Testing pmax=0 issue with synthetic matrices\n")

    # Test different sizes
    for n in [500, 1000, 2000, 3000]:
        S = create_gaussian_kernel_like_matrix(n, mean_sim=0.59, std_sim=0.08)
        analyze_and_test_bounds(S, f"Synthetic (n={n})")

    # Compare with truly low-rank matrix
    print("\n" + "="*60)
    print("COMPARISON: True low-rank matrix")
    print("="*60)
    n, rank = 2000, 10
    rng = np.random.RandomState(42)
    W = rng.rand(n, rank)
    S_lowrank = W @ W.T
    S_lowrank = (S_lowrank + S_lowrank.T) / 2
    # Normalize to [0,1]
    S_lowrank = S_lowrank / S_lowrank.max()
    np.fill_diagonal(S_lowrank, 1.0)

    analyze_and_test_bounds(S_lowrank, "Low-rank (n=2000, rank=10)")


if __name__ == "__main__":
    main()
