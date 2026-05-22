"""Debug scaling issue with NSD bounds."""
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.nsd_utils import get_roi, load_nsd_betas
from tools.metrics import gaussian_kernel_similarity
from pysrf.bounds import (
    precompute_matrix_info,
    _p_upper_only_k_ultra,
    lambda_bulk_dyson_raw,
    estimate_sampling_bounds_ultra,
)


def test_scaling(betas: np.ndarray, sizes: list[int]):
    """Test bounds at different matrix sizes."""
    print(f"{'n':>6} | {'λ1':>12} | {'s2_max':>12} | {'eff_dim':>8} | {'pmin':>8} | {'pmax':>8}")
    print("-" * 70)

    rng = np.random.RandomState(42)

    for n in sizes:
        idx = rng.choice(betas.shape[0], min(n, betas.shape[0]), replace=False)
        betas_sub = betas[idx]

        S = gaussian_kernel_similarity(betas_sub, betas_sub, sigma=None)

        info = precompute_matrix_info(S)

        try:
            pmin, pmax, _ = estimate_sampling_bounds_ultra(S, random_state=42, verbose=False)
        except Exception as e:
            pmin, pmax = float('nan'), float('nan')
            print(f"Error at n={n}: {e}")

        print(f"{n:>6} | {info.s_norm:>12.2f} | {info.s2_max:>12.2f} | {info.eff_dim:>8} | {pmin:>8.4f} | {pmax:>8.4f}")


def check_vde_numerics(S: np.ndarray, name: str):
    """Check VDE solver numerics."""
    print(f"\n{'='*60}")
    print(f"VDE numerics check: {name}")
    print(f"{'='*60}")

    info = precompute_matrix_info(S)
    n = S.shape[0]

    print(f"Matrix size: {n}")
    print(f"s_norm (λ1): {info.s_norm:.4f}")
    print(f"s2_max: {info.s2_max:.4f}")
    print(f"eff_dim: {info.eff_dim}")

    # VDE uses V = p(1-p) * S²
    # z_max = s2_max * p * (1-p) + s_norm
    # For p=0.5: V_max ≈ 0.25 * s2_max

    print(f"\nVDE scaling analysis:")
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        V_scale = p * (1-p) * info.s2_max
        z_max = V_scale + info.s_norm
        print(f"  p={p:.1f}: V_scale={V_scale:.2e}, z_max={z_max:.2e}")

    # The issue: when z_max >> 1, the VDE iteration can become numerically unstable
    # or converge to trivial solutions

    print(f"\nTesting lambda_bulk_dyson at p=0.5...")
    try:
        edge = lambda_bulk_dyson_raw(S, 0.5, omega=0.8, eta=1e-3, jump_frac=0.1)
        print(f"  edge = {edge:.4f}")

        # Check if any eigenvalues are above this edge
        n_above = np.sum(0.5 * info.eigvals > edge)
        print(f"  eigenvalues above edge: {n_above}")
    except Exception as e:
        print(f"  ERROR: {e}")


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

    # Test scaling
    print("\n" + "="*70)
    print("SCALING TEST")
    print("="*70)
    sizes = [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 9000]
    test_scaling(betas, sizes)

    # Check numerics on problematic size
    print("\n" + "="*70)
    print("NUMERICS CHECK ON n=5000")
    print("="*70)
    rng = np.random.RandomState(42)
    idx = rng.choice(betas.shape[0], 5000, replace=False)
    S_5k = gaussian_kernel_similarity(betas[idx], betas[idx], sigma=None)
    check_vde_numerics(S_5k, "NSD n=5000")


if __name__ == "__main__":
    main()
