"""Test coherence-based bounds estimation on DINOv3.

The standard pmax estimation fails for dinov3 because:
- Effective dimension is ~1.0007 (ceil gives 2)
- lambda_2 ≈ lambda_3, so no valid p separates exactly 2 eigenvalues
- Result: pmax = 0.0 (broken)

This script tests Ka Chun's projected coherence approach as an alternative.
The key idea: activation_p[k] gives the minimum p needed to detect component k.

Usage:
    ./scripts/submit sandbox/things/dino_bounds/run.py --bg
"""

from pathlib import Path

import numpy as np
from numpy.linalg import eigvalsh
from sklearn.metrics import pairwise_distances, pairwise_kernels

from src.utils import get_output_dir

# Import coherence functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "coherence/projected_iproj"))
from run import compute_projected_coherence, estimate_rank_from_activations

OUTPUT_DIR = get_output_dir()

DINO_FEATURES_PATH = Path("data/features/dinov3/dinov3_features.npy")


def load_rsm():
    """Load DINOv3 features and compute RSM."""
    print("Loading DINOv3 features...")
    features = np.load(DINO_FEATURES_PATH)
    print(f"Features shape: {features.shape}")

    print("Computing RSM with Gaussian kernel (median heuristic)...")
    dist = pairwise_distances(features, metric="euclidean")
    sigma = np.median(dist)
    gamma = 1 / (2 * sigma**2)
    rsm = pairwise_kernels(features, features, metric="rbf", gamma=gamma)
    print(f"RSM shape: {rsm.shape}, range: [{rsm.min():.4f}, {rsm.max():.4f}]")

    return rsm


def analyze_spectrum(rsm):
    """Analyze eigenvalue spectrum."""
    print("\n=== Eigenvalue Analysis ===")
    eigs = np.sort(eigvalsh(rsm))[::-1]

    fro_norm = np.linalg.norm(rsm, "fro")
    spec_norm = np.linalg.norm(rsm, 2)
    eff_dim_raw = (fro_norm / spec_norm) ** 2

    print(f"Top 10 eigenvalues: {eigs[:10].round(2)}")
    print(f"Effective dimension (raw): {eff_dim_raw:.4f}")
    print(f"Effective dimension (ceil): {int(np.ceil(eff_dim_raw))}")
    print(f"Ratio lambda_1/lambda_2: {eigs[0]/eigs[1]:.1f}")
    print(f"Ratio lambda_2/lambda_3: {eigs[1]/eigs[2]:.4f}")

    return eigs


def compute_coherence_bounds(rsm, max_k=20):
    """Compute bounds using coherence approach."""
    print(f"\n=== Computing Projected Coherence (k=1..{max_k}) ===")

    k_list = list(range(1, max_k + 1))
    p_list = np.linspace(0.05, 0.95, 25)

    result = compute_projected_coherence(
        rsm,
        k_list,
        p_list,
        b=30,
        random_state=42,
        use_baseline_correction=True,
        n_jobs=-1,
        show_progress=True,
    )

    return result


def main():
    rsm = load_rsm()
    eigs = analyze_spectrum(rsm)

    # Run coherence analysis
    result = compute_coherence_bounds(rsm, max_k=15)

    # Print activation points
    print("\n=== Activation Points ===")
    print("(Minimum p to detect each component)")
    for i, k in enumerate(result["k_list"]):
        p_act = result["activation_p"][i]
        if np.isfinite(p_act):
            print(f"  k={k:2d}: p_act = {p_act:.3f}")
        else:
            print(f"  k={k:2d}: never activates")

    # Estimate bounds for different target ranks
    print("\n=== Bounds Estimates ===")
    for target_rank in [5, 10, 15]:
        if target_rank <= len(result["activation_p"]):
            p_act = result["activation_p"][target_rank - 1]
            if np.isfinite(p_act):
                print(f"  Rank {target_rank}: pmin (coherence) = {p_act:.4f}")
            else:
                print(f"  Rank {target_rank}: component never activates")

    # Compare with current (broken) bounds
    print("\n=== Comparison with Current Bounds ===")
    print("  Current pmin: 0.0067")
    print("  Current pmax: 0.0 (broken due to eff_dim ceiling)")

    # Save results
    np.savez(
        OUTPUT_DIR / "coherence_results.npz",
        p=result["p"],
        k_list=result["k_list"],
        activation_p=result["activation_p"],
        x_mean=result["x_mean"],
        tau_kp=result["tau_kp"],
    )
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
