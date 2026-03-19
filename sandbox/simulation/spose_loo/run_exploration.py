"""Systematic exploration of simulation models for RSA vs SRF-LOO.

Tests three generative models:
1. Direct embedding: S = X @ X.T (like SPOSE)
2. Linear expansion: R = X @ A, S = R @ R.T
3. Gaussian tuning: R = tuning(X), S = R @ R.T

For each model:
- Visualize ground truth and similarity
- Sanity check: does SRF recover meaningful structure?
- Calibration: FPR at SNR=0 should be ≤5%
- Power: detection rate at SNR=1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from pysrf import SRF
from src.utils import get_output_dir
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import gaussian_tuning_response
from src.colors import TEAL, CYAN, GRAY_DARK
from src.utils.figure_theme import despine
from src.tools.rsa import mantel_test, loo_alignment_test_multi

OUTPUT_DIR = get_output_dir()


# =============================================================================
# Generative Models
# =============================================================================

def generate_direct_embedding(n_objects, n_dims, sparsity=0.3, rng=None):
    """Model 1: Direct embedding S = X @ X.T (like SPOSE)."""
    rng = np.random.default_rng(rng)
    X = np.abs(rng.standard_normal((n_objects, n_dims)))
    if sparsity > 0:
        mask = rng.random((n_objects, n_dims)) > sparsity
        X = X * mask
    return {"X": X, "R": X, "model": "direct"}


def generate_linear_expansion(n_objects, n_dims, expansion=5, rng=None):
    """Model 2: Linear expansion R = X @ A, S = R @ R.T."""
    rng = np.random.default_rng(rng)
    X = np.abs(rng.standard_normal((n_objects, n_dims)))
    # Random projection to higher dimension
    A = rng.standard_normal((n_dims, n_dims * expansion))
    R = X @ A
    R = np.maximum(R, 0)  # Keep positive
    return {"X": X, "R": R, "A": A, "model": "linear"}


def generate_gaussian_tuning(n_objects, n_dims, n_neurons=10, width=0.2, rng=None):
    """Model 3: Gaussian tuning R = tuning(X), S = R @ R.T."""
    rng = np.random.default_rng(rng)
    X = rng.uniform(0, 1, (n_objects, n_dims))
    R = gaussian_tuning_response(X, n_neurons_per_dim=n_neurons, tuning_width=width)
    return {"X": X, "R": R, "model": "tuning", "width": width}


def add_noise_and_compute_similarity(data, snr, rng=None):
    """Add noise to data and compute similarity."""
    if snr < 1.0:
        data_noisy = add_positive_noise_with_snr(data, snr, rng)
    else:
        data_noisy = data
    S = data_noisy @ data_noisy.T
    return S, data_noisy


# =============================================================================
# Visualization
# =============================================================================

def visualize_model(gen_func, name, n_objects=50, n_dims=4, seed=42):
    """Visualize a generative model: X, R, S, and SRF recovery."""
    data = gen_func(n_objects, n_dims, rng=seed)
    X, R = data["X"], data["R"]
    S, _ = add_noise_and_compute_similarity(R, snr=1.0)

    # Fit SRF
    W = SRF(rank=n_dims, verbose=False, random_state=seed).fit_transform(S)

    # Align W to X for visualization
    W_aligned = align_columns(W, X)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Model: {name}", fontsize=14, fontweight='bold')

    # Row 1: Ground truth
    ax = axes[0, 0]
    im = ax.imshow(X, aspect='auto', cmap='viridis')
    ax.set_title(f"Ground truth X\n({X.shape})")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Objects")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(R[:, :min(20, R.shape[1])], aspect='auto', cmap='viridis')
    ax.set_title(f"Responses R\n({R.shape})")
    ax.set_xlabel("Features (first 20)")
    ax.set_ylabel("Objects")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.imshow(S, aspect='auto', cmap='RdBu_r')
    ax.set_title(f"Similarity S = R @ R.T\n({S.shape})")
    ax.set_xlabel("Objects")
    ax.set_ylabel("Objects")
    plt.colorbar(im, ax=ax)

    # Row 2: SRF recovery
    ax = axes[1, 0]
    im = ax.imshow(W_aligned, aspect='auto', cmap='viridis')
    ax.set_title(f"SRF embedding W\n(aligned to X)")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Objects")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    # Correlation between X and W columns
    corrs = [pearsonr(X[:, i], W_aligned[:, i]).statistic for i in range(n_dims)]
    ax.bar(range(n_dims), corrs, color=TEAL)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Correlation (X, W)")
    ax.set_title(f"Column correlations\nmean={np.mean(corrs):.3f}")
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    # Reconstruction error
    S_recon = W @ W.T
    error = np.abs(S - S_recon)
    im = ax.imshow(error, aspect='auto', cmap='Reds')
    ax.set_title(f"Reconstruction error\n|S - WW.T|, mean={error.mean():.3f}")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig, {"X": X, "R": R, "S": S, "W": W, "W_aligned": W_aligned, "corrs": corrs}


def align_columns(W, X):
    """Align W columns to X using Hungarian algorithm."""
    k = min(W.shape[1], X.shape[1])
    W_use = W[:, :k]
    X_use = X[:, :k]

    combined = np.hstack([X_use, W_use])
    corr_full = np.corrcoef(combined, rowvar=False)
    corr = corr_full[:k, k:]

    _, col_perm = linear_sum_assignment(-np.abs(corr))

    # Also handle sign flips
    W_aligned = W_use[:, col_perm].copy()
    for i in range(k):
        if pearsonr(X_use[:, i], W_aligned[:, i]).statistic < 0:
            W_aligned[:, i] *= -1

    return W_aligned


# =============================================================================
# Statistical Tests
# =============================================================================

def run_single_test(gen_func, n_objects, n_dims, snr, n_perm, seed):
    """Run RSA and LOO tests for one repetition."""
    rng = np.random.default_rng(seed)

    # Generate data
    data = gen_func(n_objects, n_dims, rng=rng)
    X, R = data["X"], data["R"]

    # Add noise and compute similarity
    S, _ = add_noise_and_compute_similarity(R, snr, rng=rng)

    # Fit SRF
    W = SRF(rank=n_dims, verbose=False, tol=0.0, random_state=seed).fit_transform(S)

    # RSA: test each X dimension against S
    rsa_ps = []
    for i in range(n_dims):
        x_i = X[:, [i]]
        H = x_i @ x_i.T  # Outer product hypothesis
        p, _, _ = mantel_test(H, S, permutations=n_perm, random_state=seed + i, two_sided=True)
        rsa_ps.append(p)

    # SRF-LOO: test W against X
    loo_results = loo_alignment_test_multi(W, X, permutations=n_perm, alpha=0.05, random_state=seed + 100)

    # FDR correction
    rsa_reject = multipletests(rsa_ps, alpha=0.05, method="fdr_bh")[0]

    return {
        "rsa_power": rsa_reject.mean(),
        "loo_power": loo_results["significant"].mean(),
        "rsa_ps": rsa_ps,
        "loo_ps": loo_results["raw_p"].tolist(),
    }


def run_calibration_and_power(gen_func, name, n_objects=100, n_dims=10, n_reps=30, n_perm=500):
    """Run calibration (SNR=0) and power (SNR=1) tests."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    results = {}
    for snr, label in [(0.0, "calibration"), (1.0, "power")]:
        print(f"  Running {label} (SNR={snr})...")

        rep_results = Parallel(n_jobs=-1)(
            delayed(run_single_test)(gen_func, n_objects, n_dims, snr, n_perm, seed=rep * 1000)
            for rep in range(n_reps)
        )

        results[label] = {
            "rsa_power": np.mean([r["rsa_power"] for r in rep_results]),
            "loo_power": np.mean([r["loo_power"] for r in rep_results]),
            "rsa_ps": [p for r in rep_results for p in r["rsa_ps"]],
            "loo_ps": [p for r in rep_results for p in r["loo_ps"]],
        }

        print(f"    RSA: {results[label]['rsa_power']*100:.1f}%")
        print(f"    LOO: {results[label]['loo_power']*100:.1f}%")

    return results


def plot_results(all_results, output_path):
    """Plot comparison of all models."""
    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 6))

    for i, (name, results) in enumerate(all_results.items()):
        # Power comparison
        ax = axes[0, i]
        x = [0, 1]
        rsa = [results["calibration"]["rsa_power"] * 100, results["power"]["rsa_power"] * 100]
        loo = [results["calibration"]["loo_power"] * 100, results["power"]["loo_power"] * 100]

        ax.plot(x, rsa, "^-", color=CYAN, lw=2, ms=10, label="RSA")
        ax.plot(x, loo, "s-", color=TEAL, lw=2, ms=10, label="SRF-LOO")
        ax.axhline(5, color=GRAY_DARK, ls="--", lw=1, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SNR=0\n(calib)", "SNR=1\n(power)"])
        ax.set_ylabel("Detection rate (%)")
        ax.set_ylim([-5, 105])
        ax.set_title(name, fontweight='bold')
        ax.legend(fontsize=8)
        despine(ax)

        # P-value distribution at SNR=0
        ax = axes[1, i]
        bins = np.linspace(0, 1, 21)
        ax.hist(results["calibration"]["rsa_ps"], bins=bins, alpha=0.5, color=CYAN, label="RSA", density=True)
        ax.hist(results["calibration"]["loo_ps"], bins=bins, alpha=0.5, color=TEAL, label="LOO", density=True)
        ax.axhline(1, color=GRAY_DARK, ls="--", lw=1.5)
        ax.axvline(0.05, color="red", ls=":", lw=1.5)
        ax.set_xlabel("p-value at SNR=0")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Simulation Model Exploration")
    print("=" * 60)

    # Define models
    models = {
        "Direct (S=X@X.T)": lambda n, k, rng: generate_direct_embedding(n, k, sparsity=0.3, rng=rng),
        "Linear (R=X@A)": lambda n, k, rng: generate_linear_expansion(n, k, expansion=5, rng=rng),
        "Gaussian tuning": lambda n, k, rng: generate_gaussian_tuning(n, k, n_neurons=10, width=0.2, rng=rng),
    }

    # Step 1: Visualize each model
    print("\n[1/2] Visualizing models...")
    for name, gen_func in models.items():
        fig, data = visualize_model(gen_func, name, n_objects=50, n_dims=4)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("@", "").replace("=", "")
        fig.savefig(OUTPUT_DIR / f"viz_{safe_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: viz_{safe_name}.png")
        print(f"    Mean column correlation: {np.mean(data['corrs']):.3f}")

    # Step 2: Run calibration and power tests
    print("\n[2/2] Running statistical tests...")
    all_results = {}
    for name, gen_func in models.items():
        all_results[name] = run_calibration_and_power(gen_func, name, n_objects=100, n_dims=10, n_reps=30, n_perm=500)

    # Plot comparison
    fig = plot_results(all_results, OUTPUT_DIR / "model_comparison.png")
    plt.close(fig)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Calib RSA':<12} {'Calib LOO':<12} {'Power RSA':<12} {'Power LOO':<12}")
    print("-" * 73)
    for name, results in all_results.items():
        print(f"{name:<25} "
              f"{results['calibration']['rsa_power']*100:>10.1f}% "
              f"{results['calibration']['loo_power']*100:>10.1f}% "
              f"{results['power']['rsa_power']*100:>10.1f}% "
              f"{results['power']['loo_power']*100:>10.1f}%")

    print(f"\nSaved all figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
