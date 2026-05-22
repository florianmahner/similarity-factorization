"""Quick calibration check for the fixed LOO test.

Tests only SNR=0 (null hypothesis) to verify FPR ≤ 5%.
Uses fewer permutations for speed since this is computationally expensive.
"""

import numpy as np
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from pysrf import SRF
from src.utils import get_output_dir
from src.utils.helpers import add_positive_noise_with_snr
from src.colors import TEAL, CYAN, GRAY_DARK
from src.utils.figure_theme import create_figure, despine
from src.tools.rsa import mantel_test, loo_alignment_test_multi

OUTPUT_DIR = get_output_dir()


def generate_data(n_objects, n_dims, sparsity=0.3, rng=None):
    """Generate sparse positive embedding (like SPOSE)."""
    rng = np.random.default_rng(rng)
    X = np.abs(rng.standard_normal((n_objects, n_dims)))
    if sparsity > 0:
        mask = rng.random((n_objects, n_dims)) > sparsity
        X = X * mask
    return X


def run_single_test(n_objects, n_dims, snr, n_perm, seed):
    """Run one test at given SNR."""
    rng = np.random.default_rng(seed)

    # Generate ground truth
    X = generate_data(n_objects, n_dims, sparsity=0.3, rng=rng)

    # Add noise and compute similarity
    if snr < 1.0:
        X_noisy = add_positive_noise_with_snr(X, snr, rng=rng)
    else:
        X_noisy = X
    S = X_noisy @ X_noisy.T

    # Fit SRF
    W = SRF(rank=n_dims, verbose=False, tol=0.0, random_state=seed).fit_transform(S)

    # RSA: test each dimension
    rsa_ps = []
    for i in range(n_dims):
        x_i = X[:, [i]]
        H = x_i @ x_i.T
        p, _, _ = mantel_test(H, S, permutations=n_perm, random_state=seed + i, two_sided=True)
        rsa_ps.append(p)

    # LOO: use fixed implementation
    loo_results = loo_alignment_test_multi(W, X, permutations=n_perm, alpha=0.05, random_state=seed + 100)

    # FDR correction
    rsa_reject = multipletests(rsa_ps, alpha=0.05, method="fdr_bh")[0]

    return {
        "rsa_power": rsa_reject.mean(),
        "loo_power": loo_results["significant"].mean(),
        "rsa_ps": rsa_ps,
        "loo_ps": loo_results["raw_p"].tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Calibration Check: Fixed LOO Test")
    print("=" * 60)

    # Parameters - use fewer permutations for speed (new test is expensive)
    n_reps = 20
    n_objects = 50  # Smaller for speed
    n_dims = 5  # Fewer dims for speed
    n_perm = 200  # Fewer perms for speed

    print(f"\nParameters:")
    print(f"  Objects: {n_objects}, Dims: {n_dims}")
    print(f"  Reps: {n_reps}, Permutations: {n_perm}")
    print(f"\nNote: Using fewer permutations because LOO re-alignment is expensive")

    results = {"rsa_ps": [], "loo_ps": []}

    for snr, label in [(0.0, "Calibration (SNR=0)"), (1.0, "Power (SNR=1)")]:
        print(f"\n{label}...")

        rep_results = Parallel(n_jobs=-1)(
            delayed(run_single_test)(n_objects, n_dims, snr, n_perm, seed=rep * 1000)
            for rep in range(n_reps)
        )

        rsa_power = np.mean([r["rsa_power"] for r in rep_results])
        loo_power = np.mean([r["loo_power"] for r in rep_results])

        print(f"  RSA: {rsa_power * 100:.1f}%")
        print(f"  LOO: {loo_power * 100:.1f}%")

        if snr == 0.0:
            results["rsa_ps"] = [p for r in rep_results for p in r["rsa_ps"]]
            results["loo_ps"] = [p for r in rep_results for p in r["loo_ps"]]
            results["rsa_calib"] = rsa_power
            results["loo_calib"] = loo_power
        else:
            results["rsa_power"] = rsa_power
            results["loo_power"] = loo_power

    # Plot p-value distributions at SNR=0
    fig, axes = create_figure("wide", ncols=2)

    ax = axes[0]
    x = [0, 1]
    ax.bar([0, 1], [results["rsa_calib"] * 100, results["loo_calib"] * 100],
           color=[CYAN, TEAL], width=0.5)
    ax.axhline(5, color="red", ls="--", lw=2, label="5% target")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RSA", "SRF-LOO"])
    ax.set_ylabel("False positive rate (%)")
    ax.set_title("Calibration at SNR=0")
    ax.set_ylim([0, 15])
    ax.legend()
    despine(ax)

    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(results["rsa_ps"], bins=bins, alpha=0.5, color=CYAN, label="RSA", density=True)
    ax.hist(results["loo_ps"], bins=bins, alpha=0.5, color=TEAL, label="LOO", density=True)
    ax.axhline(1, color=GRAY_DARK, ls="--", lw=1.5, label="Uniform")
    ax.axvline(0.05, color="red", ls=":", lw=1.5)
    ax.set_xlabel("p-value at SNR=0")
    ax.set_ylabel("Density")
    ax.set_title("P-value distribution (should be uniform)")
    ax.legend(fontsize=8)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "calibration_check.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUTPUT_DIR / 'calibration_check.png'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Calibration (SNR=0, target ≤5%):")
    print(f"  RSA: {results['rsa_calib']*100:.1f}% {'✓' if results['rsa_calib'] <= 0.05 else '✗'}")
    print(f"  LOO: {results['loo_calib']*100:.1f}% {'✓' if results['loo_calib'] <= 0.05 else '✗'}")
    print(f"\nPower (SNR=1):")
    print(f"  RSA: {results['rsa_power']*100:.1f}%")
    print(f"  LOO: {results['loo_power']*100:.1f}%")


if __name__ == "__main__":
    main()
