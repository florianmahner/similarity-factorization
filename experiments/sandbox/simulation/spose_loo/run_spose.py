"""Compare RSA vs SRF-LOO for SPOSE dimension recovery.

Uses actual SPOSE embedding data with the FIXED LOO test
(re-aligns for each permutation to properly account for selection bias).
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
from utils.io import load_spose_embedding

OUTPUT_DIR = get_output_dir()


def run_single_rep(
    full_data: np.ndarray,
    n_objects: int,
    n_dims: int,
    snr: float,
    n_perm: int,
    seed: int,
) -> dict:
    """Run one SPOSE condition."""
    rng = np.random.default_rng(seed)

    # Randomly select dimensions and objects
    dims = rng.choice(full_data.shape[1], size=n_dims, replace=False)
    objects = rng.choice(full_data.shape[0], size=n_objects, replace=False)

    # Extract data for this condition
    data = full_data[objects][:, dims]

    # Add noise and compute similarity
    noisy_data = add_positive_noise_with_snr(data, snr, rng=seed)
    S = noisy_data @ noisy_data.T

    # Fit SRF
    W = SRF(rank=n_dims, verbose=False, tol=0.0, random_state=seed).fit_transform(S)

    # === RSA: Mantel test per dimension ===
    rsa_ps = []
    for i in range(n_dims):
        x_i = data[:, [i]]
        H = x_i @ x_i.T
        p, _, _ = mantel_test(H, S, permutations=n_perm, random_state=seed + i, two_sided=True)
        rsa_ps.append(p)

    # === SRF-LOO: Fixed implementation (re-aligns for each permutation) ===
    loo_results = loo_alignment_test_multi(W, data, permutations=n_perm, alpha=0.05, random_state=seed + 100)

    # FDR correction for RSA
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
    print("RSA vs SRF-LOO: SPOSE (Fixed LOO Test)")
    print("=" * 60)

    # Load SPOSE data
    full_data = load_spose_embedding(num_dims=66)
    print(f"Loaded SPOSE data: {full_data.shape}")

    n_reps = 100  # Match main experiment
    n_objects = 50  # Match main experiment
    n_dims = 5  # Match main experiment (was 10)
    n_perm = 1000  # Match main experiment
    snr_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    print(f"\nParameters: {n_objects} objects, {n_dims} dims, {n_reps} reps, {n_perm} perms")

    results = {"snr": snr_levels, "rsa_power": [], "loo_power": [], "rsa_ps": [], "loo_ps": []}

    for snr in snr_levels:
        print(f"  SNR={snr:.2f}...")

        rep_results = Parallel(n_jobs=-1)(
            delayed(run_single_rep)(full_data, n_objects, n_dims, snr, n_perm, seed=rep * 1000 + int(snr * 100))
            for rep in range(n_reps)
        )

        results["rsa_power"].append(np.mean([r["rsa_power"] for r in rep_results]))
        results["loo_power"].append(np.mean([r["loo_power"] for r in rep_results]))
        results["rsa_ps"].append([p for r in rep_results for p in r["rsa_ps"]])
        results["loo_ps"].append([p for r in rep_results for p in r["loo_ps"]])

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (calibration should be ≤5% at SNR=0):")
    print("=" * 60)
    print(f"{'SNR':<8} {'RSA':<12} {'SRF-LOO':<12}")
    print("-" * 32)
    for i, snr in enumerate(snr_levels):
        print(f"{snr:<8.2f} {results['rsa_power'][i]*100:<12.1f} {results['loo_power'][i]*100:<12.1f}")

    # Plot
    fig, axes = create_figure("wide", ncols=2)

    # Power curves
    ax = axes[0]
    ax.plot(snr_levels, np.array(results["rsa_power"]) * 100, "^-", color=CYAN, lw=2, ms=6, label="RSA")
    ax.plot(snr_levels, np.array(results["loo_power"]) * 100, "s-", color=TEAL, lw=2, ms=6, label="SRF-LOO")
    ax.axhline(5, color=GRAY_DARK, ls="--", lw=1, alpha=0.7, label="α = 5%")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=8)
    ax.set_title("SPOSE: Power comparison")
    despine(ax)

    # P-value distribution at SNR=0
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(results["rsa_ps"][0], bins=bins, alpha=0.5, color=CYAN, label="RSA", density=True)
    ax.hist(results["loo_ps"][0], bins=bins, alpha=0.5, color=TEAL, label="LOO", density=True)
    ax.axhline(1, color=GRAY_DARK, ls="--", lw=1.5, label="Uniform")
    ax.axvline(0.05, color="red", ls=":", lw=1.5)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Calibration at SNR=0")
    ax.legend(fontsize=8)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "spose_power.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUTPUT_DIR / 'spose_power.png'}")


if __name__ == "__main__":
    main()
