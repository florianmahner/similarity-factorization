"""Compare RSA vs SRF-LOO using Gaussian tuning simulation.

Generative model:
- X = positions in [0,1]^K (latent stimulus space)
- R = gaussian_tuning(X) (population response via Gaussian basis functions)
- S = R @ R.T (similarity from responses)

Key insight: SRF recovers W ≈ R (responses), not X (positions).
So we test W against R for the LOO test.

For RSA, we use Gaussian kernel hypothesis per dimension.
"""

import numpy as np
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from pysrf import SRF
from src.utils import get_output_dir
from src.utils.simulation import generate_tuning_simulation
from src.colors import TEAL, CYAN, GRAY_DARK
from src.utils.figure_theme import create_figure, despine
from src.tools.rsa import mantel_test, loo_alignment_test_multi

OUTPUT_DIR = get_output_dir()


def run_single_rep(
    n_objects: int,
    n_dims: int,
    n_neurons_per_dim: int,
    tuning_width: float,
    snr: float,
    n_perm: int,
    seed: int,
) -> dict:
    """Run one simulation rep."""
    rng = np.random.default_rng(seed)

    # Generate simulation data
    sim = generate_tuning_simulation(
        n_objects=n_objects,
        n_dims=n_dims,
        n_neurons_per_dim=n_neurons_per_dim,
        tuning_width=tuning_width,
        snr=snr,
        similarity="dot",  # Use dot product (valid kernel)
        rng=rng,
    )
    X = sim["positions"]  # Latent positions (n_objects, n_dims)
    R = sim["responses"]  # Population responses (n_objects, n_neurons)
    S = sim["similarity"]  # Measured similarity

    # Fit SRF - use rank = n_neurons to match response dimensionality
    # But for fair comparison, use n_dims (testing if low-rank captures structure)
    W = SRF(rank=n_dims * n_neurons_per_dim, verbose=False, tol=0.0, random_state=seed).fit_transform(S)

    # === RSA: Mantel test per dimension with Gaussian kernel hypothesis ===
    rsa_ps = []
    for i in range(n_dims):
        x_i = X[:, [i]]
        # Gaussian kernel hypothesis: exp(-(x_i - x_j)^2 / 2σ^2)
        dist_sq = (x_i - x_i.T) ** 2
        H = np.exp(-dist_sq / (2 * tuning_width**2))
        p, _, _ = mantel_test(H, S, permutations=n_perm, random_state=seed + i, two_sided=True)
        rsa_ps.append(p)

    # === SRF-LOO: Test W against responses R (what SRF actually recovers) ===
    srf_results = loo_alignment_test_multi(
        W, R, permutations=n_perm, alpha=0.05, random_state=seed + 200
    )

    # FDR correction for RSA
    rsa_reject = multipletests(rsa_ps, alpha=0.05, method="fdr_bh")[0]

    return {
        "rsa_power": rsa_reject.mean(),
        "loo_power": srf_results["significant"].mean(),
        "rsa_ps": rsa_ps,
        "loo_ps": srf_results["raw_p"].tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RSA vs SRF-LOO: Gaussian Tuning Simulation")
    print("=" * 60)

    # Simulation parameters
    n_reps = 30
    n_objects = 100
    n_dims = 4
    n_neurons_per_dim = 10
    tuning_width = 0.15
    n_perm = 500
    snr_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    print(f"\nParameters:")
    print(f"  Objects: {n_objects}, Dims: {n_dims}")
    print(f"  Neurons/dim: {n_neurons_per_dim}, Tuning width: {tuning_width}")
    print(f"  Reps: {n_reps}, Permutations: {n_perm}")

    results = {"snr": snr_levels, "rsa_power": [], "loo_power": [], "rsa_ps": [], "loo_ps": []}

    for snr in snr_levels:
        print(f"  SNR={snr:.2f}...")

        rep_results = Parallel(n_jobs=-1)(
            delayed(run_single_rep)(
                n_objects, n_dims, sparsity, snr, n_perm,
                seed=rep * 1000 + int(snr * 100)
            )
            for rep in range(n_reps)
        )

        results["rsa_power"].append(np.mean([r["rsa_power"] for r in rep_results]))
        results["loo_power"].append(np.mean([r["loo_power"] for r in rep_results]))
        results["rsa_ps"].append([p for r in rep_results for p in r["rsa_ps"]])
        results["loo_ps"].append([p for r in rep_results for p in r["loo_ps"]])

    # Print results
    print("\n" + "=" * 60)
    print("POWER (should be ≤5% at SNR=0):")
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
    ax.set_title("Synthetic embedding: Power")
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

    fig.savefig(OUTPUT_DIR / "synthetic_power.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUTPUT_DIR / 'synthetic_power.png'}")


if __name__ == "__main__":
    main()
