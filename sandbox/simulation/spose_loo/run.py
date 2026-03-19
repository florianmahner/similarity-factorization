"""Compare RSA vs SRF-LOO using Gaussian tuning function simulation.

Generative model:
- K continuous latent dimensions (stimulus features in [0, 1])
- N objects positioned randomly in K-dimensional stimulus space
- M neurons per dimension with Gaussian tuning curves
- Similarity computed from population response patterns

This creates realistic continuous structure where:
- Ground truth dimensions are known
- Similarity emerges from population coding (not direct dot product)
- We can control SNR via response noise
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests

from pysrf import SRF
from src.utils import get_output_dir
from src.utils.simulation import generate_tuning_simulation
from src.colors import TEAL, CYAN, GRAY_DARK
from src.utils.figure_theme import create_figure, despine
from tools.rsa import mantel_test

OUTPUT_DIR = get_output_dir()


def loo_alignment(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """LOO column alignment: align W to X using held-out permutation."""
    n, k = W.shape
    W_aligned = np.zeros_like(W)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        combined = np.hstack([X[mask], W[mask]])
        corr_full = np.corrcoef(combined, rowvar=False)
        corr = corr_full[:k, k:]
        _, col_perm = linear_sum_assignment(-np.abs(corr))
        W_aligned[i] = W[i, col_perm]
    return W_aligned


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
        similarity="correlation",
        rng=rng,
    )
    X = sim["positions"]  # Ground truth (n_objects, n_dims)
    S = sim["similarity"]  # Measured similarity

    # Fit SRF
    W = SRF(rank=n_dims, verbose=False, tol=0.0, random_state=seed).fit_transform(S)

    # === RSA: Mantel test per dimension ===
    rsa_ps = []
    for i in range(n_dims):
        # Hypothesis RDM from ground truth dimension
        x_i = X[:, [i]]
        # Use Euclidean distance for continuous dimension
        H = -np.abs(x_i - x_i.T)  # Negative distance = similarity
        p, _, _ = mantel_test(H, S, permutations=n_perm, random_state=seed + i, two_sided=True)
        rsa_ps.append(p)

    # === SRF-LOO: Leave-one-out alignment test ===
    W_loo = loo_alignment(W, X)
    loo_ps = []
    for i in range(n_dims):
        r_obs = pearsonr(X[:, i], W_loo[:, i]).statistic
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(n_objects)
            r_null = pearsonr(X[:, i], W_loo[perm, i]).statistic
            if np.abs(r_null) >= np.abs(r_obs):
                count += 1
        loo_ps.append((count + 1) / (n_perm + 1))

    # FDR correction
    rsa_reject = multipletests(rsa_ps, alpha=0.05, method="fdr_bh")[0]
    loo_reject = multipletests(loo_ps, alpha=0.05, method="fdr_bh")[0]

    return {
        "rsa_power": rsa_reject.mean(),
        "loo_power": loo_reject.mean(),
        "rsa_ps": rsa_ps,
        "loo_ps": loo_ps,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RSA vs SRF-LOO: Gaussian Tuning Function Simulation")
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
                n_objects, n_dims, n_neurons_per_dim, tuning_width, snr, n_perm,
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
    ax.set_title("Power comparison")
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

    fig.savefig(OUTPUT_DIR / "tuning_simulation.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUTPUT_DIR / 'tuning_simulation.png'}")


if __name__ == "__main__":
    main()
