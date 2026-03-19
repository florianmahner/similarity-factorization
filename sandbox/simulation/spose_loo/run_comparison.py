"""Direct comparison: exact same logic as experiments/rsa_comparison/spose.py.

This script replicates the EXACT logic from the main experiment to identify
any differences with the sandbox implementation.
"""

import numpy as np
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from pysrf import SRF
from src.utils import get_output_dir
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.figure_theme import create_figure, despine
from src.tools.rsa import mantel_test, loo_alignment_test_multi
from utils.io import load_spose_embedding

OUTPUT_DIR = get_output_dir()


def run_single_rep_main_style(
    full_data: np.ndarray,
    n_objects: int,
    num_dims: int,
    snr: float,
    n_permutations: int,
    seed: int,
) -> dict:
    """Run one SPOSE condition - EXACT same logic as main experiment."""
    # === EXACT COPY from experiments/rsa_comparison/spose.py ===

    # Randomize dimensions and objects per repeat (avoids cherry-picking)
    rng = np.random.default_rng(seed)
    dims = rng.choice(full_data.shape[1], size=num_dims, replace=False)
    selected_objects = rng.choice(full_data.shape[0], size=n_objects, replace=False)

    # Extract data for this condition
    data = full_data[selected_objects][:, dims]
    rank = len(dims)

    # Add noise and compute similarity
    seed_base = 10000 + 97 * (seed + 1)  # MAIN EXPERIMENT FORMULA
    noisy_data = add_positive_noise_with_snr(data, snr, rng=seed_base)
    measured_similarity = noisy_data @ noisy_data.T

    # Fit SRF
    model = SRF(rank=rank, verbose=False, tol=0.0, random_state=seed)
    W = model.fit_transform(measured_similarity)

    # === RSA: Mantel test per dimension ===
    rsa_raw_ps = []
    for i in range(rank):
        x_i = data[:, [i]]
        hypothesis_rsm = x_i @ x_i.T
        p, _, _ = mantel_test(
            hypothesis_rsm,
            measured_similarity,
            permutations=n_permutations,
            random_state=seed_base + 100 * i,  # MAIN EXPERIMENT FORMULA
            two_sided=True,
        )
        rsa_raw_ps.append(p)

    # FDR correction for RSA
    rsa_reject = multipletests(rsa_raw_ps, alpha=0.05, method="fdr_bh")[0]

    # === SRF-LOO: Leave-one-out alignment test ===
    srf_results = loo_alignment_test_multi(
        W,
        data,
        permutations=n_permutations,
        alpha=0.05,
        random_state=seed_base + 200,  # MAIN EXPERIMENT FORMULA
    )

    return {
        "rsa_power": rsa_reject.mean(),
        "loo_power": srf_results["significant"].mean(),
        "rsa_ps": rsa_raw_ps,
        "loo_ps": srf_results["raw_p"].tolist(),
    }


def run_single_rep_sandbox_style(
    full_data: np.ndarray,
    n_objects: int,
    n_dims: int,
    snr: float,
    n_perm: int,
    seed: int,
) -> dict:
    """Run one SPOSE condition - SANDBOX style (for comparison)."""
    rng = np.random.default_rng(seed)

    # Randomly select dimensions and objects
    dims = rng.choice(full_data.shape[1], size=n_dims, replace=False)
    objects = rng.choice(full_data.shape[0], size=n_objects, replace=False)

    # Extract data for this condition
    data = full_data[objects][:, dims]

    # Add noise and compute similarity
    noisy_data = add_positive_noise_with_snr(data, snr, rng=seed)  # SANDBOX: uses seed directly
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

    # === SRF-LOO: Fixed implementation ===
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
    print("Direct Comparison: Main Experiment vs Sandbox Style")
    print("=" * 60)

    full_data = load_spose_embedding(num_dims=66)
    print(f"Loaded SPOSE data: {full_data.shape}")

    # Match main experiment config EXACTLY
    n_reps = 100
    n_objects = 50
    num_dims = 5  # Main experiment had 5 dims based on CSV
    n_permutations = 1000
    snr_levels = np.array([0.0, 0.5, 1.0])  # Test key levels

    print(f"\nParameters: {n_objects} objects, {num_dims} dims, {n_reps} reps, {n_permutations} perms")

    for style_name, run_func in [("MAIN_STYLE", run_single_rep_main_style), ("SANDBOX_STYLE", run_single_rep_sandbox_style)]:
        print(f"\n{'='*60}")
        print(f"Running: {style_name}")
        print("=" * 60)

        results = {"snr": snr_levels, "rsa_power": [], "loo_power": []}

        for snr in snr_levels:
            print(f"  SNR={snr:.2f}...")

            if style_name == "MAIN_STYLE":
                # Main experiment uses seed = 0, 1, 2, ... for repeats
                rep_results = Parallel(n_jobs=-1)(
                    delayed(run_func)(full_data, n_objects, num_dims, snr, n_permutations, seed=rep)
                    for rep in range(n_reps)
                )
            else:
                # Sandbox uses seed = rep * 1000 + int(snr * 100)
                rep_results = Parallel(n_jobs=-1)(
                    delayed(run_func)(full_data, n_objects, num_dims, snr, n_permutations, seed=rep * 1000 + int(snr * 100))
                    for rep in range(n_reps)
                )

            results["rsa_power"].append(np.mean([r["rsa_power"] for r in rep_results]))
            results["loo_power"].append(np.mean([r["loo_power"] for r in rep_results]))

        print(f"\n{style_name} Results:")
        print(f"{'SNR':<8} {'RSA':<12} {'SRF-LOO':<12}")
        print("-" * 32)
        for i, snr in enumerate(snr_levels):
            print(f"{snr:<8.2f} {results['rsa_power'][i]*100:<12.1f} {results['loo_power'][i]*100:<12.1f}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
