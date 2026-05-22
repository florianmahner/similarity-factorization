"""Investigate FPR calibration under pure noise (SNR=0)."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.tools.rsa import alignment_test, rsa_test
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def generate_pure_noise(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent noise for x and s."""
    rng = np.random.default_rng(seed)

    # Random features (no structure)
    x = rng.random((n, k))

    # Random similarity matrix (symmetric, non-negative)
    noise = rng.random((n, n))
    s = (noise + noise.T) / 2
    np.fill_diagonal(s, 1.0)

    return x, s


def run_single(n: int, k: int, n_perm: int, two_sided: bool, seed: int) -> dict:
    """Run one replicate and return significance flags."""
    x, s = generate_pure_noise(n, k, seed)

    # Fit SRF on noise
    w = SRF(rank=k, verbose=False, tol=0.0, random_state=seed).fit_transform(s)

    # Test alignment
    srf_loo = alignment_test(
        w, x, alignment="loo", permutations=n_perm, alpha=0.05,
        fdr=True, two_sided=two_sided, random_state=seed
    )
    srf_global = alignment_test(
        w, x, alignment="global", permutations=n_perm, alpha=0.05,
        fdr=True, two_sided=two_sided, random_state=seed
    )
    rsa = rsa_test(
        x, s, permutations=n_perm, alpha=0.05, fdr=True,
        two_sided=two_sided, random_state=seed
    )

    return {
        "srf_loo_sig": srf_loo["significant"],
        "srf_loo_raw_p": srf_loo["raw_p"],
        "srf_global_sig": srf_global["significant"],
        "srf_global_raw_p": srf_global["raw_p"],
        "rsa_sig": rsa["significant"],
        "rsa_raw_p": rsa["raw_p"],
    }


def compute_fpr(results: list[dict], key: str) -> float:
    """Compute false positive rate across all columns and repeats."""
    all_sig = np.concatenate([r[key] for r in results])
    return np.mean(all_sig)


def compute_raw_p_distribution(results: list[dict], key: str) -> np.ndarray:
    """Get distribution of raw p-values."""
    return np.concatenate([r[key] for r in results])


def main():
    n = 50
    n_repeats = 200
    n_perm = 500

    print(f"Config: n={n}, n_repeats={n_repeats}, n_perm={n_perm}")
    print(f"Expected FPR at alpha=0.05: 0.05")
    print("=" * 60)

    # Test different k values to see FDR behavior
    for k in [5, 20, 50]:
        print(f"\n{'='*60}")
        print(f"k={k} columns")
        print("=" * 60)

        results = Parallel(n_jobs=-1)(
            delayed(run_single)(n, k, n_perm, True, seed)  # two_sided=True
            for seed in range(n_repeats)
        )

        # FPR after FDR correction
        fpr_loo = compute_fpr(results, "srf_loo_sig")
        fpr_global = compute_fpr(results, "srf_global_sig")
        fpr_rsa = compute_fpr(results, "rsa_sig")

        print(f"  FPR (after FDR):")
        print(f"    SRF-LOO:    {fpr_loo:.4f}")
        print(f"    SRF-Global: {fpr_global:.4f}")
        print(f"    RSA:        {fpr_rsa:.4f}")

        # Check raw p-value distribution (should be uniform under null)
        raw_p_loo = compute_raw_p_distribution(results, "srf_loo_raw_p")
        raw_p_global = compute_raw_p_distribution(results, "srf_global_raw_p")
        raw_p_rsa = compute_raw_p_distribution(results, "rsa_raw_p")

        # FPR from raw p-values (before FDR)
        fpr_raw_loo = np.mean(raw_p_loo < 0.05)
        fpr_raw_global = np.mean(raw_p_global < 0.05)
        fpr_raw_rsa = np.mean(raw_p_rsa < 0.05)

        print(f"  FPR (raw, no FDR):")
        print(f"    SRF-LOO:    {fpr_raw_loo:.4f}")
        print(f"    SRF-Global: {fpr_raw_global:.4f}")
        print(f"    RSA:        {fpr_raw_rsa:.4f}")

        # Check if p-values are uniform (KS test)
        from scipy.stats import kstest
        ks_loo = kstest(raw_p_loo, 'uniform')
        ks_global = kstest(raw_p_global, 'uniform')
        ks_rsa = kstest(raw_p_rsa, 'uniform')

        print(f"  KS test for uniform p-values:")
        print(f"    SRF-LOO:    stat={ks_loo.statistic:.4f}, p={ks_loo.pvalue:.4f}")
        print(f"    SRF-Global: stat={ks_global.statistic:.4f}, p={ks_global.pvalue:.4f}")
        print(f"    RSA:        stat={ks_rsa.statistic:.4f}, p={ks_rsa.pvalue:.4f}")


if __name__ == "__main__":
    main()
