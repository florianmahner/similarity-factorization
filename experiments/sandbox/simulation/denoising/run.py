"""Test denoising properties of similarity-space vs feature-space factorization.

Hypothesis: SRF operating on similarity matrices has implicit denoising properties
because similarity entries aggregate across features.

Experiment:
1. Generate ground-truth: W_true (n×k), F (k×p), X_true = W_true @ F
2. Compute S_true = W_true @ W_true.T (ground truth similarity from embedding)
3. Add noise to X → X_noisy
4. Compare:
   - NMF on X_noisy → W_nmf → S_nmf = W_nmf @ W_nmf.T
   - SRF on S_noisy (= X_noisy @ X_noisy.T) → W_srf → S_srf = W_srf @ W_srf.T
5. Measure: correlation of recovered S with S_true

SNR convention:
- SNR = 1.0: no noise (perfect signal)
- SNR = 0.5: signal variance = noise variance
- SNR = 0.0: all noise

At SNR=1.0, both methods should recover S_true perfectly.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import NMF
from joblib import Parallel, delayed
import pandas as pd

from src.utils import get_output_dir, create_figure, despine
from src.colors import ROSE, TEAL, GRAY
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import simulation_dirichlet
from pysrf import SRF

OUTPUT_DIR = get_output_dir()


def generate_data(n: int, k: int, p: int, alpha: float, rng: np.random.Generator):
    """Generate ground truth embedding, features, and data matrix."""
    # W_true: soft cluster memberships (n × k)
    W_true = simulation_dirichlet(n, k, alpha=alpha, rng=rng, sort_by_cluster_size=False)

    # F: feature loadings (k × p)
    F = rng.exponential(scale=1.0, size=(k, p))

    # X_true: data matrix (n × p)
    X_true = W_true @ F

    # S_true: ground truth similarity computed from clean features
    # This is what we'd observe with no noise
    S_true = X_true @ X_true.T

    return W_true, F, X_true, S_true


def compute_similarity_correlation(S_true: np.ndarray, S_rec: np.ndarray) -> float:
    """Compute correlation between upper triangles of two similarity matrices."""
    n = S_true.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    s_true_vec = S_true[triu_idx]
    s_rec_vec = S_rec[triu_idx]

    if s_true_vec.std() > 0 and s_rec_vec.std() > 0:
        return pearsonr(s_true_vec, s_rec_vec)[0]
    return 0.0


def run_single(n: int, k: int, p: int, snr: float, alpha: float, seed: int) -> dict:
    """Run single comparison."""
    rng = np.random.default_rng(seed)

    # Generate ground truth
    W_true, F, X_true, S_true = generate_data(n, k, p, alpha, rng)

    # Add noise to data matrix X
    X_noisy = add_positive_noise_with_snr(X_true, ratio=snr, rng=rng)

    # Compute noisy similarity (what SRF sees)
    S_noisy = X_noisy @ X_noisy.T

    # Method 1: NMF on noisy data X_noisy
    nmf = NMF(n_components=k, max_iter=1000, random_state=seed)
    try:
        W_nmf = nmf.fit_transform(X_noisy)
        H_nmf = nmf.components_
        X_nmf_reconstructed = W_nmf @ H_nmf
    except Exception:
        X_nmf_reconstructed = np.zeros_like(X_noisy)
    S_nmf = X_nmf_reconstructed @ X_nmf_reconstructed.T

    # Method 2: SRF on noisy similarity S_noisy
    srf = SRF(rank=k, max_outer=200, random_state=seed, verbose=0)
    try:
        srf.fit(S_noisy)
        W_srf = srf.components_
    except Exception:
        W_srf = np.zeros((n, k))
    S_srf = W_srf @ W_srf.T

    # Compute correlations with ground truth S
    corr_nmf = compute_similarity_correlation(S_true, S_nmf)
    corr_srf = compute_similarity_correlation(S_true, S_srf)
    corr_noisy = compute_similarity_correlation(S_true, S_noisy)

    return {
        'snr': snr,
        'alpha': alpha,
        'seed': seed,
        'corr_input': corr_noisy,
        'corr_nmf': corr_nmf,
        'corr_srf': corr_srf,
    }


def main():
    print(f"Output: {OUTPUT_DIR}")

    # Parameters
    n = 200   # items
    k = 10    # latent dimensions
    p = 50    # observed features
    alpha = 1.0  # Dirichlet concentration
    snr_values = np.linspace(0.1, 1.0, 10)
    n_seeds = 20

    print(f"Running denoising comparison: n={n}, k={k}, p={p}, alpha={alpha}")
    print(f"SNR values: {snr_values}")
    print(f"Seeds per condition: {n_seeds}")

    # Run experiments
    tasks = [
        (n, k, p, snr, alpha, seed)
        for snr in snr_values
        for seed in range(n_seeds)
    ]

    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_single)(*t) for t in tasks
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "denoising_comparison.csv", index=False)

    # Aggregate and plot
    agg = df.groupby('snr').agg(['mean', 'std']).reset_index()

    fig, ax = create_figure("single")
    snr = agg['snr']

    # Input (S_noisy)
    input_mean = agg[('corr_input', 'mean')]
    ax.plot(snr, input_mean, '--', color=GRAY, label='Input (S_noisy)', linewidth=1.5)

    # NMF (feature-space)
    nmf_mean = agg[('corr_nmf', 'mean')]
    nmf_std = agg[('corr_nmf', 'std')]
    ax.fill_between(snr, nmf_mean - nmf_std, nmf_mean + nmf_std, alpha=0.2, color=ROSE)
    ax.plot(snr, nmf_mean, 's-', color=ROSE, label='NMF (features)', markersize=4)

    # SRF (similarity-space)
    srf_mean = agg[('corr_srf', 'mean')]
    srf_std = agg[('corr_srf', 'std')]
    ax.fill_between(snr, srf_mean - srf_std, srf_mean + srf_std, alpha=0.2, color=TEAL)
    ax.plot(snr, srf_mean, 'o-', color=TEAL, label='SRF (similarity)', markersize=4)

    ax.set_xlabel('SNR (1 = no noise)')
    ax.set_ylabel('Correlation with true S')
    ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.05, 1.05)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "denoising_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')

    # Print summary
    print("\n=== Summary ===")
    for snr_val in [0.1, 0.5, 1.0]:
        row = agg[np.isclose(agg['snr'], snr_val, atol=0.05)]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"\nSNR={snr_val:.1f}:")
            print(f"  Input corr: {row[('corr_input', 'mean')]:.3f}")
            print(f"  NMF corr:   {row[('corr_nmf', 'mean')]:.3f} ± {row[('corr_nmf', 'std')]:.3f}")
            print(f"  SRF corr:   {row[('corr_srf', 'mean')]:.3f} ± {row[('corr_srf', 'std')]:.3f}")


if __name__ == "__main__":
    main()
