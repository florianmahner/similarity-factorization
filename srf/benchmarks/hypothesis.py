import numpy as np
import pandas as pd

from srf.helpers import load_spose_embedding
from srf.mixed.admm import ADMM
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from pathlib import Path
from srf.helpers import add_noise_with_snr, align_latent_dimensions
from tools.metrics import compute_similarity

METRIC = "linear"


def mantel_test(A, B, permutations=10000, random_state=None, two_sided=False):
    if random_state is not None:
        np.random.seed(random_state)

    idx_upper = np.triu_indices_from(A, k=1)
    sim1 = A[idx_upper]
    sim2 = B[idx_upper]
    obs = pearsonr(sim1, sim2).statistic
    if two_sided:
        obs = abs(obs)
    nulls = np.zeros(permutations)
    for i in range(permutations):
        perm = np.random.permutation(B.shape[0])
        Bp = B[perm][:, perm]
        sc = pearsonr(sim1, Bp[idx_upper]).statistic
        if two_sided:
            sc = abs(sc)
        nulls[i] = sc

    p = (np.sum(nulls >= obs) + 1) / (permutations + 1)
    return p, nulls, obs


def permutation_test(A, B, permutations=10000, random_state=None, two_sided=False):
    """Permutation test for the correlation between two vectors"""
    if random_state is not None:
        np.random.seed(random_state)

    observed_corr = pearsonr(A, B).statistic
    greater = 0
    null_corrs = np.zeros(permutations)
    for i in range(permutations):
        perm = np.random.permutation(B)

        # Permuted correlation of null distribution
        perm_corr = pearsonr(A, perm).statistic
        if two_sided:
            perm_corr = np.abs(perm_corr)

            # NOTE this is bad to modify the observed correlation, then i also return the absolute value
            observed_corr = np.abs(observed_corr)

        if perm_corr >= observed_corr:
            greater += 1

        null_corrs[i] = perm_corr

    # Empirical p-value, make slight correction to avoid p=0
    p_value = (greater + 1) / (permutations + 1)

    return p_value, null_corrs, observed_corr


def process_single_run(
    model, data, hypotheses, snr, repeat, n_permutations, alpha=0.05
):
    seed = int(snr * 100 + repeat)
    noisy_data = add_noise_with_snr(data, snr, rng=seed)

    measured_similarity = compute_similarity(noisy_data, noisy_data, METRIC)
    w = model.fit_transform(measured_similarity)
    denoised_similarity = model.s_hat_

    rank = data.shape[1]
    w_aligned = align_latent_dimensions(data, w)

    # Run RSA and NMF tests: Mantel between hypothesis and (observed/denoised)
    rsa_tests = [
        mantel_test(h, measured_similarity, permutations=n_permutations)
        for h in hypotheses
    ]
    nmf_tests = [
        mantel_test(h, denoised_similarity, permutations=n_permutations)
        for h in hypotheses
    ]

    latent_tests = [
        permutation_test(data[:, i], w_aligned[:, i], permutations=n_permutations)
        for i in range(rank)
    ]

    # Combine results into a dict for iteration
    test_results = {
        "RSA": rsa_tests,
        "NMF": nmf_tests,
        "Latent": latent_tests,
    }

    results = []
    for method, tests in test_results.items():
        raw_ps = [t[0] for t in tests]  # p_value from (p_value, null_corrs, obs_corr)
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]
        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            results.append(
                {
                    "SNR": snr,
                    "Repeat": repeat,
                    "Hypothesis": i + 1,
                    "Method": method,
                    "Raw_p": test[0],
                    "Corrected_p": corrected_p,
                    "Significant": bool(rej),
                    "Rec_error": model.history["rec_error"],
                    # You can optionally add null_corrs / obs_corr if needed:
                    # "Observed_corr": test[2],
                    # "Null_corr_mean": np.mean(test[1]),
                }
            )
    return results


# TODO replace by parseargs possibly.
class Config:
    snr_list = np.linspace(0, 0.3, 8)
    n_repeats = 100
    n_permutations = 500
    max_objects = 200
    MAX_JOBS = 100
    dims = [3, 5, 8, 12, 14]  # subs    ample of the spose dimensions


if __name__ == "__main__":
    cfg = Config()

    data = load_spose_embedding(max_objects=cfg.max_objects, max_dims=66)
    data = data[:, cfg.dims]
    cfg.rank = len(cfg.dims)
    hypotheses = []
    for i in range(cfg.rank):
        x_i = data[:, [i]]
        # NOTE here again check the kernel, ensure it is consistent throughout!!
        s_i = compute_similarity(x_i, x_i, METRIC)

        hypotheses.append(s_i)

    all_tasks = [(snr, rep) for snr in cfg.snr_list for rep in range(cfg.n_repeats)]

    model = ADMM(rank=cfg.rank, max_outer=10, w_inner=20, tol=1e-6, verbose=False)

    # test = process_single_run(
    #     model, data, hypotheses, snr=0.5, repeat=0, n_permutations=cfg.n_permutations
    # )

    print("Starting parallel processing...")
    results = Parallel(n_jobs=cfg.MAX_JOBS, verbose=10)(
        delayed(process_single_run)(
            model, data, hypotheses, snr, repeat, cfg.n_permutations
        )
        for snr, repeat in all_tasks
    )

    results_df = pd.DataFrame([item for sublist in results for item in sublist])

    path = Path("/LOCAL/fmahner/srf/results/benchmarks/hypothesis_tests.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
