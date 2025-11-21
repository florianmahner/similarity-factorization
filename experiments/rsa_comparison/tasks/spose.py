from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from statsmodels.stats.multitest import multipletests

from pysrf import SRF

from tools.rsa import mantel_test, permutation_test
from utils.helpers import add_positive_noise_with_snr, align_latent_dimensions
from utils.io import load_spose_embedding


def run(cfg: DictConfig) -> None:
    """
    Atomic task: Run ONE SPoSE condition.
    Hydra handles sweeping over n_objects, snr, and seed.
    """
    # Load full data
    full_data = load_spose_embedding(num_dims=49)

    # Select dimensions deterministically (same across all jobs)
    rng = np.random.default_rng(0)
    dims = rng.choice(full_data.shape[1], size=cfg.num_dims, replace=False)

    # Select objects deterministically (same for same n_objects across all jobs)
    selected_objects = rng.choice(full_data.shape[0], size=cfg.n_objects, replace=False)

    # Extract data for this condition
    data = full_data[selected_objects][:, dims]

    # Create hypotheses (one per dimension)
    hypotheses = []
    rank = len(dims)
    for i in range(rank):
        x_i = data[:, [i]]
        hypotheses.append(x_i @ x_i.T)

    # Create model
    model = SRF(rank=rank, verbose=False, tol=0.0, random_state=cfg.seed)

    # Evaluate condition
    seed_base = 10000 + 97 * (cfg.seed + 1)
    noisy_data = add_positive_noise_with_snr(data, cfg.snrs, rng=seed_base)

    measured_similarity = noisy_data @ noisy_data.T
    w = model.fit_transform(measured_similarity)
    w_aligned = align_latent_dimensions(data, w)

    # RSA tests
    rsa_tests = [
        mantel_test(
            h,
            measured_similarity,
            permutations=cfg.n_permutations,
            random_state=seed_base + 100 * i,
            two_sided=True,
        )
        for i, h in enumerate(hypotheses)
    ]

    # Latent tests
    latent_tests = [
        permutation_test(
            data[:, i],
            w_aligned[:, i],
            permutations=cfg.n_permutations,
            random_state=seed_base + 200 * i,
            two_sided=True,
        )
        for i in range(rank)
    ]

    # Process results
    rows = []
    for method, tests in {"RSA": rsa_tests, "SRF": latent_tests}.items():
        raw_ps = [t[0] for t in tests]
        reject, corr_ps = multipletests(raw_ps, cfg.alpha, method="fdr_bh")[:2]
        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            rows.append(
                {
                    "n_objects": int(cfg.n_objects),
                    "snr": float(cfg.snrs),
                    "repeat": int(cfg.seed),
                    "hypothesis": i + 1,
                    "method": method,
                    "raw_p": float(test[0]),
                    "corrected_p": float(corrected_p),
                    "significant": bool(rej),
                }
            )

    # Save results as JSON
    output_file = Path.cwd() / "results.json"
    with open(output_file, "w") as f:
        json.dump(rows, f)
