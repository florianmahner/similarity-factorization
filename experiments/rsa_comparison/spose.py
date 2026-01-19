"""SPOSE dimension recovery: RSA vs SRF comparison.

Compares two approaches for testing dimension recovery:
1. RSA: Mantel test (hypothesis RSM vs measured similarity)
2. SRF-LOO: Leave-one-out alignment test (prevents overfitting)

Usage:
    ./scripts/submit experiments/rsa_comparison/spose.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from statsmodels.stats.multitest import multipletests

from pysrf import SRF

from tools.rsa import mantel_test, loo_alignment_test_multi
from src.utils.helpers import add_positive_noise_with_snr
from utils.io import load_spose_embedding


def run(cfg: DictConfig) -> None:
    """Run one SPOSE condition (Hydra sweeps over n_objects, snr, seed)."""
    # Load full data (66 dimensions)
    full_data = load_spose_embedding(num_dims=66)

    # Randomize dimensions and objects per repeat (avoids cherry-picking)
    rng = np.random.default_rng(cfg.seed)
    dims = rng.choice(full_data.shape[1], size=cfg.num_dims, replace=False)
    selected_objects = rng.choice(full_data.shape[0], size=cfg.n_objects, replace=False)

    # Extract data for this condition
    data = full_data[selected_objects][:, dims]
    rank = len(dims)

    # Add noise and compute similarity
    seed_base = 10000 + 97 * (cfg.seed + 1)
    noisy_data = add_positive_noise_with_snr(data, cfg.snrs, rng=seed_base)
    measured_similarity = noisy_data @ noisy_data.T

    # Fit SRF
    model = SRF(rank=rank, verbose=False, tol=0.0, random_state=cfg.seed)
    W = model.fit_transform(measured_similarity)

    # === RSA: Mantel test per dimension ===
    rsa_raw_ps = []
    rsa_r_obs = []
    for i in range(rank):
        x_i = data[:, [i]]
        hypothesis_rsm = x_i @ x_i.T
        p, _, r = mantel_test(
            hypothesis_rsm,
            measured_similarity,
            permutations=cfg.n_permutations,
            random_state=seed_base + 100 * i,
            two_sided=True,
        )
        rsa_raw_ps.append(p)
        rsa_r_obs.append(r)

    # FDR correction for RSA
    rsa_reject, rsa_corrected_ps = multipletests(
        rsa_raw_ps, cfg.alpha, method="fdr_bh"
    )[:2]

    # === SRF-LOO: Leave-one-out alignment test ===
    srf_results = loo_alignment_test_multi(
        W,
        data,
        permutations=cfg.n_permutations,
        alpha=cfg.alpha,
        random_state=seed_base + 200,
    )

    # Collect results
    rows = []
    for i in range(rank):
        # RSA result
        rows.append({
            "n_objects": int(cfg.n_objects),
            "snr": float(cfg.snrs),
            "repeat": int(cfg.seed),
            "dimension": i + 1,
            "method": "RSA",
            "r_obs": float(rsa_r_obs[i]),
            "raw_p": float(rsa_raw_ps[i]),
            "corrected_p": float(rsa_corrected_ps[i]),
            "significant": bool(rsa_reject[i]),
        })
        # SRF-LOO result
        rows.append({
            "n_objects": int(cfg.n_objects),
            "snr": float(cfg.snrs),
            "repeat": int(cfg.seed),
            "dimension": i + 1,
            "method": "SRF-LOO",
            "r_obs": float(srf_results["r_obs"][i]),
            "raw_p": float(srf_results["raw_p"][i]),
            "corrected_p": float(srf_results["corrected_p"][i]),
            "significant": bool(srf_results["significant"][i]),
        })

    # Save results
    output_file = Path.cwd() / "results.json"
    with open(output_file, "w") as f:
        json.dump(rows, f)
