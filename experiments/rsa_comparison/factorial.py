from __future__ import annotations

import itertools as it
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pysrf import SRF
from statsmodels.stats.multitest import multipletests

from tools.rsa import mantel_test, permutation_test
from utils.helpers import add_positive_noise_with_snr, align_latent_dimensions
from tools.rsa import compute_similarity


def create_factorial_data(
    levels: dict[str, list[str]], max_objects: int | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], list[tuple[str, ...]]]:
    items = list(it.product(*levels.values()))
    n = len(items)

    if max_objects is not None and n > max_objects:
        rng = np.random.default_rng(42)
        selected_indices = rng.choice(n, size=max_objects, replace=False)
        items = [items[i] for i in selected_indices]

    blocks: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        blocks[factor] = Z
        features.append(Z)

    X = np.concatenate(features, axis=1)
    return X, blocks, items


def run(cfg: DictConfig) -> None:
    """
    Atomic task: Run ONE factorial condition.
    Hydra handles sweeping over snr and seed.
    """
    # Convert levels from config
    levels = OmegaConf.to_container(cfg.levels, resolve=True)

    # Create factorial data (deterministic based on levels)
    X, blocks, _ = create_factorial_data(levels)
    rank = X.shape[1]

    # Create hypotheses (deterministic)
    hypotheses = [
        compute_similarity(Z, Z, metric=cfg.similarity_metric) for Z in blocks.values()
    ]

    # Create model
    model = SRF(rank=rank, random_state=0)

    # Evaluate ONE condition
    seed_base = int(cfg.snrs * 100 + cfg.seed)
    noisy_data = add_positive_noise_with_snr(X, cfg.snrs, rng=seed_base)

    measured_similarity = compute_similarity(
        noisy_data, noisy_data, metric=cfg.similarity_metric
    )
    w = model.fit_transform(measured_similarity)
    w_aligned = align_latent_dimensions(X, w)

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
            X[:, i],
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
