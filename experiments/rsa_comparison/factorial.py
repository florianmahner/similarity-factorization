"""Factorial design: RSA vs SRF alignment comparison."""

from __future__ import annotations

import itertools as it
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from pysrf import SRF

from src.tools.rsa import rsa_test, alignment_test
from src.utils.helpers import add_positive_noise_with_snr

log = logging.getLogger(__name__)


def _create_factorial(
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    items = list(it.product(*levels.values()))
    features, col_ranges = [], {}
    col = 0
    for i, (factor, lvls) in enumerate(levels.items()):
        idx = [lvls.index(item[i]) for item in items]
        features.append(np.eye(len(lvls))[idx])
        col_ranges[factor] = (col, col + len(lvls))
        col += len(lvls)
    return np.hstack(features), col_ranges


def _get_factor(col: int, col_ranges: dict) -> str:
    for factor, (start, end) in col_ranges.items():
        if start <= col < end:
            return factor
    return "unknown"


def _build_rows(
    col_ranges: dict, rsa: dict, srf_loo: dict, srf_global: dict, snr: float, seed: int
) -> list[dict]:
    rows = []
    k = len(rsa["r_obs"])
    methods = [("RSA", rsa), ("SRF-LOO", srf_loo), ("SRF-Global", srf_global)]
    for col in range(k):
        base = {
            "snr": snr,
            "repeat": seed,
            "factor": _get_factor(col, col_ranges),
            "column": col,
        }
        for method, res in methods:
            rows.append(
                {
                    **base,
                    "method": method,
                    "r_obs": res["r_obs"][col],
                    "raw_p": res["raw_p"][col],
                    "significant": res["significant"][col],
                }
            )
    return rows


def _add_noise_to_similarity(s: np.ndarray, snr: float, rng) -> np.ndarray:
    """Add noise directly to similarity matrix (measurement noise model)."""
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    if snr == 0:
        # Pure noise
        noise = rng.random(s.shape)
        noise = (noise + noise.T) / 2  # Symmetric
        return noise
    # SNR = signal_var / noise_var -> noise_var = signal_var / snr
    signal_var = np.var(s)
    noise_std = np.sqrt(signal_var / snr) if snr > 0 else 1.0
    noise = rng.normal(0, noise_std, s.shape)
    noise = (noise + noise.T) / 2  # Symmetric
    s_noisy = s + noise
    # Keep non-negative
    return np.maximum(s_noisy, 0)


def _run_single(
    x: np.ndarray, col_ranges: dict, snr: float, n_perm: int, seed: int, alpha: float,
    noise_model: str = "features"
) -> list[dict]:
    seed_base = 10000 + 97 * (seed + 1)

    if noise_model == "features":
        x_noisy = add_positive_noise_with_snr(x, snr, rng=seed_base)
        s = x_noisy @ x_noisy.T
    else:  # noise_model == "similarity"
        s_clean = x @ x.T
        s = _add_noise_to_similarity(s_clean, snr, rng=seed_base)

    w = SRF(rank=x.shape[1], verbose=False, tol=0.0, random_state=seed).fit_transform(s)

    rsa = rsa_test(
        x, s, permutations=n_perm, alpha=alpha, fdr=True, random_state=seed_base
    )
    srf_loo = alignment_test(
        w,
        x,
        alignment="loo",
        permutations=n_perm,
        alpha=alpha,
        fdr=True,
        random_state=seed_base + 1000,
    )
    srf_global = alignment_test(
        w,
        x,
        alignment="global",
        permutations=n_perm,
        alpha=alpha,
        fdr=True,
        random_state=seed_base + 2000,
    )

    return _build_rows(col_ranges, rsa, srf_loo, srf_global, snr, seed)


def run(cfg: DictConfig) -> None:
    levels = OmegaConf.to_container(cfg.levels, resolve=True)
    x, col_ranges = _create_factorial(levels)
    noise_model = cfg.get("noise_model", "features")
    log.info(
        f"Factorial: {x.shape[0]} items, {x.shape[1]} columns, {len(levels)} factors, noise={noise_model}"
    )

    alpha = cfg.get("alpha", 0.05)
    rows = []
    for snr in list(cfg.snrs):
        log.info(f"SNR={snr:.2f}")
        results = Parallel(n_jobs=cfg.get("n_jobs", -1))(
            delayed(_run_single)(x, col_ranges, snr, cfg.n_permutations, seed, alpha, noise_model)
            for seed in range(cfg.n_repeats)
        )
        for r in results:
            rows.extend(r)

    pd.DataFrame(rows).to_csv(Path.cwd() / "factorial.csv", index=False)
