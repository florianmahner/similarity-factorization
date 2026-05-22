"""SPOSE dimension recovery: RSA vs SRF-LOO comparison."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF

from src.tools.rsa import rsa_test, alignment_test
from src.utils.helpers import add_positive_noise_with_snr
from utils.io import load_spose_embedding

log = logging.getLogger(__name__)


def _sample_subset(data: np.ndarray, n_obj: int, n_dim: int, rng) -> np.ndarray:
    dims = rng.choice(data.shape[1], size=n_dim, replace=False)
    objs = rng.choice(data.shape[0], size=n_obj, replace=False)
    return data[objs][:, dims]


def _build_rows(x: np.ndarray, rsa: dict, srf: dict, snr: float, seed: int) -> list[dict]:
    rows = []
    for i in range(x.shape[1]):
        base = {"snr": snr, "repeat": seed, "dim": i + 1, "dim_var": float(np.var(x[:, i]))}
        for method, res in [("RSA", rsa), ("SRF-LOO", srf)]:
            rows.append({
                **base,
                "method": method,
                "r_obs": res["r_obs"][i],
                "raw_p": res["raw_p"][i],
                "significant": res["significant"][i],
            })
    return rows


def _run_single(data: np.ndarray, n_obj: int, n_dim: int, snr: float, n_perm: int, seed: int, alpha: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = _sample_subset(data, n_obj, n_dim, rng)

    seed_base = 10000 + 97 * (seed + 1)
    x_noisy = add_positive_noise_with_snr(x, snr, rng=seed_base)
    s = x_noisy @ x_noisy.T

    w = SRF(rank=n_dim, verbose=False, tol=0.0, random_state=seed).fit_transform(s)

    rsa = rsa_test(x, s, permutations=n_perm, alpha=alpha, fdr=True, random_state=seed_base)
    srf = alignment_test(w, x, alignment="loo", permutations=n_perm, alpha=alpha, fdr=True, random_state=seed_base + 1000)

    return _build_rows(x, rsa, srf, snr, seed)


def run(cfg: DictConfig) -> None:
    data = load_spose_embedding(num_dims=66)
    log.info(f"SPOSE: n={cfg.n_objects}, k={cfg.num_dims}, repeats={cfg.n_repeats}")

    alpha = cfg.get("alpha", 0.05)
    snrs = list(cfg.snrs)
    seeds = list(range(cfg.n_repeats))
    conditions = [(snr, seed) for snr in snrs for seed in seeds]
    log.info(f"{len(conditions)} total jobs ({len(snrs)} SNRs x {len(seeds)} repeats)")

    results = Parallel(n_jobs=cfg.get("n_jobs", -1), verbose=10)(
        delayed(_run_single)(data, cfg.n_objects, cfg.num_dims, snr, cfg.n_permutations, seed, alpha)
        for snr, seed in conditions
    )

    rows = []
    for r in results:
        rows.extend(r)
    pd.DataFrame(rows).to_csv(Path.cwd() / "spose.csv", index=False)
