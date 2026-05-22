"""Peterson-animals CV: pin down why argmin moves between 11 and 13.

In sandbox/pysrf_integration_test/spectrum_vs_cv (the run that worked) we
got argmin_rank=11 for Peterson-animals at max_outer=1000. In our new
dimensionality config we get argmin_rank=13. The two configs differ only in:
  - random_state for cross_val_score (sandbox=1000, dimensionality=0)
  - similarity matrix source (sandbox=cached .npy, dimensionality=build_similarity)

This script runs all four combinations on Peterson-animals and reports the
CV curve + argmin for each, so we can attribute the shift to one knob.

Run:
    ./scripts/submit sandbox/dimensionality/peterson_cv/run.py --bg
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from pysrf import cross_val_score, estimate_rank
from similarity import build_similarity
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()

CACHE_PATH = (
    PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings"
    / "outputs" / ".cache" / "sim_peterson-animals.npy"
)

K_CV = 5
N_REPEATS = 5
SRF_KWARGS = {"rho": 3.0, "max_inner": 30, "tol": 0.0, "max_outer": 1000}
RANKS_UPPER = 30


def load_cached() -> np.ndarray:
    return np.load(CACHE_PATH)


def load_built() -> np.ndarray:
    cfg_path = PROJECT_ROOT / "configs" / "dataset" / "peterson_animals.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": OmegaConf.load(PROJECT_ROOT / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(PROJECT_ROOT),
    })
    OmegaConf.resolve(parent)
    return build_similarity(parent.dataset)


def run_cv(label: str, s: np.ndarray, random_state: int) -> dict:
    n = s.shape[0]
    log.info(f"\n[{label}]  matrix shape={s.shape}, random_state={random_state}")
    est = estimate_rank(s)
    log.info(f"  k_cut={est.rank}  p*={est.sampling_fraction:.3f}")

    ranks = list(range(2, min(n - 1, RANKS_UPPER + 1)))
    t0 = time.time()
    curve = cross_val_score(
        s, ranks=ranks, sampling_fraction=est.sampling_fraction,
        n_folds=K_CV, n_repeats=N_REPEATS,
        random_state=random_state, srf_kwargs=SRF_KWARGS,
    )
    elapsed = time.time() - t0

    grouped = curve.groupby("rank")["val_mse"]
    mean = grouped.mean().reindex(ranks).to_numpy()
    sem = grouped.std().reindex(ranks).to_numpy() / np.sqrt(
        np.maximum(grouped.count().reindex(ranks).to_numpy(), 1)
    )
    argmin_rank = int(ranks[int(np.nanargmin(mean))])

    log.info(f"  argmin_rank={argmin_rank}  v-mse@min={float(np.nanmin(mean)):.4e}  "
             f"({elapsed:.1f}s)")
    return {
        "label": label,
        "k_cut": int(est.rank),
        "p_star": float(est.sampling_fraction),
        "argmin_rank": argmin_rank,
        "ranks": [int(r) for r in ranks],
        "mean": [float(v) for v in mean],
        "sem": [float(v) for v in sem],
        "random_state": int(random_state),
        "runtime_sec": round(elapsed, 1),
    }


def main() -> None:
    log.info("=" * 70)
    log.info("Peterson-animals CV: matrix source x random_state ablation")
    log.info("=" * 70)

    log.info("Loading matrices...")
    s_cached = load_cached()
    s_built = load_built()
    log.info(f"cached: shape={s_cached.shape}, "
             f"finite={np.all(np.isfinite(s_cached))}")
    log.info(f"built : shape={s_built.shape}, "
             f"finite={np.all(np.isfinite(s_built))}")
    if s_cached.shape == s_built.shape:
        diff = np.linalg.norm(s_cached - s_built) / np.linalg.norm(s_cached)
        log.info(f"||cached - built||_F / ||cached||_F = {diff:.6e}")

    variants = [
        ("cached_rs1000", s_cached, 1000),  # closest to the working sandbox
        ("cached_rs0",    s_cached, 0),
        ("built_rs1000",  s_built,  1000),
        ("built_rs0",     s_built,  0),     # current dimensionality config
    ]

    results = []
    for label, s, rs in variants:
        results.append(run_cv(label, s, rs))

    log.info("\n" + "=" * 70)
    log.info("Summary  (sandbox baseline argmin_rank = 11)")
    log.info("=" * 70)
    log.info(f"{'variant':<16}  {'k_cut':>5}  {'argmin':>6}  {'p*':>6}  {'runtime':>7}")
    for r in results:
        log.info(f"{r['label']:<16}  {r['k_cut']:>5}  {r['argmin_rank']:>6}  "
                 f"{r['p_star']:>6.3f}  {r['runtime_sec']:>5.1f}s")

    log.info("\nMean V-MSE around the changepoint (ranks 8..15):")
    log.info(f"{'rank':>4}  " + "  ".join(f"{r['label']:>14}" for r in results))
    ranks = results[0]["ranks"]
    for r in [8, 9, 10, 11, 12, 13, 14, 15]:
        if r in ranks:
            i = ranks.index(r)
            cells = "  ".join(f"{res['mean'][i]:>14.5f}" for res in results)
            log.info(f"{r:>4}  {cells}")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    log.info(f"\nSaved {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
