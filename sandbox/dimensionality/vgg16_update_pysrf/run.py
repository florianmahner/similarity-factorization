"""Sanity check: run vgg16 through update_pysrf (the reference pipeline).

If update_pysrf's spectral_pass + recipe_K + 5-fold CV also gives CV
argmin >> k_cut, then the divergence is a real property of vgg16's
spectrum, not a bug in our pysrf implementation.

Steps:
  1. load vgg16 similarity (our pipeline)
  2. update_pysrf.spectral_pass -> k_cut, p_star_raw
  3. update_pysrf.recipe_K -> calibrated p_star, p_cv
  4. 5-fold CV at p_cv using update_pysrf.ADMM, ranks [30, 45, 60, 90]
  5. compare argmin vs k_cut

Run:
    poetry run python sandbox/dimensionality/vgg16_update_pysrf/run.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPDATE_PYSRF_SRC = PROJECT_ROOT / "update_pysrf" / "src"
sys.path.insert(0, str(UPDATE_PYSRF_SRC))

from _common import spectral_pass, recipe_K, split_omega_into_folds  # noqa: E402
from symmnmf.cross_validation import mask_missing_entries  # noqa: E402
from symmnmf.models.admm import ADMM  # noqa: E402

from similarity import build_similarity  # noqa: E402
from src.utils import get_output_dir  # noqa: E402

OUTPUT_DIR = get_output_dir()
RANKS = [30, 45, 60, 90]
N_FOLDS = 5
N_REPS = 1
SEED = 0
MAX_OUTER = 50
MAX_INNER = 30
N_JOBS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_vgg16() -> np.ndarray:
    cfg_path = PROJECT_ROOT / "configs" / "dataset" / "vgg16.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": OmegaConf.load(PROJECT_ROOT / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(PROJECT_ROOT),
    })
    OmegaConf.resolve(parent)
    return build_similarity(parent.dataset)


def fit_score_admm(s, holdout_mask, val_mask, rank, bounds, seed,
                   max_outer, max_inner):
    x = s.copy()
    x[holdout_mask] = np.nan
    est = ADMM(
        rank=rank, missing_values=np.nan, bounds=bounds,
        random_state=seed, max_outer=max_outer, max_inner=max_inner,
        verbose=1,
    )
    est.fit(x)
    rec = est.reconstruct()
    iu = np.triu_indices(s.shape[0], k=1)
    m = val_mask[iu] & np.isfinite(s[iu]) & np.isfinite(rec[iu])
    if not m.any():
        return float("nan")
    return float(np.mean((s[iu][m] - rec[iu][m]) ** 2))


def main() -> None:
    log("=" * 70)
    log("vgg16 reference-pipeline sanity check (update_pysrf)")
    log("=" * 70)

    log("Loading vgg16 similarity matrix...")
    s = load_vgg16()
    s = np.asarray(s, dtype=np.float64)
    log(f"  shape={s.shape}")
    bounds = (float(np.nanmin(s)), float(np.nanmax(s)))

    log("Running update_pysrf.spectral_pass (B=20)...")
    t0 = time.time()
    sp = spectral_pass(s, B=20, smooth_window=10, show_progress=False)
    log(f"  spectral_pass done in {time.time() - t0:.1f}s")
    log(f"  k_cut={int(sp['k_cut'])}  "
        f"k_smooth_legacy={int(sp.get('k_smooth_legacy', -1))}")

    log("Running update_pysrf.recipe_K (delta=0.10, k_cv=5)...")
    t0 = time.time()
    rk = recipe_K(sp, delta=0.10, k_cv=N_FOLDS, p_floor=0.5,
                  p_floor_mode="adaptive")
    log(f"  recipe_K done in {time.time() - t0:.1f}s")
    k_cut = int(rk["k_cut"])
    p_star = float(rk["p_star"])
    p_star_raw = float(rk["p_star_raw"])
    p_cv = float(rk["p_cv"])
    log(f"  k_cut       = {k_cut}")
    log(f"  p_star_raw  = {p_star_raw:.4f}")
    log(f"  p_star      = {p_star:.4f}  (after recipe_K calibration)")
    log(f"  p_cv        = {p_cv:.4f}  (pool fraction for {N_FOLDS}-fold)")
    log(f"  status      = {rk.get('status', 'n/a')}")

    log(f"\nRunning {N_FOLDS}-fold CV at p_cv={p_cv:.4f}  ranks={RANKS}")

    rng = np.random.default_rng(SEED)
    n = s.shape[0]
    off_diag = ~np.eye(n, dtype=bool)

    jobs = []
    for rep in range(N_REPS):
        M_outer = mask_missing_entries(s, p_cv, rng, missing_values=np.nan)
        folds = split_omega_into_folds(M_outer, N_FOLDS, rng)
        if folds is None:
            log("  insufficient pool for CV; aborting")
            return
        for fold_idx, V in enumerate(folds):
            holdout = V | M_outer
            val_mask = V & off_diag
            for r_idx, r in enumerate(RANKS):
                seed_rfk = int(SEED + 1000 * (rep + 1) + 100 * fold_idx + 13 * r)
                jobs.append((rep, fold_idx, r, holdout, val_mask, seed_rfk))

    log(f"  total fits = {len(jobs)} (n_jobs={N_JOBS}, max_outer={MAX_OUTER})")

    t0 = time.time()
    scores = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(fit_score_admm)(
            s, holdout, val_mask, r, bounds, seed,
            MAX_OUTER, MAX_INNER,
        )
        for (_, _, r, holdout, val_mask, seed) in jobs
    )
    elapsed = time.time() - t0
    log(f"CV done in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    df = pd.DataFrame(
        [(rep, fold, r, sc)
         for (rep, fold, r, *_), sc in zip(jobs, scores)],
        columns=["rep", "fold", "rank", "val_mse"],
    )
    df.to_csv(OUTPUT_DIR / "cv.csv", index=False)

    grouped = df.groupby("rank")["val_mse"]
    mean = grouped.mean().reindex(RANKS).to_numpy()
    counts = np.maximum(grouped.count().reindex(RANKS).to_numpy(), 1)
    sem = grouped.std().reindex(RANKS).to_numpy() / np.sqrt(counts)
    argmin_rank = int(RANKS[int(np.nanargmin(mean))])

    log("")
    log(f"{'rank':>5}  {'mean':>12}  {'sem':>10}")
    for r, m, e in zip(RANKS, mean, sem):
        marker = ""
        if r == argmin_rank:
            marker = "  <- argmin"
        if r == k_cut:
            marker += "  <- k_cut"
        log(f"{r:>5}  {m:>12.3f}  {e:>10.3f}{marker}")
    log("")
    log(f"REFERENCE (update_pysrf) RESULT:")
    log(f"  k_cut       = {k_cut}")
    log(f"  argmin_rank = {argmin_rank}")
    if argmin_rank == k_cut:
        log("  PASS: reference CV argmin matches reference k_cut.")
    elif abs(argmin_rank - k_cut) <= 5:
        log("  CLOSE.")
    else:
        log("  MISMATCH: reference also shows divergence -> our finding is real.")


if __name__ == "__main__":
    main()
