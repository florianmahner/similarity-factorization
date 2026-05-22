"""vgg16 CV with looser recovery_tolerance.

Principled test: does using a looser recovery_tolerance (0.20 vs the default
0.10) shift the CV argmin toward k_cut=45?

Mechanism:
  - recovery_tolerance controls p* directly
  - 0.20 -> p* = 0.49 (vs 0.10 -> p* = 0.70)
  - lower p* = less train data per fold
  - less train data = high-rank fits are more constrained
  - if rank-90 was winning only because of abundant train data, it should
    lose to rank-45 at this lower p*

If the argmin moves to 45 here, we have a principled fix:
"recovery_tolerance is the knob that aligns CV with leakage k_cut."
If it stays at 90, the methods genuinely disagree for heavy-tailed spectra.

Run:
    poetry run python sandbox/dimensionality/vgg16_tolerance_cv/run.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from pysrf.cross_validation import (
    _cv_pool_fraction,
    _entry_splits,
    _fit_score,
    _observed_bounds,
    _split_fit_seeds,
)
from pysrf._common import observation_mask
from similarity import build_similarity
from src.utils import get_output_dir
from src.utils.figure_theme import despine
from src.colors import GRAY, INDIGO, ROSE

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()

RANKS = [30, 45, 60, 90]
N_FOLDS = 5
N_REPEATS = 1
RANDOM_STATE = 0
RECOVERY_TOLERANCE = 0.20  # loosened from default 0.10
# Skip estimate_rank (~80s) — p* at this tolerance is cached from
# sandbox/dimensionality/vgg16_spectrum/tolerance_sweep.py output.
P_STAR_CACHED = 0.4945
K_CUT_CACHED = 45
SRF_KWARGS = {"rho": 3.0, "max_inner": 30, "tol": 0.0, "max_outer": 50, "verbose": 1}
N_JOBS = 20  # one worker per fit; OMP=1 globally -> no nested threading


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


def main() -> None:
    log("=" * 70)
    log(f"vgg16 CV with recovery_tolerance={RECOVERY_TOLERANCE}")
    log("=" * 70)

    log("Loading vgg16 similarity matrix...")
    s = load_vgg16()
    s = np.asarray(s, dtype=np.float64)
    log(f"  shape={s.shape}")

    p_star = P_STAR_CACHED
    k_cut = K_CUT_CACHED
    log(f"  using cached k_cut={k_cut}  p*={p_star:.4f}  "
        f"(recovery_tolerance={RECOVERY_TOLERANCE}, from prior estimate_rank)")
    log(f"CV at p*={p_star:.4f}  ranks={RANKS}")

    obs_mask = observation_mask(s, np.nan)
    bounds = _observed_bounds(s, obs_mask)
    pool_fraction = _cv_pool_fraction(p_star, N_FOLDS, s.shape[0])
    split_seeds, fit_seeds = _split_fit_seeds(
        RANDOM_STATE, N_REPEATS, N_FOLDS, len(RANKS),
    )
    splits = _entry_splits(obs_mask, pool_fraction, N_FOLDS, split_seeds)

    jobs = []
    for rep, fold, train_mask, validation_mask in splits:
        for r_idx, rank in enumerate(RANKS):
            seed = int(fit_seeds[rep, fold, r_idx])
            jobs.append((rep, fold, int(rank), train_mask, validation_mask, seed))

    log(f"  pool_fraction={pool_fraction:.4f}  total fits={len(jobs)}")

    t0 = time.time()
    scores = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_fit_score)(s, tm, vm, rank, bounds, seed, SRF_KWARGS)
        for (_, _, rank, tm, vm, seed) in jobs
    )
    elapsed = time.time() - t0
    log(f"CV done in {elapsed:.1f}s")

    df = pd.DataFrame(
        [(rep, fold, rank, score)
         for (rep, fold, rank, *_), score in zip(jobs, scores)],
        columns=["rep", "fold", "rank", "val_mse"],
    )
    df.to_csv(OUTPUT_DIR / "cv.csv", index=False)

    grouped = df.groupby("rank")["val_mse"]
    mean = grouped.mean().reindex(RANKS).to_numpy()
    counts = np.maximum(grouped.count().reindex(RANKS).to_numpy(), 1)
    sem = grouped.std().reindex(RANKS).to_numpy() / np.sqrt(counts)
    argmin_rank = int(RANKS[int(np.nanargmin(mean))])

    log("")
    log(f"{'rank':>5}  {'mean':>10}  {'sem':>9}")
    for r, m, e in zip(RANKS, mean, sem):
        marker = ""
        if r == argmin_rank:
            marker = "  <- argmin"
        if r == k_cut:
            marker += "  <- k_cut"
        log(f"{r:>5}  {m:>10.3f}  {e:>9.3f}{marker}")
    log("")
    log(f"k_cut       = {k_cut}")
    log(f"argmin_rank = {argmin_rank}")
    if argmin_rank == k_cut:
        log("PASS: CV argmin matches k_cut at this tolerance.")
    elif abs(argmin_rank - k_cut) <= 5:
        log("CLOSE: CV argmin near k_cut.")
    else:
        log("STILL MISMATCH: spectrum truly heavy-tailed, methods disagree.")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(RANKS, mean, yerr=sem, marker="o", color=INDIGO, capsize=2)
    ax.axvline(k_cut, color=GRAY, linestyle="--", label=f"k_cut={k_cut}")
    ax.axvline(argmin_rank, color=ROSE, linestyle=":", label=f"argmin={argmin_rank}")
    ax.set_xlabel("rank")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"vgg16 CV  tol={RECOVERY_TOLERANCE}  p*={p_star:.3f}")
    ax.legend(frameon=False)
    despine(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cv_curve.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    summary = {
        "dataset": "vgg16",
        "n": int(s.shape[0]),
        "recovery_tolerance": RECOVERY_TOLERANCE,
        "k_cut": int(k_cut),
        "p_star": p_star,
        "srf_kwargs": SRF_KWARGS,
        "n_folds": N_FOLDS,
        "ranks": RANKS,
        "val_mse_mean": [float(v) for v in mean],
        "val_mse_sem": [float(v) for v in sem],
        "argmin_rank": argmin_rank,
        "runtime_sec": round(elapsed, 1),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
