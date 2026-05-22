"""Quick Mur92-only reproduction (fast) for diffing against the sandbox baseline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from pysrf import cross_val_score, estimate_rank
from src.utils import get_output_dir


OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MUR_PATH = PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings" / "outputs" / ".cache" / "sim_mur92.npy"

K_CV = 5
N_REPEATS = 5
MAX_OUTER_VALUES = [100, 1000]
SRF_KWARGS = {"rho": 3.0, "max_inner": 30, "tol": 0.0}
RANKS = list(range(2, 31))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

    s = np.load(MUR_PATH)
    est = estimate_rank(s)
    log.info("k_cut=%d  p_star=%.4f", est.rank, est.sampling_fraction)

    new = []
    for mo in MAX_OUTER_VALUES:
        curve = cross_val_score(
            s, ranks=RANKS, sampling_fraction=est.sampling_fraction,
            n_folds=K_CV, n_repeats=N_REPEATS,
            random_state=mo, srf_kwargs={**SRF_KWARGS, "max_outer": mo},
        )
        curve = curve.assign(max_outer=mo)
        new.append(curve)
    new = pd.concat(new, ignore_index=True)
    new.to_csv(OUTPUT_DIR / "mur_new.csv", index=False)

    sandbox_path = (
        PROJECT_ROOT
        / "sandbox" / "rank_estimation" / "cv_diagnostics" / "spectrum_vs_cv" / "outputs" / "records.csv"
    )
    old = pd.read_csv(sandbox_path)
    old = old[old["dataset"] == "mur92"]

    new_mean = new.groupby(["max_outer", "rank"])["val_mse"].mean()
    old_mean = old.groupby(["max_outer", "rank"])["val_mse"].mean()

    log.info("\nMur92 comparison (new pysrf vs sandbox baseline):")
    log.info("  %-8s %-5s  %-12s  %-12s  %-7s", "max_outer", "rank", "new", "sandbox", "ratio")
    for mo in MAX_OUTER_VALUES:
        for r in RANKS:
            n_val = float(new_mean.get((mo, r), np.nan))
            o_val = float(old_mean.get((mo, r), np.nan))
            ratio = n_val / o_val if o_val > 0 else float("nan")
            log.info("  %-8d %-5d  %-12.4e  %-12.4e  %-7.3f",
                     mo, r, n_val, o_val, ratio)

    summary = []
    for mo in MAX_OUTER_VALUES:
        diffs = []
        for r in RANKS:
            n_val = float(new_mean.get((mo, r), np.nan))
            o_val = float(old_mean.get((mo, r), np.nan))
            if np.isfinite(n_val) and np.isfinite(o_val) and o_val > 0:
                diffs.append(abs(n_val - o_val) / o_val)
        if diffs:
            a = np.array(diffs)
            summary.append(f"max_outer={mo}: relative diff median={np.median(a):.3%}, max={a.max():.3%}")
    log.info("\nSummary:")
    for line in summary:
        log.info("  %s", line)
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
