"""things_behavior longer-training CV — BATCHED.

All 15 ranks dispatched in ONE cross_val_score call -> 15*5*10 = 750 fits
in one joblib pool. With n_jobs=72 on insula (all physical cores), this
is roughly 750/72 ≈ 10 waves -> wall time ≈ 10x slowest-fit-time
instead of 15x per-rank-time of the serial version.

Writes one CSV when done (since cross_val_score is one batched call).
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd

from pysrf import cross_val_score
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
SIM_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/cache/things_behavior.npy")
CV_JSON = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json")

RANKS = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120, 150]
N_REPEATS = 10
MAX_OUTER = 200
N_JOBS = 72  # insula physical cores


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = np.load(SIM_PATH).astype(np.float64)
    p_star = json.loads(CV_JSON.read_text())["validations"]["5fold"]["params"]["sampling_fraction"]
    n_fits = len(RANKS) * 5 * N_REPEATS
    log(f"n={sim.shape[0]} p*={p_star:.4f}")
    log(f"BATCHED: ranks={RANKS} n_repeats={N_REPEATS} max_outer={MAX_OUTER}")
    log(f"dispatching {n_fits} fits to joblib (n_jobs={N_JOBS})")

    srf_kwargs = dict(rho=3.0, max_inner=30, max_outer=MAX_OUTER, tol=0.0, check_input=False)
    t0 = time.time()
    curve = cross_val_score(
        sim, ranks=RANKS, sampling_fraction=p_star,
        n_folds=5, n_repeats=N_REPEATS, random_state=42, n_jobs=N_JOBS,
        srf_kwargs=srf_kwargs,
    )
    elapsed = time.time() - t0
    log(f"all {n_fits} fits done in {elapsed:.1f}s")

    # Aggregate per-rank stats
    summary = curve.groupby("rank")["val_mse"].agg(["mean", "std", "count"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary = summary.reset_index().rename(columns={"mean": "val_mse_mean", "sem": "val_mse_sem"})
    summary.to_csv(OUTPUT_DIR / "cv_results_batched.csv", index=False)
    for _, row in summary.iterrows():
        log(f"  rank={int(row['rank']):>3}: val_mse={row['val_mse_mean']:.6e} ± {row['val_mse_sem']:.2e}")
    argmin = summary.loc[summary["val_mse_mean"].idxmin()]
    log(f"argmin rank: {int(argmin['rank'])} (val_mse={argmin['val_mse_mean']:.6e})")


if __name__ == "__main__":
    main()
