"""Try several random_state values on Peterson-various; report which one best
matches the sandbox baseline argmins (12 at mo=100, 8 at mo=1000)."""

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
PETERSON_PATH = (
    PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings" / "outputs"
    / ".cache" / "sim_peterson-various.npy"
)
SANDBOX_RECORDS = (
    PROJECT_ROOT / "sandbox" / "rank_estimation" / "cv_diagnostics"
    / "spectrum_vs_cv" / "outputs" / "records.csv"
)

K_CV = 5
N_REPEATS = 5
MAX_OUTER_VALUES = [100, 1000]
SRF_KWARGS = {"rho": 3.0, "max_inner": 30, "tol": 0.0}
RANKS = list(range(2, 31))
RANDOM_STATE_SWEEP = [0, 1, 7, 42, 100, 1000, 6_000_000, 12345]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

    s = np.load(PETERSON_PATH)
    est = estimate_rank(s)
    log.info("Peterson-various: k_cut=%d, p_star=%.4f", est.rank, est.sampling_fraction)

    sandbox = pd.read_csv(SANDBOX_RECORDS)
    sandbox = sandbox[sandbox["dataset"] == "peterson-various"]
    sandbox_mean = sandbox.groupby(["max_outer", "rank"])["val_mse"].mean()
    sandbox_argmin = {mo: int(sandbox_mean.loc[mo].idxmin()) for mo in MAX_OUTER_VALUES}
    log.info("Sandbox baseline argmins: %s", sandbox_argmin)

    summary_rows = []
    for rs in RANDOM_STATE_SWEEP:
        log.info("--- random_state=%d ---", rs)
        argmins = {}
        rel_diffs = {}
        for mo in MAX_OUTER_VALUES:
            curve = cross_val_score(
                s, ranks=RANKS, sampling_fraction=est.sampling_fraction,
                n_folds=K_CV, n_repeats=N_REPEATS,
                random_state=rs, srf_kwargs={**SRF_KWARGS, "max_outer": mo},
            )
            mean = curve.groupby("rank")["val_mse"].mean()
            argmins[mo] = int(mean.idxmin())
            # Per-rank relative diff vs sandbox
            diffs = []
            for r in RANKS:
                n_val = float(mean.loc[r])
                o_val = float(sandbox_mean.loc[(mo, r)]) if (mo, r) in sandbox_mean.index else np.nan
                if np.isfinite(n_val) and np.isfinite(o_val) and o_val > 0:
                    diffs.append(abs(n_val - o_val) / o_val)
            rel_diffs[mo] = (np.median(diffs), np.max(diffs)) if diffs else (np.nan, np.nan)
            log.info("  mo=%d: argmin=%d (sandbox=%d)  median_rel=%.2%%  max_rel=%.2%%",
                     mo, argmins[mo], sandbox_argmin[mo],
                     rel_diffs[mo][0], rel_diffs[mo][1])
        match_score = sum(abs(argmins[mo] - sandbox_argmin[mo]) for mo in MAX_OUTER_VALUES)
        summary_rows.append({
            "random_state": rs,
            "argmin_100": argmins[100],
            "argmin_1000": argmins[1000],
            "delta_100": argmins[100] - sandbox_argmin[100],
            "delta_1000": argmins[1000] - sandbox_argmin[1000],
            "argmin_distance": match_score,
            "med_rel_100": rel_diffs[100][0],
            "med_rel_1000": rel_diffs[1000][0],
        })

    summary = pd.DataFrame(summary_rows).sort_values("argmin_distance")
    summary.to_csv(OUTPUT_DIR / "sweep.csv", index=False)
    log.info("\nSummary (sorted by argmin distance to sandbox):")
    log.info("\n%s", summary.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
