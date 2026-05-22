"""VGG16 5-fold CV protocol check at three ranks.

This is intentionally a small sandbox run: one JSON output, no CSV fragments.
It runs the patched public pysrf CV path at ranks where the VGG16 curve should
show the high-rank rebound, and separately checks that the CV split mechanics
match update_pysrf when driven by the same random stream.
"""

from __future__ import annotations

import json
import os
import sys
import time
import importlib
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THIRD_PARTY_PYSRF = PROJECT_ROOT / "third_party" / "pysrf"
UPDATE_PYSRF_SRC = PROJECT_ROOT / "update_pysrf" / "src"
DATASET = os.environ.get("DATASET", "vgg16")
CACHE_PATH = Path(os.environ.get(
    "CACHE_PATH",
    PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs" / "cache" / f"{DATASET}.npy",
))
SCALE_MODE = os.environ.get("SCALE_MODE", "none")

sys.path.insert(0, str(THIRD_PARTY_PYSRF))
from pysrf import SRF, cross_val_score  # noqa: E402

pysrf_cv = importlib.import_module("pysrf.cross_validation")

sys.path.insert(0, str(UPDATE_PYSRF_SRC))
from _common import split_omega_into_folds  # noqa: E402
from symmnmf.cross_validation import mask_missing_entries  # noqa: E402

RANKS = [int(r) for r in os.environ.get("RANKS", "50,90,120").split(",")]
N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
N_REPEATS = int(os.environ.get("N_REPEATS", "1"))
SEED = int(os.environ.get("SEED", "0"))
P_CV = float(os.environ.get("P_CV", "0.872"))
MAX_OUTER = int(os.environ.get("MAX_OUTER", "30"))
MAX_INNER = int(os.environ.get("MAX_INNER", "30"))
TOL = float(os.environ.get("TOL", "1e-4"))
RHO = 3.0
N_JOBS = int(os.environ.get("N_JOBS", str(min(len(RANKS) * N_FOLDS, 15))))
RUN_UPDATE_SPLIT = os.environ.get("RUN_UPDATE_SPLIT", "1") != "0"
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "summary.json")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_similarity() -> np.ndarray:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing cache: {CACHE_PATH}")
    s = np.asarray(np.load(CACHE_PATH), dtype=np.float64)
    if SCALE_MODE == "none":
        return s
    if SCALE_MODE == "max":
        return s / float(np.nanmax(s))
    if SCALE_MODE == "absmax":
        return s / float(np.nanmax(np.abs(s)))
    raise ValueError(f"Unknown SCALE_MODE={SCALE_MODE!r}; expected none, max, or absmax.")


def compare_split_functions(s: np.ndarray) -> dict[str, Any]:
    """No-fit check that update_pysrf and pysrf build equivalent masks."""
    observed_mask = np.isfinite(s)

    rng_update = check_random_state(SEED)
    update_missing = mask_missing_entries(s, P_CV, rng_update, missing_values=np.nan)
    update_pool = ~update_missing
    update_folds = split_omega_into_folds(update_missing, N_FOLDS, rng_update)
    if update_folds is None:
        raise RuntimeError("update_pysrf split has too few observed entries.")

    rng_pysrf = check_random_state(SEED)
    pysrf_pool = pysrf_cv._sample_cv_pool(observed_mask, P_CV, rng_pysrf)
    pysrf_folds = pysrf_cv._validation_masks(pysrf_pool, N_FOLDS, rng_pysrf)

    triu = np.triu_indices(s.shape[0], k=1)
    return {
        "same_pool_mask": bool(np.array_equal(update_pool[triu], pysrf_pool[triu])),
        "same_validation_masks": [
            bool(np.array_equal(u[triu], p[triu]))
            for u, p in zip(update_folds, pysrf_folds, strict=True)
        ],
        "pool_pairs": int(np.count_nonzero(update_pool[triu])),
        "validation_pairs_per_fold": [
            int(np.count_nonzero(mask[triu])) for mask in update_folds
        ],
        "diagonal_observed_in_update_pool": bool(np.all(np.diag(update_pool))),
    }


def make_update_pysrf_splits(s: np.ndarray) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    rng = check_random_state(SEED)
    off_diag = ~np.eye(s.shape[0], dtype=bool)
    splits = []
    for rep in range(N_REPEATS):
        missing_outer = mask_missing_entries(s, P_CV, rng, missing_values=np.nan)
        folds = split_omega_into_folds(missing_outer, N_FOLDS, rng)
        if folds is None:
            raise RuntimeError("update_pysrf split has too few observed entries.")
        for fold, validation_mask in enumerate(folds):
            holdout_mask = missing_outer | validation_mask
            splits.append((rep, fold, holdout_mask, validation_mask & off_diag))
    return splits


def score_srf_on_split(
    s: np.ndarray,
    holdout_mask: np.ndarray,
    validation_mask: np.ndarray,
    rank: int,
    bounds: tuple[float, float],
    seed: int,
) -> float:
    with threadpool_limits(limits=1):
        train = np.full(s.shape, np.nan, dtype=np.float64)
        train[~holdout_mask] = s[~holdout_mask]
        est = SRF(
            rank=rank,
            rho=RHO,
            max_outer=MAX_OUTER,
            max_inner=MAX_INNER,
            tol=TOL,
            missing_values=np.nan,
            bounds=bounds,
            random_state=seed,
            check_input=False,
        )
        est.fit(train)
        rec = est.reconstruct()
        valid = validation_mask & np.isfinite(s) & np.isfinite(rec)
        return float(np.mean((s[valid] - rec[valid]) ** 2))


def run_update_split_pysrf(s: np.ndarray) -> pd.DataFrame:
    bounds = (float(np.nanmin(s)), float(np.nanmax(s)))
    splits = make_update_pysrf_splits(s)
    jobs = []
    for rep, fold, holdout_mask, validation_mask in splits:
        for rank_idx, rank in enumerate(RANKS):
            fit_seed = int(SEED + 1000 * (rep + 1) + 100 * fold + rank_idx)
            jobs.append((rep, fold, rank, holdout_mask, validation_mask, fit_seed))

    log(f"Running update_pysrf_split_with_srf: {len(jobs)} fits, n_jobs={N_JOBS}")
    t0 = time.time()
    scores = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(score_srf_on_split)(s, holdout, val, rank, bounds, fit_seed)
        for _, _, rank, holdout, val, fit_seed in jobs
    )
    elapsed = time.time() - t0
    log(f"Finished update_pysrf_split_with_srf in {elapsed:.1f}s")
    return pd.DataFrame(
        {
            "engine": "update_pysrf_split_with_srf",
            "rep": rep,
            "fold": fold,
            "rank": rank,
            "val_mse": score,
        }
        for (rep, fold, rank, *_), score in zip(jobs, scores)
    )


def run_public_pysrf(s: np.ndarray) -> pd.DataFrame:
    sampling_fraction = P_CV * (N_FOLDS - 1) / N_FOLDS
    log(
        "Running pysrf.cross_val_score: "
        f"sampling_fraction={sampling_fraction:.4f}, ranks={RANKS}, n_jobs={N_JOBS}"
    )
    t0 = time.time()
    df = cross_val_score(
        s,
        ranks=RANKS,
        sampling_fraction=sampling_fraction,
        n_folds=N_FOLDS,
        n_repeats=N_REPEATS,
        random_state=SEED,
        n_jobs=N_JOBS,
        srf_kwargs={
            "rho": RHO,
            "max_outer": MAX_OUTER,
            "max_inner": MAX_INNER,
            "tol": TOL,
            "check_input": False,
        },
    )
    elapsed = time.time() - t0
    log(f"Finished pysrf.cross_val_score in {elapsed:.1f}s")
    df = df.copy()
    df.insert(0, "engine", "pysrf_cross_val_score")
    return df


def summarize(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for engine, g_engine in df.groupby("engine", sort=False):
        means = g_engine.groupby("rank")["val_mse"].mean().reindex(RANKS)
        sems = g_engine.groupby("rank")["val_mse"].sem().reindex(RANKS)
        argmin_rank = int(means.idxmin())
        for rank in RANKS:
            rows.append(
                {
                    "engine": engine,
                    "rank": int(rank),
                    "mean_val_mse": float(means.loc[rank]),
                    "sem_val_mse": float(sems.loc[rank]) if np.isfinite(sems.loc[rank]) else None,
                    "argmin_rank": argmin_rank,
                }
            )
    return rows


def main() -> None:
    log(f"Loading cached {DATASET} similarity.")
    s = load_similarity()
    log(f"shape={s.shape}, bounds=({float(np.nanmin(s)):.3f}, {float(np.nanmax(s)):.3f})")
    log(f"p_cv={P_CV:.4f}; effective train fraction={P_CV * 4 / 5:.4f}")

    split_check = compare_split_functions(s)
    log(
        "Split-function check: "
        f"same_pool={split_check['same_pool_mask']}, "
        f"same_folds={all(split_check['same_validation_masks'])}, "
        f"pool_pairs={split_check['pool_pairs']}"
    )

    frames = []
    if RUN_UPDATE_SPLIT:
        frames.append(run_update_split_pysrf(s))
    frames.append(run_public_pysrf(s))
    df = pd.concat(frames, ignore_index=True)
    summary = summarize(df)

    result = {
        "metadata": {
            "dataset": DATASET,
            "ranks": RANKS,
            "n_folds": N_FOLDS,
            "n_repeats": N_REPEATS,
            "seed": SEED,
            "p_cv": P_CV,
            "sampling_fraction_for_pysrf": P_CV * (N_FOLDS - 1) / N_FOLDS,
            "max_outer": MAX_OUTER,
            "max_inner": MAX_INNER,
            "tol": TOL,
            "rho": RHO,
            "n_jobs": N_JOBS,
            "run_update_split": RUN_UPDATE_SPLIT,
            "cache_path": str(CACHE_PATH),
            "scale_mode": SCALE_MODE,
        },
        "split_check": split_check,
        "summary": summary,
        "records": df.to_dict(orient="records"),
    }
    out_path = OUTPUT_DIR / OUTPUT_NAME
    out_path.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")

    log("")
    log(f"{'engine':>24}  {'rank':>5}  {'mean_val_mse':>14}  {'sem':>10}  marker")
    for row in summary:
        marker = "<- argmin" if row["rank"] == row["argmin_rank"] else ""
        sem = "nan" if row["sem_val_mse"] is None else f"{row['sem_val_mse']:.3f}"
        log(
            f"{row['engine']:>24}  {row['rank']:>5}  "
            f"{row['mean_val_mse']:>14.3f}  {sem:>10}  {marker}"
        )
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
