"""Grid search: SWOW similarity / SRF / consensus knobs vs. Glasgow norm prediction.

Two phases:
  Phase 0 — Ceiling: kernel ridge on the RAW similarity matrix, no SRF.
            Gives an upper bound on extractable information.
  Phase 1 — Single-run grid: SRF (one run, no consensus) under each
            (rank, symmetrization, zeros_as_missing, top_n_words) combo,
            then Ridge CV on the 9 Glasgow norms.

Each config writes a CSV row immediately so the script is resume-safe and
visibly progressing. Re-running skips configs already in results.csv.

Phase 2 (consensus on best configs) is a separate launch — see
consensus_top.py once you've inspected the grid.

Usage:
    poetry run python sandbox/swow/predict_grid/run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from datasets import load_swow_ppmi
from experiments.analyses.swow.norms import load_behavioral_ratings
from pysrf import SRF
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
RESULTS_CSV = OUTPUT_DIR / "results.csv"
LOG_PATH = OUTPUT_DIR / "log.txt"

PROJECT_ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
DATA_DIR = PROJECT_ROOT / "data"
SWOW_DIR = DATA_DIR / "small-world-of-words"

PROPERTIES = ["arousal", "valence", "dominance", "concreteness",
              "imageability", "familiarity", "aoa", "size", "gender"]

# Phase 1 grid. Kept small enough to run in a few hours; expand later if useful.
RANKS = [100, 200, 268, 350]
SYMS = ["sum", "mean", "geometric_mean"]
ZEROS_AS_MISSING = [False, True]
TOP_N_WORDS = [None, 5000]

SRF_KWARGS = dict(rho=3.0, max_inner=30, tol=0.0, max_outer=50, check_input=False)
RANDOM_STATE = 42
N_OUTER_FOLDS = 5


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def ridge_cv_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = RidgeCV(alphas=np.logspace(-4, 2, 13))
    cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, x_scaled, y, cv=cv)


def kernel_ridge_cv_predict(k: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kernel ridge on a pre-computed kernel matrix."""
    model = KernelRidge(alpha=1.0, kernel="precomputed")
    cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full_like(y, np.nan, dtype=float)
    for train_idx, test_idx in cv.split(np.arange(len(y))):
        # KernelRidge with precomputed kernel requires (n_test, n_train) at predict time
        k_train = k[np.ix_(train_idx, train_idx)]
        k_test = k[np.ix_(test_idx, train_idx)]
        model.fit(k_train, y[train_idx])
        preds[test_idx] = model.predict(k_test)
    return preds


def score_norms(predict_fn, x_or_k, vocabulary: list[str],
                ratings: pd.DataFrame) -> dict[str, float]:
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}
    out = {}
    for prop in PROPERTIES:
        valid = ratings[["word", prop]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]
        if len(valid) < 50:
            out[prop] = np.nan
            continue
        idx = np.array([word_to_idx[w] for w in valid["word"]])
        y = valid[prop].to_numpy()
        if predict_fn.__name__ == "kernel_ridge_cv_predict":
            k_sub = x_or_k[np.ix_(idx, idx)]
            preds = predict_fn(k_sub, y)
        else:
            preds = predict_fn(x_or_k[idx], y)
        rho, _ = spearmanr(preds, y)
        out[prop] = float(rho)
    return out


def append_row(row: dict) -> None:
    df_new = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df_new = pd.concat([pd.read_csv(RESULTS_CSV), df_new], ignore_index=True)
    df_new.to_csv(RESULTS_CSV, index=False)


def already_done(config_id: str) -> bool:
    if not RESULTS_CSV.exists():
        return False
    df = pd.read_csv(RESULTS_CSV)
    return (df["config_id"] == config_id).any()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ratings = load_behavioral_ratings(DATA_DIR)
    log(f"loaded ratings: {len(ratings)} rows")

    # =====================================================================
    # PHASE 0 — Ceiling: kernel ridge on raw similarity
    # =====================================================================
    config_id = "ceiling/raw_swow_ppmi_default"
    if not already_done(config_id):
        log("=== PHASE 0: ceiling (kernel ridge on raw SWOW PPMI) ===")
        sim, vocab, _ = load_swow_ppmi(SWOW_DIR, use_all_responses=True)
        log(f"  loaded swow ppmi: shape={sim.shape}")
        # Make sure it's a proper kernel (PSD-ish) for kernel ridge.
        # PPMI can have negatives — replace nan with 0, symmetrize, add tiny ridge.
        k = np.nan_to_num(sim, nan=0.0)
        k = 0.5 * (k + k.T)
        scores = score_norms(kernel_ridge_cv_predict, k, vocab, ratings)
        mean_rho = float(np.nanmean(list(scores.values())))
        log(f"  ceiling mean_rho={mean_rho:.4f}  per-norm={scores}")
        append_row({
            "config_id": config_id, "phase": "ceiling", "method": "KernelRidge",
            "rank": None, "symmetrization": "sum", "zeros_as_missing": False,
            "top_n_words": None, "n": int(sim.shape[0]),
            "mean_rho": mean_rho, **scores,
        })
    else:
        log("PHASE 0 already done, skipping")

    # =====================================================================
    # PHASE 1 — single-SRF-run grid
    # =====================================================================
    log("=== PHASE 1: SRF single-run grid ===")
    # Cache similarities per (sym, zeros, top_n) so we don't rebuild for each rank
    sim_cache: dict[tuple, tuple[np.ndarray, list[str]]] = {}

    grid = [
        (rank, sym, zam, topn)
        for sym in SYMS
        for zam in ZEROS_AS_MISSING
        for topn in TOP_N_WORDS
        for rank in RANKS
    ]
    log(f"  total configs: {len(grid)}")

    for i, (rank, sym, zam, topn) in enumerate(grid, 1):
        config_id = f"srf/rank={rank}_sym={sym}_zam={zam}_topn={topn}"
        if already_done(config_id):
            log(f"[{i}/{len(grid)}] {config_id} already done, skipping")
            continue
        log(f"[{i}/{len(grid)}] {config_id} starting")

        key = (sym, zam, topn)
        if key not in sim_cache:
            t0 = time.time()
            sim, vocab, _ = load_swow_ppmi(
                SWOW_DIR, use_all_responses=True,
                symmetrization=sym, top_n_words=topn,
                bidirectional_only=False,
            )
            if zam:
                sim = np.where(sim == 0, np.nan, sim).astype(np.float64)
            else:
                sim = sim.astype(np.float64)
            np.fill_diagonal(sim, np.nanmax(sim))  # SRF expects diag = max
            sim_cache[key] = (sim, vocab)
            log(f"  built similarity sym={sym} zam={zam} topn={topn} n={sim.shape[0]} in {time.time()-t0:.1f}s")
        sim, vocab = sim_cache[key]

        t0 = time.time()
        srf = SRF(rank=rank, random_state=RANDOM_STATE, missing_values=np.nan, **SRF_KWARGS)
        srf.fit(sim)
        embedding = srf.w_
        fit_sec = time.time() - t0
        log(f"  SRF fit done in {fit_sec:.1f}s  W shape={embedding.shape}  sparsity={(embedding == 0).mean():.3f}")

        t0 = time.time()
        scores = score_norms(ridge_cv_predict, embedding, vocab, ratings)
        eval_sec = time.time() - t0
        mean_rho = float(np.nanmean(list(scores.values())))
        log(f"  ridge eval done in {eval_sec:.1f}s  mean_rho={mean_rho:.4f}")

        append_row({
            "config_id": config_id, "phase": "srf_single", "method": "Ridge",
            "rank": rank, "symmetrization": sym, "zeros_as_missing": zam,
            "top_n_words": topn, "n": int(sim.shape[0]),
            "fit_sec": fit_sec, "eval_sec": eval_sec,
            "mean_rho": mean_rho, **scores,
        })

    log("=== ALL DONE ===")
    log(f"results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
