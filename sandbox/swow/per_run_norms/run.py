"""Evaluate Glasgow Norms prediction for each individual SRF run.

Checks whether the v2 (1000 iter) downstream drop is specific to the
selected run or affects all 50 runs uniformly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.datasets.swow import load_swow_ppmi
from src.utils import get_output_dir
from experiments.analyses.swow.norms import PROPERTIES, load_behavioral_ratings

log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_PATH = PROJECT_ROOT / "experiments/analyses/swow/missing_zeros/outputs/runs.npy"
SUMMARY_PATH = PROJECT_ROOT / "experiments/analyses/swow/missing_zeros/outputs/summary.json"


def _eval_one_run(embedding: np.ndarray, vocabulary: list[str],
                   ratings: pd.DataFrame) -> dict[str, float]:
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}
    out = {}
    for prop in PROPERTIES:
        valid = ratings[["word", prop]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]
        if len(valid) < 50:
            out[prop] = np.nan
            continue
        indices = np.array([word_to_idx[w] for w in valid["word"]])
        x = embedding[indices]
        y = valid[prop].to_numpy()
        x_scaled = StandardScaler().fit_transform(x)
        model = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        preds = cross_val_predict(
            model, x_scaled, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        rho, _ = spearmanr(preds, y)
        out[prop] = float(rho)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info(f"Loading runs from {RUNS_PATH}")
    runs = np.load(RUNS_PATH)
    log.info(f"Runs shape: {runs.shape}")

    summary = json.loads(SUMMARY_PATH.read_text())
    selected_idx = summary["selected_run_idx"]
    log.info(f"Selected run index: {selected_idx}")

    log.info("Loading SWOW vocabulary...")
    swow_dir = DATA_DIR / "small-world-of-words"
    _, vocabulary, _ = load_swow_ppmi(
        swow_dir, use_all_responses=True, symmetrization="sum",
    )

    log.info("Loading Glasgow Norms...")
    ratings = load_behavioral_ratings(DATA_DIR)

    n_runs = runs.shape[0]
    rows = []
    for i in range(n_runs):
        log.info(f"Evaluating run {i+1}/{n_runs}...")
        result = _eval_one_run(runs[i], vocabulary, ratings)
        result["run_idx"] = i
        result["is_selected"] = (i == selected_idx)
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "per_run_norms.csv", index=False)

    log.info("\n=== Per-run Glasgow Norms (mean across properties) ===")
    df["mean_rho"] = df[PROPERTIES].mean(axis=1)
    log.info(f"\n{df[['run_idx', 'is_selected', 'mean_rho']].to_string(index=False)}")

    log.info("\n=== Summary across runs ===")
    for prop in PROPERTIES + ["mean_rho"]:
        vals = df[prop].dropna()
        log.info(f"  {prop:15s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                 f"min={vals.min():.4f}  max={vals.max():.4f}")

    selected_row = df[df["is_selected"]].iloc[0]
    log.info(f"\n=== Selected run (idx={selected_idx}) ===")
    log.info(f"  mean_rho: {selected_row['mean_rho']:.4f}")
    log.info(f"  rank among 50 runs: {(df['mean_rho'] > selected_row['mean_rho']).sum() + 1}/50")

    log.info(f"\nSaved to {OUTPUT_DIR / 'per_run_norms.csv'}")


if __name__ == "__main__":
    main()
