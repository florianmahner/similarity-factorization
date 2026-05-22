"""SWOW consensus embedding with PPMI zeros treated as missing data.

The SWOW PPMI matrix has 98.7% zero entries representing word pairs that were
never sampled together in the association task. Following the precedent of GloVe
(which assigns zero weight to unobserved pairs) and the PMI definition (undefined
for zero-count pairs), we treat these as missing (NaN) rather than observed zeros.

Pipeline:
  1. Load PPMI matrix and set zeros to NaN
  2. Run 50 independent SRF fits (rank=91, 1000 outer x 20 inner)
  3. Hungarian alignment + consensus selection (AlignedConsensus)
  4. Compute R² on observed entries and per-dimension reliability
  5. Generate word clouds and Glasgow Norms prediction scatter plots

Usage:
    ./scripts/submit experiments/analyses/swow/missing_zeros/run.py --bg
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
from pysrf.model import _w_solver_backend
from src.colors import TEAL, ROSE, setup_style
from src.datasets.swow import load_swow_ppmi
from src.utils.figure_theme import despine

from experiments.analyses.swow.behavioral_prediction import evaluate_ridge_encoding
from experiments.analyses.swow.norms import PROPERTIES, load_behavioral_ratings

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"

PROPERTY_LABELS = {
    "arousal": "Arousal", "valence": "Valence", "dominance": "Dominance",
    "concreteness": "Concreteness", "imageability": "Imageability",
    "familiarity": "Familiarity", "aoa": "AoA", "size": "Size",
    "gender": "Gender",
}


def _load_ppmi():
    swow_dir = DATA_DIR / "small-world-of-words"
    ppmi, vocabulary, _ = load_swow_ppmi(
        swow_dir, use_all_responses=True, symmetrization="sum",
    )
    if issparse(ppmi):
        ppmi = np.array(ppmi.todense())
    return ppmi.astype(np.float64), vocabulary


def _dimension_reliability(aligned: np.ndarray) -> np.ndarray:
    n_runs, n_samples, rank = aligned.shape
    reliability = np.zeros(rank)
    for d in range(rank):
        r_pairs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                r = np.corrcoef(aligned[i, :, d], aligned[j, :, d])[0, 1]
                if np.isfinite(r):
                    r_pairs.append(r)
        if r_pairs:
            z = np.arctanh(np.clip(r_pairs, -0.999, 0.999))
            reliability[d] = float(np.tanh(np.mean(z)))
    return reliability


def _make_wordclouds(embedding, vocabulary, out_dir, n_dims=9):
    out_dir.mkdir(parents=True, exist_ok=True)
    var_per_dim = np.var(embedding, axis=0)
    dims = np.argsort(var_per_dim)[::-1][:n_dims]

    for dim_idx in dims:
        order = np.argsort(embedding[:, dim_idx])[::-1][:40]
        freqs = {vocabulary[i]: max(float(embedding[i, dim_idx]), 1e-6)
                 for i in order}
        wc = WordCloud(
            width=800, height=800, background_color="white",
            relative_scaling=0.4, min_font_size=8, max_font_size=200,
            prefer_horizontal=0.7, margin=5,
        )
        wc.generate_from_frequencies(freqs)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"Dim {dim_idx}", fontsize=10, pad=4)
        fig.savefig(out_dir / f"dim_{dim_idx:03d}.png", dpi=150,
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)
    log.info(f"Saved {n_dims} word clouds to {out_dir}")


def _predict_and_plot(embedding, vocabulary, ratings, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    results = []
    for prop in PROPERTIES:
        valid = ratings[["word", prop]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]
        if len(valid) < 50:
            continue

        indices = np.array([word_to_idx[w] for w in valid["word"]])
        x = embedding[indices]
        y = valid[prop].to_numpy()

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        model = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        preds = cross_val_predict(
            model, x_scaled, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        rho, _ = spearmanr(preds, y)
        results.append({"property": prop, "rho": rho, "n": len(y)})

        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-10)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(y_norm, p_norm, color=TEAL, alpha=0.18, s=1.0,
                   linewidths=0, rasterized=True)
        z = np.polyfit(y_norm, p_norm, 1)
        ax.plot(np.linspace(0, 1, 100), np.poly1d(z)(np.linspace(0, 1, 100)),
                color=ROSE, linewidth=0.8)
        ax.set_xlabel("Actual (normalized)", fontsize=7)
        ax.set_ylabel("Predicted (normalized)", fontsize=7)
        label = PROPERTY_LABELS.get(prop, prop)
        ax.set_title(f"{label}  $\\rho$ = {rho:.2f}", fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        despine(ax)
        fig.savefig(out_dir / f"scatter_{prop}.png", dpi=200,
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "norms_prediction.csv", index=False)
    return df


def run(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    log.info(f"BSUM implementation: {_w_solver_backend}")

    rank = cfg.srf.rank
    max_outer = cfg.srf.max_outer
    max_inner = cfg.srf.max_inner
    n_runs = cfg.generate.n_stable_runs
    n_jobs = cfg.common.n_jobs
    seed = cfg.common.random_state

    log.info("Loading PPMI and vocabulary...")
    ppmi, vocabulary = _load_ppmi()
    n = ppmi.shape[0]
    n_total = ppmi.size

    n_nan_orig = np.isnan(ppmi).sum()
    n_zero = ((ppmi == 0) & ~np.isnan(ppmi)).sum()
    n_positive = (ppmi > 0).sum()
    log.info(f"PPMI shape: {ppmi.shape}")
    log.info(f"  NaN (diagonal):      {n_nan_orig:>12,} ({n_nan_orig/n_total*100:.2f}%)")
    log.info(f"  Zero (unobserved):   {n_zero:>12,} ({n_zero/n_total*100:.2f}%)")
    log.info(f"  Positive (observed): {n_positive:>12,} ({n_positive/n_total*100:.2f}%)")

    ppmi[ppmi == 0] = np.nan
    n_observed = np.isfinite(ppmi).sum()
    log.info(f"After setting zeros to NaN: {n_observed:,} observed ({n_observed/n_total*100:.2f}%)")

    log.info(f"Running {n_runs} SRF fits (rank={rank}, "
             f"max_outer={max_outer}, max_inner={max_inner}, n_jobs={n_jobs})...")

    def _fit_one(run_seed):
        model = SRF(rank=rank, random_state=run_seed,
                     max_outer=max_outer, max_inner=max_inner)
        model.fit(ppmi)
        return model.w_, model.n_iter_, model.history_

    seeds = [seed + i for i in range(n_runs)]
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_fit_one)(s) for s in seeds
    )
    embeddings_list = [r[0] for r in results]
    n_iters = [r[1] for r in results]
    histories = [r[2] for r in results]

    stacked = np.hstack(embeddings_list)  # (n_samples, n_runs * rank)
    log.info(f"All {n_runs} fits complete. Stacked shape: {stacked.shape}")

    converged = [n < max_outer for n in n_iters]
    log.info(f"Iterations: min={min(n_iters)}, max={max(n_iters)}, "
             f"mean={np.mean(n_iters):.0f}, converged={sum(converged)}/{n_runs}")
    for i, (ni, h) in enumerate(zip(n_iters, histories)):
        final_evar = h["evar"][-1] if "evar" in h else float("nan")
        log.info(f"  Run {i:2d}: n_iter={ni:4d}, final_evar={final_evar:.4f}")

    consensus = AlignedConsensus(rank=rank, aggregation="select")
    consensus.fit(stacked)
    embedding = consensus.transform(stacked)

    log.info(f"Consensus embedding shape: {embedding.shape}")
    log.info(f"Selected run index: {consensus.selected_run_idx_}")

    recon = embedding @ embedding.T
    obs_mask = np.isfinite(ppmi)
    y_obs = ppmi[obs_mask]
    yhat_obs = recon[obs_mask]
    ss_res = np.sum((y_obs - yhat_obs) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r_pearson = np.corrcoef(y_obs, yhat_obs)[0, 1]
    sparsity = float((embedding < 1e-6).mean())

    log.info(f"R² on observed entries: {r2:.4f}")
    log.info(f"Pearson r on observed entries: {r_pearson:.4f}")
    log.info(f"Sparsity: {sparsity:.3f}")

    aligned = consensus.aligned_embeddings_
    reliability = _dimension_reliability(aligned)
    log.info(f"Dimension reliability: mean={reliability.mean():.3f}, "
             f"min={reliability.min():.3f}, max={reliability.max():.3f}")

    np.save(output_dir / "embedding.npy", embedding)
    np.save(output_dir / "runs.npy", aligned)
    np.save(output_dir / "reliability.npy", reliability)

    max_len = max(len(h.get("evar", [])) for h in histories)
    evar_curves = np.full((n_runs, max_len), np.nan)
    rec_error_curves = np.full((n_runs, max_len), np.nan)
    for i, h in enumerate(histories):
        evar = h.get("evar", [])
        rec = h.get("rec_error", [])
        evar_curves[i, :len(evar)] = evar
        rec_error_curves[i, :len(rec)] = rec
    np.savez(
        output_dir / "convergence.npz",
        evar=evar_curves,
        rec_error=rec_error_curves,
        n_iters=np.array(n_iters),
    )
    log.info(f"Saved convergence curves: {evar_curves.shape}")

    summary = {
        "dataset": "swow-missing",
        "n_samples": n,
        "rank": rank,
        "n_runs": n_runs,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "n_observed": int(n_observed),
        "pct_observed": float(n_observed / n_total * 100),
        "r2_observed": float(r2),
        "pearson_r_observed": float(r_pearson),
        "sparsity": sparsity,
        "selected_run_idx": int(consensus.selected_run_idx_),
        "mean_reliability": float(reliability.mean()),
        "min_reliability": float(reliability.min()),
        "max_reliability": float(reliability.max()),
        "n_iters_min": int(min(n_iters)),
        "n_iters_max": int(max(n_iters)),
        "n_iters_mean": float(np.mean(n_iters)),
        "n_converged": int(sum(converged)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("Generating word clouds...")
    _make_wordclouds(embedding, vocabulary, output_dir / "wordclouds")

    log.info("Predicting Glasgow Norms...")
    ratings = load_behavioral_ratings(DATA_DIR)
    norms_df = _predict_and_plot(embedding, vocabulary, ratings, output_dir / "norms")
    log.info(f"\n{norms_df.to_string(index=False)}")

    log.info(f"All outputs saved to {output_dir}")
