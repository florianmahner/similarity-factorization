"""SWOW factorization: treat PPMI zeros as missing (NaN) and evaluate.

The SWOW PPMI matrix already contains NaN for word pairs with no association
data. The non-zero PPMI entries are genuine observed values, including zeros
(which mean "co-occurs, but not above chance"). Here we test what happens
if we additionally treat those PPMI=0 entries as missing, leaving only the
positive associations as observed.

Pipeline:
  1. Load PPMI (already has NaN for unobserved pairs)
  2. Set PPMI==0 entries to NaN (treat as missing)
  3. Fit SRF with rank=91, max_outer=100, max_inner=30
  4. Compute R² on observed entries via model.score()
  5. Generate word clouds (actual WordCloud, not bar charts)
  6. Predict Glasgow Norms via cross-validated ridge regression
  7. Plot predicted vs actual scatter for each property
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from pysrf import SRF
from pysrf.model import _update_w_source
from src.colors import TEAL, ROSE, GRAY, GRAY_DARK, GRAY_LIGHT, soft, setup_style
from src.datasets.swow import load_swow_ppmi
from src.utils import get_output_dir
from src.utils.figure_theme import despine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

OUTPUT_DIR = get_output_dir()

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
CONSENSUS_DIR = (
    Path(__file__).resolve().parents[3]
    / "datasets" / "consensus" / "outputs" / "swow"
)
NORMS_PATH = DATA_DIR / "semantic_norms" / "glasgow_norms.csv"

RANK = 91
MAX_OUTER = 100
MAX_INNER = 30
PROPERTIES = [
    "arousal", "valence", "dominance", "concreteness",
    "imageability", "familiarity", "aoa", "size", "gender",
]
PROPERTY_LABELS = {
    "arousal": "Arousal", "valence": "Valence", "dominance": "Dominance",
    "concreteness": "Concreteness", "imageability": "Imageability",
    "familiarity": "Familiarity", "aoa": "AoA", "size": "Size",
    "gender": "Gender",
}


def load_ppmi_and_vocab():
    swow_dir = DATA_DIR / "small-world-of-words"
    ppmi, vocabulary, _ = load_swow_ppmi(swow_dir, use_all_responses=True)
    if issparse(ppmi):
        ppmi = np.array(ppmi.todense())
    return ppmi.astype(np.float64), vocabulary


def load_glasgow_norms() -> pd.DataFrame:
    norms = pd.read_csv(NORMS_PATH)
    norms = norms.rename(columns={"semsize": "size"})
    norms["word"] = norms["word"].str.lower()
    return norms[["word"] + PROPERTIES]


def make_wordclouds(embedding, vocabulary, out_dir, n_dims=9):
    """Generate actual word clouds using the WordCloud library."""
    out_dir.mkdir(parents=True, exist_ok=True)

    var_per_dim = np.var(embedding, axis=0)
    sorted_dims = np.argsort(var_per_dim)[::-1]
    dims = sorted_dims[:n_dims]

    for dim_idx in dims:
        top_k = 40
        order = np.argsort(embedding[:, dim_idx])[::-1][:top_k]
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


def predict_and_plot(embedding, vocabulary, norms, out_dir):
    """Cross-validated ridge regression + scatter plots for each property."""
    out_dir.mkdir(parents=True, exist_ok=True)
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    setup_style()
    results = []

    for prop in PROPERTIES:
        valid = norms[["word", prop]].dropna()
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

        # Normalize to [0,1] for plotting
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-10)

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(y_norm, p_norm, color=TEAL, alpha=0.18, s=1.0,
                   linewidths=0, rasterized=True)
        z = np.polyfit(y_norm, p_norm, 1)
        xline = np.linspace(0, 1, 100)
        ax.plot(xline, np.poly1d(z)(xline), color=ROSE, linewidth=0.8)
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
    log.info(f"\n{df.to_string(index=False)}")
    return df


def main():
    log.info(f"BSUM implementation: {_update_w_source}")

    log.info("Loading PPMI and vocabulary...")
    ppmi, vocabulary = load_ppmi_and_vocab()
    norms = load_glasgow_norms()

    n_total = ppmi.size
    n_nan_original = np.isnan(ppmi).sum()
    n_zero = ((ppmi == 0) & ~np.isnan(ppmi)).sum()
    n_positive = (ppmi > 0).sum()

    log.info(f"PPMI shape: {ppmi.shape}")
    log.info(f"  NaN (unobserved):    {n_nan_original:>12,} ({n_nan_original/n_total*100:.1f}%)")
    log.info(f"  Zero (no excess):    {n_zero:>12,} ({n_zero/n_total*100:.1f}%)")
    log.info(f"  Positive (assoc.):   {n_positive:>12,} ({n_positive/n_total*100:.1f}%)")

    # Set PPMI==0 to NaN (treat as missing)
    ppmi_missing = ppmi.copy()
    ppmi_missing[ppmi_missing == 0] = np.nan
    n_observed = np.isfinite(ppmi_missing).sum()
    log.info(f"\nAfter setting zeros to NaN:")
    log.info(f"  Observed entries:    {n_observed:>12,} ({n_observed/n_total*100:.1f}%)")

    # Fit SRF
    log.info(f"\nFitting SRF (rank={RANK}, max_outer={MAX_OUTER}, max_inner={MAX_INNER})...")
    model = SRF(
        rank=RANK, random_state=42,
        max_outer=MAX_OUTER, max_inner=MAX_INNER,
        missing_values=np.nan, verbose=1,
    )
    model.fit(ppmi_missing)
    w = model.w_
    log.info(f"Embedding shape: {w.shape}, converged in {model.n_iter_} iters")

    # R² on observed entries
    r2_score = -model.score(ppmi_missing)  # score returns -MSE
    obs_mask = np.isfinite(ppmi_missing)
    recon = w @ w.T
    y_obs = ppmi_missing[obs_mask]
    yhat_obs = recon[obs_mask]
    ss_res = np.sum((y_obs - yhat_obs) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r_pearson = np.corrcoef(y_obs, yhat_obs)[0, 1]
    log.info(f"\nReconstruction on observed entries:")
    log.info(f"  R² (coeff. of determination): {r2:.4f}")
    log.info(f"  Pearson r: {r_pearson:.4f}")
    log.info(f"  MSE: {r2_score:.6f}")

    # Save embedding
    np.save(OUTPUT_DIR / "embedding.npy", w)

    # Word clouds
    log.info("\nGenerating word clouds...")
    make_wordclouds(w, vocabulary, OUTPUT_DIR / "wordclouds")

    # Glasgow Norms prediction + scatter plots
    log.info("\nPredicting Glasgow Norms...")
    predict_and_plot(w, vocabulary, norms, OUTPUT_DIR / "norms")

    log.info(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
