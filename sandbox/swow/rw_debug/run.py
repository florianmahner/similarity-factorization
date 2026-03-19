"""Debug: Explore RW intermediate representations for SRF compatibility.

The key question: cosine similarity is normalized, so S ~ WW^T is a bad fit.
What intermediate representation from the RW pipeline works best with SRF?

Candidates:
1. cosine: Final cosine similarity (current -- forces ||w_i|| ~ 1)
2. katz_ppmi_sym: Symmetrized PPMI-Katz matrix (sparse, non-negative, unnormalized)
3. ppmi_local: Standard PPMI on symmetrized counts (baseline)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pysrf import SRF
from sklearn.metrics.pairwise import cosine_similarity

from src.datasets.swow import (
    _katz_walk,
    _l1_normalize,
    _ppmi_rw,
    _select_vocabulary,
    compute_ppmi,
    load_swow_data,
)
from src.utils import get_output_dir
from utils.graphs import build_directed_graph, clean_graph, graph_to_matrix, symmetrize_matrix

OUTPUT_DIR = get_output_dir()

RANK = 20
N_TOP = 50
MAX_OUTER = 30


def _flush(msg):
    print(msg)
    sys.stdout.flush()


def compute_rw_stages(counts, alpha=0.75):
    """Return intermediate stages of the RW pipeline."""
    p = _l1_normalize(counts)
    p = _ppmi_rw(p)
    p = _l1_normalize(p)
    _flush("  Katz walk...")
    katz_raw = _katz_walk(p, alpha)
    katz_ppmi = _ppmi_rw(katz_raw)
    katz_ppmi_l1 = _l1_normalize(katz_ppmi)
    s_cos = cosine_similarity(katz_ppmi_l1)
    return {
        "katz_ppmi": katz_ppmi,
        "cosine": s_cos,
    }


def describe_matrix(name, m):
    obs = m[~np.isnan(m)]
    diag = np.diag(m)
    _flush(f"\n{'='*60}")
    _flush(f"{name}: shape={m.shape}")
    _flush(f"  Range: [{obs.min():.4f}, {obs.max():.4f}]")
    _flush(f"  Mean: {obs.mean():.4f}, Std: {obs.std():.4f}")
    _flush(f"  Diagonal mean: {np.nanmean(diag):.4f}")
    _flush(f"  NaN frac: {np.isnan(m).sum() / m.size:.4f}")
    _flush(f"  Sparsity (==0): {(obs == 0).sum() / obs.size:.4f}")
    _flush(f"  Symmetric: {np.allclose(np.nan_to_num(m), np.nan_to_num(m.T), atol=1e-6)}")


def try_factorize(name, s, rank, vocab):
    _flush(f"\n--- Factorizing {name} at rank={rank} ---")

    np.fill_diagonal(s, np.nan)

    model = SRF(rank=rank, max_outer=MAX_OUTER, verbose=1)
    model.fit(s)

    w = model.w_
    s_hat = w @ w.T
    mask = ~np.isnan(s)
    ss_tot = np.var(s[mask]) * mask.sum()
    ss_res = np.sum((s[mask] - s_hat[mask]) ** 2)
    evar = 1 - ss_res / ss_tot

    _flush(f"  Explained variance: {evar:.4f}")

    dim_variance = np.sum(w ** 2, axis=0)
    dim_order = np.argsort(dim_variance)[::-1]

    records = []
    for idx, d in enumerate(dim_order):
        top_idx = np.argsort(w[:, d])[::-1][:N_TOP]
        top_words = [vocab[i] for i in top_idx]
        top_weights = [w[i, d] for i in top_idx]

        if idx < 10:
            label = ", ".join(top_words[:8])
            _flush(f"  Dim {idx} (var={dim_variance[d]:.4f}): {label}")

        for i, (word, weight) in enumerate(zip(top_words, top_weights)):
            records.append({
                "method": name,
                "dim_sorted": idx,
                "variance": dim_variance[d],
                "word_rank": i,
                "word": word,
                "loading": weight,
            })

    return w, evar, pd.DataFrame(records)


def main():
    _flush("Loading SWOW data...")
    data_dir = Path("data/small-world-of-words")
    df = load_swow_data(data_dir, use_all_responses=True)

    g = build_directed_graph(
        df["cue"].tolist(), df["response"].tolist(), df["count"].tolist()
    )
    g = clean_graph(g)
    vocab = _select_vocabulary(g, None)
    subgraph = g.subgraph(vocab).copy()
    counts = graph_to_matrix(subgraph, vocab)
    vocab = np.array(vocab)

    _flush(f"Vocabulary: {len(vocab)} words")

    _flush("\nComputing RW pipeline stages...")
    stages = compute_rw_stages(counts, alpha=0.75)

    candidates = {}

    # 1. Cosine similarity (current, problematic)
    candidates["cosine"] = stages["cosine"]

    # 2. The PPMI-Katz matrix itself, symmetrized (most promising)
    g_ppmi = stages["katz_ppmi"]
    candidates["katz_ppmi_sym"] = np.maximum((g_ppmi + g_ppmi.T) / 2, 0)

    # 3. PPMI on symmetrized counts (baseline)
    counts_sym = symmetrize_matrix(counts, method="sum")
    candidates["ppmi_local"] = compute_ppmi(counts_sym, negative_as_nan=False)

    for name, s in candidates.items():
        describe_matrix(name, s)

    all_results = []
    evars = {}
    for name, s in candidates.items():
        w, evar, result_df = try_factorize(name, s.copy(), RANK, vocab)
        all_results.append(result_df)
        evars[name] = evar
        np.save(OUTPUT_DIR / f"embedding_{name}.npy", w)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "top_words.csv", index=False)

    _flush("\n" + "=" * 60)
    _flush("SUMMARY: Explained variance at rank=20")
    for name, evar in evars.items():
        _flush(f"  {name:20s}: {evar:.4f}")

    from experiments.swow.plotting import plot_word_clouds_grid
    for name in candidates:
        method_df = combined[combined["method"] == name].copy()
        method_df = method_df.rename(columns={"dim_sorted": "dimension"})
        plot_word_clouds_grid(
            method_df,
            OUTPUT_DIR / f"wordclouds_{name}.png",
            n_words=N_TOP,
        )
        _flush(f"Created: wordclouds_{name}.png")

    np.save(OUTPUT_DIR / "vocabulary.npy", vocab)
    _flush(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
