"""
Explore behavioral prediction for GRAND dataset using SWOW embeddings.

This sandbox investigates:
1. Word overlap between SWOW embeddings and GRAND ratings
2. Prediction performance across 9 domains and 56 dimensions
3. Comparison of regression methods

Usage:
    poetry run python sandbox/grand_prediction_debug/run.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/experiments/word_association/generate_embedding"
GRAND_DIR = PROJECT_ROOT / "data/grand_natbehav/extracted"
OUTPUT_DIR = Path(os.environ.get("SANDBOX_OUTPUT_DIR", Path(__file__).parent / "outputs" / "dev"))


def load_embedding(rank: int = 50) -> tuple[np.ndarray, dict[str, int]]:
    """Load SWOW embedding and build word-to-index mapping."""
    emb_path = EMBEDDING_DIR / str(rank)
    embeddings = np.load(emb_path / "word_embedding.npy")

    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)

    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    return embeddings, word_to_idx


def load_grand_ratings() -> dict[str, pd.DataFrame]:
    """Load all GRAND domain ratings."""
    domains = {}
    for csv_path in GRAND_DIR.glob("*.csv"):
        domain = csv_path.stem
        df = pd.read_csv(csv_path)
        df["word"] = df["word"].str.lower()
        domains[domain] = df
    return domains


def analyze_overlap(
    word_to_idx: dict[str, int],
    domains: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """Analyze word overlap between embedding and each domain."""
    emb_vocab = set(word_to_idx.keys())

    print("=== Word Overlap Analysis ===")
    print(f"Embedding vocabulary size: {len(emb_vocab)}")

    overlap_stats = {}
    for domain, df in sorted(domains.items()):
        domain_words = set(df["word"].str.lower())
        overlap = domain_words & emb_vocab
        missing = domain_words - emb_vocab

        overlap_stats[domain] = {
            "total": len(domain_words),
            "overlap": len(overlap),
            "pct": 100 * len(overlap) / len(domain_words),
            "missing": list(missing)[:5],
        }

        print(f"\n{domain}:")
        print(f"  Total items: {len(domain_words)}")
        print(f"  In vocabulary: {len(overlap)} ({100*len(overlap)/len(domain_words):.1f}%)")
        if missing:
            print(f"  Missing ({len(missing)}): {list(missing)[:5]}...")

    return overlap_stats


def evaluate_prediction(
    embeddings: np.ndarray,
    word_to_idx: dict[str, int],
    df: pd.DataFrame,
    dimension: str,
    model_name: str = "ridge",
    n_folds: int = 5,
) -> dict:
    """Evaluate prediction for a single dimension using cross-validation."""
    # Get words that are in vocabulary
    valid = df[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]

    if len(valid) < 20:
        return None

    word_indices = np.array([word_to_idx[w] for w in valid["word"]])
    X = embeddings[word_indices]
    y = valid[dimension].to_numpy()

    if model_name == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
        ])
    elif model_name == "lasso":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LassoCV(cv=3, max_iter=10000))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=cv)

    r, p = spearmanr(predictions, y)
    return {
        "correlation": r,
        "pvalue": p,
        "n_samples": len(y),
        "predictions": predictions,
        "true_values": y,
        "words": valid["word"].tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading embedding...")
    embeddings, word_to_idx = load_embedding(rank=50)
    print(f"Embedding shape: {embeddings.shape}")

    print("\nLoading GRAND ratings...")
    domains = load_grand_ratings()
    print(f"Loaded {len(domains)} domains")

    # Analyze overlap
    overlap_stats = analyze_overlap(word_to_idx, domains)

    # Evaluate predictions
    print("\n\n=== Prediction Results ===")
    results = []

    for domain, df in sorted(domains.items()):
        dims = [c for c in df.columns if c != "word"]
        print(f"\n{domain} ({len(dims)} dimensions):")

        for dim in dims:
            result = evaluate_prediction(
                embeddings, word_to_idx, df, dim, model_name="ridge"
            )

            if result is None:
                print(f"  {dim}: insufficient data")
                continue

            results.append({
                "domain": domain,
                "dimension": dim,
                "correlation": result["correlation"],
                "pvalue": result["pvalue"],
                "n_samples": result["n_samples"],
            })

            print(f"  {dim}: r={result['correlation']:.3f} (n={result['n_samples']})")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

    # Summary statistics
    print("\n\n=== Summary by Domain ===")
    summary = results_df.groupby("domain")["correlation"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print("\n\n=== Top 10 Best Predicted Dimensions ===")
    top10 = results_df.nlargest(10, "correlation")
    print(top10[["domain", "dimension", "correlation", "n_samples"]].to_string(index=False))

    print("\n\n=== Bottom 10 Worst Predicted Dimensions ===")
    bottom10 = results_df.nsmallest(10, "correlation")
    print(bottom10[["domain", "dimension", "correlation", "n_samples"]].to_string(index=False))

    # Overall stats
    print(f"\n\nOverall mean correlation: {results_df['correlation'].mean():.3f}")
    print(f"Overall std: {results_df['correlation'].std():.3f}")
    print(f"Range: [{results_df['correlation'].min():.3f}, {results_df['correlation'].max():.3f}]")

    print(f"\nResults saved to {OUTPUT_DIR / 'prediction_results.csv'}")


if __name__ == "__main__":
    main()
