"""
Explore behavioral prediction improvements for word_association experiment.

This sandbox investigates:
1. Word overlap between embeddings and ratings
2. Potential ceiling effects
3. Different regression approaches

Usage:
    poetry run python sandbox/word_association_debug/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/experiments/word_association/generate_embedding"
OUTPUT_DIR = get_output_dir()


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load and merge all behavioral ratings."""
    # Size from THINGS
    size_df = pd.read_csv(data_dir / "things/size_ratings.csv")
    size_df = size_df.rename(columns={"Word": "word", "Size_mean": "size"})
    size_df["word"] = size_df["word"].str.lower()
    size_df = size_df[["word", "size"]]

    # Concreteness from Brysbaert
    conc_df = pd.read_excel(data_dir / "concreteness_ratings_brysbaert.xlsx")
    conc_df = conc_df.rename(columns={"Word": "word", "Conc.M": "concreteness"})
    conc_df["word"] = conc_df["word"].str.lower()
    conc_df = conc_df[["word", "concreteness"]]

    # Property ratings from THINGS
    props_df = pd.read_csv(data_dir / "things/things_property_ratings.csv")
    props_df = props_df.rename(columns={
        "Word": "word",
        "pleasant_mean": "valence",
        "heavy_mean": "heaviness",
        "lives_mean": "animacy",
    })
    props_df["word"] = props_df["word"].str.lower()
    props_df = props_df[["word", "valence", "heaviness", "animacy"]]

    # Merge all
    merged = size_df.merge(conc_df, on="word", how="outer")
    merged = merged.merge(props_df, on="word", how="outer")

    return merged


def analyze_word_overlap(embeddings_vocab: list[str], ratings: pd.DataFrame, word_to_idx: dict, data_dir: Path) -> dict:
    """Analyze overlap between embedding vocabulary and rating datasets."""
    emb_words = set(w.lower() for w in embeddings_vocab)

    print("\n=== Word Overlap Analysis ===")
    print(f"Embedding vocabulary size: {len(emb_words)}")

    # Load THINGS with uniqueID to count objects properly
    things_full = pd.read_csv(data_dir / "things/size_ratings.csv")
    n_objects = len(things_full)  # 1854 unique objects
    n_unique_words = things_full["Word"].str.lower().nunique()

    # Count objects that have matching word in vocabulary
    things_full["word_lower"] = things_full["Word"].str.lower()
    things_matched = things_full[things_full["word_lower"].isin(emb_words)]
    n_matched_objects = len(things_matched)
    n_matched_words = things_matched["word_lower"].nunique()

    print(f"\nTHINGS:")
    print(f"  Total objects (uniqueID): {n_objects}")
    print(f"  Unique word forms: {n_unique_words}")
    print(f"  Objects with word in vocab: {n_matched_objects} ({100*n_matched_objects/n_objects:.1f}%)")
    print(f"  Unique words matched: {n_matched_words}")

    # Sample missing
    missing_words = set(things_full["word_lower"]) - emb_words
    print(f"  Missing words ({len(missing_words)}): {list(missing_words)[:8]}")

    # Concreteness
    conc_df = pd.read_excel(data_dir / "concreteness_ratings_brysbaert.xlsx")
    conc_df["word_lower"] = conc_df["Word"].str.lower()
    n_conc = len(conc_df)
    n_conc_matched = conc_df["word_lower"].isin(emb_words).sum()

    print(f"\nConcreteness:")
    print(f"  Total words: {n_conc}")
    print(f"  Words in vocab: {n_conc_matched} ({100*n_conc_matched/n_conc:.1f}%)")

    # Actual training data counts
    print(f"\n=== Actual Training Data ===")
    dimensions = ["size", "concreteness", "valence", "heaviness", "animacy"]
    for dim in dimensions:
        valid = ratings[["word", dim]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]
        print(f"  {dim:15s}: n={len(valid)}")

    return {}


def evaluate_with_method(
    embeddings: np.ndarray,
    word_indices: np.ndarray,
    ratings: np.ndarray,
    model_name: str,
    n_folds: int = 5,
) -> dict:
    """Evaluate a single model with cross-validation."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    X = embeddings[word_indices]
    y = ratings

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
    elif model_name == "elasticnet":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ElasticNetCV(cv=3, max_iter=10000, l1_ratio=[0.1, 0.5, 0.9]))
        ])
    elif model_name == "gbr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            ))
        ])
    elif model_name == "rf":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ))
        ])
    elif model_name == "svr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf", C=1.0))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=cv)

    r, p = spearmanr(predictions, y)
    return {"correlation": r, "pvalue": p, "n_samples": len(y)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data"
    ratings = load_ratings(data_dir)

    print("=== Ratings loaded ===")
    print(f"Total rows: {len(ratings)}")
    for col in ["size", "concreteness", "valence", "heaviness", "animacy"]:
        print(f"  {col}: {ratings[col].notna().sum()} non-null")

    dimensions = ["size", "concreteness", "valence", "heaviness", "animacy"]
    models = ["ridge", "lasso", "elasticnet"]

    results = []

    # Load rank 50 embedding
    emb_path = EMBEDDING_DIR / "50"
    embeddings = np.load(emb_path / "word_embedding.npy")
    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)

    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    # Analyze overlap
    analyze_word_overlap(vocabulary, ratings, word_to_idx, data_dir)

    print(f"\n=== Rank 50 Evaluation ===")
    print(f"Embedding shape: {embeddings.shape}")

    for dimension in dimensions:
        valid = ratings[["word", dimension]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]

        if len(valid) < 50:
            print(f"Skipping {dimension}: insufficient data ({len(valid)})")
            continue

        word_indices = np.array([word_to_idx[w] for w in valid["word"]])
        rating_values = valid[dimension].to_numpy()

        print(f"\n  {dimension} (n={len(valid)}):")
        for model_name in models:
            result = evaluate_with_method(
                embeddings, word_indices, rating_values, model_name
            )
            results.append({
                "rank": 50,
                "dimension": dimension,
                "model": model_name,
                **result
            })
            print(f"    {model_name:12s}: r={result['correlation']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "method_comparison.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / 'method_comparison.csv'}")


if __name__ == "__main__":
    main()
