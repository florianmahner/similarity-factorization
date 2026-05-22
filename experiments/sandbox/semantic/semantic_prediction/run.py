"""
Predict semantic ratings from SWOW SRF embeddings.

Analyzes how well word association structure captures various semantic dimensions:
- Warriner VAD: valence, arousal, dominance
- Lancaster Sensorimotor: 6 perceptual + 5 action modalities
- Glasgow: arousal, valence, dominance, concreteness, imageability, familiarity, AoA, size, gender
- Brysbaert: concreteness
- THINGS: various object properties

Usage:
    poetry run python sandbox/semantic_prediction/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/experiments/word_association/generate_embedding"
NORMS_DIR = PROJECT_ROOT / "data/semantic_norms"
OUTPUT_DIR = get_output_dir()


def load_embedding(rank: int = 50) -> tuple[np.ndarray, dict[str, int]]:
    """Load SWOW embedding and build word-to-index mapping."""
    emb_path = EMBEDDING_DIR / str(rank)
    embeddings = np.load(emb_path / "word_embedding.npy")

    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)

    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    return embeddings, word_to_idx


def load_warriner() -> pd.DataFrame:
    """Load Warriner VAD norms."""
    df = pd.read_csv(NORMS_DIR / "warriner_vad.csv")
    df = df.rename(columns={
        "Word": "word",
        "V.Mean.Sum": "valence",
        "A.Mean.Sum": "arousal",
        "D.Mean.Sum": "dominance",
    })
    df["word"] = df["word"].str.lower()
    return df[["word", "valence", "arousal", "dominance"]].dropna()


def load_lancaster() -> pd.DataFrame:
    """Load Lancaster Sensorimotor norms."""
    df = pd.read_csv(NORMS_DIR / "lancaster_sensorimotor.csv")
    df = df.rename(columns={"Word": "word"})
    df["word"] = df["word"].str.lower()

    # Select mean columns for each modality
    cols = ["word"]
    modalities = [
        "Auditory.mean", "Gustatory.mean", "Haptic.mean",
        "Interoceptive.mean", "Olfactory.mean", "Visual.mean",
        "Foot_leg.mean", "Hand_arm.mean", "Head.mean",
        "Mouth.mean", "Torso.mean"
    ]

    rename_map = {
        "Auditory.mean": "auditory",
        "Gustatory.mean": "gustatory",
        "Haptic.mean": "haptic",
        "Interoceptive.mean": "interoceptive",
        "Olfactory.mean": "olfactory",
        "Visual.mean": "visual",
        "Foot_leg.mean": "foot_leg",
        "Hand_arm.mean": "hand_arm",
        "Head.mean": "head",
        "Mouth.mean": "mouth",
        "Torso.mean": "torso",
    }

    df = df[["word"] + modalities].rename(columns=rename_map)
    return df.dropna()


def load_glasgow() -> pd.DataFrame:
    """Load Glasgow norms."""
    df = pd.read_csv(NORMS_DIR / "glasgow_norms.csv")
    df["word"] = df["word"].str.lower()

    cols = ["word", "arousal", "valence", "dominance", "concreteness",
            "imageability", "familiarity", "aoa", "semsize", "gender"]

    return df[cols].dropna()


def load_brysbaert() -> pd.DataFrame:
    """Load Brysbaert concreteness ratings."""
    df = pd.read_excel(PROJECT_ROOT / "data/concreteness_ratings_brysbaert.xlsx")
    df = df.rename(columns={"Word": "word", "Conc.M": "concreteness"})
    df["word"] = df["word"].str.lower()
    return df[["word", "concreteness"]].dropna()


def load_things() -> pd.DataFrame:
    """Load THINGS property ratings."""
    props = pd.read_csv(PROJECT_ROOT / "data/things/things_property_ratings.csv")
    size = pd.read_csv(PROJECT_ROOT / "data/things/size_ratings.csv")

    props = props.rename(columns={
        "Word": "word",
        "manmade_mean": "manmade",
        "precious_mean": "precious",
        "lives_mean": "animacy",
        "heavy_mean": "heaviness",
        "natural_mean": "natural",
        "moves_mean": "moves",
        "grasp_mean": "graspable",
        "hold_mean": "holdable",
        "be.moved_mean": "moveable",
        "pleasant_mean": "pleasant",
    })
    props["word"] = props["word"].str.lower()

    size = size.rename(columns={"Word": "word", "Size_mean": "size"})
    size["word"] = size["word"].str.lower()

    df = props.merge(size[["word", "size"]], on="word", how="outer")

    cols = ["word", "manmade", "precious", "animacy", "heaviness", "natural",
            "moves", "graspable", "holdable", "moveable", "pleasant", "size"]

    return df[cols].dropna(subset=["word"])


def predict_dimension(
    embeddings: np.ndarray,
    word_to_idx: dict[str, int],
    df: pd.DataFrame,
    dimension: str,
    n_folds: int = 5,
) -> dict | None:
    """Predict a single dimension using cross-validated Ridge regression."""
    valid = df[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]

    if len(valid) < 30:
        return None

    word_indices = np.array([word_to_idx[w] for w in valid["word"]])
    X = embeddings[word_indices]
    y = valid[dimension].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
    ])

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

    # Load embedding
    print("Loading SWOW embedding (rank 50)...")
    embeddings, word_to_idx = load_embedding(rank=50)
    print(f"Embedding: {embeddings.shape[0]} words × {embeddings.shape[1]} dims")

    # Load all norms
    print("\nLoading semantic norms...")
    datasets = {
        "warriner": load_warriner(),
        "lancaster": load_lancaster(),
        "glasgow": load_glasgow(),
        "brysbaert": load_brysbaert(),
        "things": load_things(),
    }

    for name, df in datasets.items():
        n_words = len(df)
        n_overlap = df["word"].isin(word_to_idx).sum()
        dims = [c for c in df.columns if c != "word"]
        print(f"  {name}: {n_words} words, {n_overlap} overlap ({100*n_overlap/n_words:.1f}%), {len(dims)} dims")

    # Run predictions
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    all_results = []
    best_predictions = {}

    for dataset_name, df in datasets.items():
        dims = [c for c in df.columns if c != "word"]
        print(f"\n{dataset_name.upper()} ({len(dims)} dimensions):")

        for dim in dims:
            result = predict_dimension(embeddings, word_to_idx, df, dim)

            if result is None:
                print(f"  {dim}: insufficient data")
                continue

            all_results.append({
                "dataset": dataset_name,
                "dimension": dim,
                "correlation": result["correlation"],
                "pvalue": result["pvalue"],
                "n_samples": result["n_samples"],
            })

            # Store predictions for plotting
            key = f"{dataset_name}_{dim}"
            best_predictions[key] = result

            sig = "***" if result["pvalue"] < 0.001 else "**" if result["pvalue"] < 0.01 else "*" if result["pvalue"] < 0.05 else ""
            print(f"  {dim}: r={result['correlation']:.3f}{sig} (n={result['n_samples']})")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY BY DATASET")
    print("="*60)
    summary = results_df.groupby("dataset")["correlation"].agg(["mean", "std", "min", "max", "count"])
    print(summary.round(3).to_string())

    print("\n" + "="*60)
    print("TOP 15 BEST PREDICTED DIMENSIONS")
    print("="*60)
    top = results_df.nlargest(15, "correlation")
    print(top[["dataset", "dimension", "correlation", "n_samples"]].to_string(index=False))

    print(f"\nOverall: mean r = {results_df['correlation'].mean():.3f}, "
          f"range [{results_df['correlation'].min():.3f}, {results_df['correlation'].max():.3f}]")

    # Save predictions for plotting
    np.savez(
        OUTPUT_DIR / "predictions.npz",
        **{k: {"true": v["true_values"], "pred": v["predictions"], "words": v["words"]}
           for k, v in best_predictions.items()}
    )

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
