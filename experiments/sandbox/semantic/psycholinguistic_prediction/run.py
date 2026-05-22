"""Predict psycholinguistic norms from SWOW embeddings."""

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
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = get_output_dir()


def load_embeddings():
    """Load SWOW word embeddings."""
    emb_path = EMBEDDING_DIR / "50"
    embeddings = np.load(emb_path / "word_embedding.npy")
    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)
    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}
    return embeddings, word_to_idx


def predict_norms(embeddings, word_to_idx, words, ratings):
    """Get cross-validated predictions using Ridge regression."""
    valid_mask = [w in word_to_idx for w in words]
    valid_words = [w for w, v in zip(words, valid_mask) if v]
    valid_ratings = ratings[valid_mask]

    if len(valid_words) < 20:
        return None

    word_indices = np.array([word_to_idx[w] for w in valid_words])
    X = embeddings[word_indices]
    y = valid_ratings

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=cv)
    r, p = spearmanr(predictions, y)

    return {
        "predictions": predictions,
        "true_values": y,
        "words": valid_words,
        "correlation": r,
        "pvalue": p,
        "n": len(y),
    }


def predict_similarity(embeddings, word_to_idx, word1_list, word2_list, scores):
    """Predict word pair similarity using cosine similarity.

    Note: Cosine is preferred over dot product because PPMI-based embeddings
    have norms that scale with word frequency. Cosine normalizes this out.
    """
    valid_mask = [(w1.lower() in word_to_idx and w2.lower() in word_to_idx)
                  for w1, w2 in zip(word1_list, word2_list)]

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if len(valid_indices) < 20:
        return None

    emb_sims = []
    human_scores = []
    for i in valid_indices:
        w1, w2 = word1_list[i].lower(), word2_list[i].lower()
        e1 = embeddings[word_to_idx[w1]]
        e2 = embeddings[word_to_idx[w2]]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        emb_sims.append(cos_sim)
        human_scores.append(scores[i])

    emb_sims = np.array(emb_sims)
    human_scores = np.array(human_scores)
    r, p = spearmanr(emb_sims, human_scores)

    return {
        "embedding_similarity": emb_sims,
        "human_scores": human_scores,
        "correlation": r,
        "pvalue": p,
        "n": len(emb_sims),
    }


def analyze_glasgow_norms(embeddings, word_to_idx):
    """Analyze Glasgow Norms prediction."""
    df = pd.read_csv(DATA_DIR / "glasgow_norms.csv")
    df["word"] = df["word"].str.lower()

    dims = {
        "AROU": "Arousal",
        "VAL": "Valence",
        "DOM": "Dominance",
        "CNC": "Concreteness",
        "IMAG": "Imageability",
        "FAM": "Familiarity",
        "AOA": "Age of Acquisition",
        "SIZE": "Size",
        "GEND": "Gender",
    }

    results = []
    for col, name in dims.items():
        result = predict_norms(
            embeddings, word_to_idx,
            df["word"].tolist(),
            df[col].to_numpy()
        )
        if result:
            results.append({
                "dataset": "Glasgow Norms",
                "dimension": name,
                "correlation": result["correlation"],
                "pvalue": result["pvalue"],
                "n": result["n"],
            })
            print(f"Glasgow {name}: ρ = {result['correlation']:.3f} (n={result['n']})")

    return results


def analyze_simlex(embeddings, word_to_idx):
    """Analyze SimLex-999 similarity prediction."""
    df = pd.read_csv(DATA_DIR / "SimLex-999/SimLex-999.txt", sep="\t")

    result = predict_similarity(
        embeddings, word_to_idx,
        df["word1"].tolist(),
        df["word2"].tolist(),
        df["SimLex999"].to_numpy()
    )

    if result:
        print(f"SimLex-999: ρ = {result['correlation']:.3f} (n={result['n']})")
        return [{
            "dataset": "SimLex-999",
            "dimension": "Similarity",
            "correlation": result["correlation"],
            "pvalue": result["pvalue"],
            "n": result["n"],
        }], result
    return [], None


def analyze_wordsim(embeddings, word_to_idx):
    """Analyze WordSim-353 relatedness prediction."""
    df = pd.read_csv(DATA_DIR / "wordsim353.csv", sep="\t", header=None,
                     names=["word1", "word2", "score"])

    result = predict_similarity(
        embeddings, word_to_idx,
        df["word1"].tolist(),
        df["word2"].tolist(),
        df["score"].to_numpy()
    )

    if result:
        print(f"WordSim-353: ρ = {result['correlation']:.3f} (n={result['n']})")
        return [{
            "dataset": "WordSim-353",
            "dimension": "Relatedness",
            "correlation": result["correlation"],
            "pvalue": result["pvalue"],
            "n": result["n"],
        }], result
    return [], None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, word_to_idx = load_embeddings()

    print(f"Loaded embeddings: {embeddings.shape[0]} words, {embeddings.shape[1]} dimensions")
    print(f"\n{'='*60}")
    print("Glasgow Norms (Ridge regression prediction)")
    print('='*60)

    all_results = []

    glasgow_results = analyze_glasgow_norms(embeddings, word_to_idx)
    all_results.extend(glasgow_results)

    print(f"\n{'='*60}")
    print("Word Pair Similarity (Cosine similarity)")
    print('='*60)

    simlex_results, simlex_data = analyze_simlex(embeddings, word_to_idx)
    all_results.extend(simlex_results)

    wordsim_results, wordsim_data = analyze_wordsim(embeddings, word_to_idx)
    all_results.extend(wordsim_results)

    results_df = pd.DataFrame(all_results)
    results_df["significant"] = results_df["pvalue"] < 0.05
    results_df.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"\nResults saved to {OUTPUT_DIR / 'prediction_results.csv'}")

    return results_df, simlex_data, wordsim_data


if __name__ == "__main__":
    main()
