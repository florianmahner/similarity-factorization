from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cli import ExperimentContext
from ..lib.plotting import plot_grouped_bars
from ..lib.semantic_axes import compute_axis
from ..lib.validation import load_behavioral_ratings

if TYPE_CHECKING:
    from ..exp import Params

RATINGS = {
    "concreteness": "concreteness_rating",
    "size": "size_rating",
    "valence": "valence_rating",
    "heaviness": "heaviness_rating",
    "animacy": "animacy_rating",
}


def select_pole_words(
    ratings: pd.DataFrame,
    rating_col: str,
    word_to_idx: dict[str, int],
    top_n: int = 50,
) -> tuple[list[str], list[str]]:
    """Select top N highest and lowest rated words for pole definition.

    Returns shuffled lists to avoid alphabetical bias.
    """
    valid_ratings = ratings[["word", rating_col]].dropna()
    valid_ratings = valid_ratings[valid_ratings["word"].isin(word_to_idx)]

    if len(valid_ratings) < 2 * top_n:
        raise ValueError(f"Not enough rated words for {rating_col}")

    sorted_ratings = valid_ratings.sort_values(rating_col)

    low_words = sorted_ratings.head(top_n)["word"].tolist()
    high_words = sorted_ratings.tail(top_n)["word"].tolist()

    rng = np.random.RandomState(42)
    rng.shuffle(low_words)
    rng.shuffle(high_words)

    return high_words, low_words


def select_stratified_words(
    ratings: pd.DataFrame,
    rating_col: str,
    word_to_idx: dict[str, int],
    n_bins: int = 5,
    n_per_bin: int = 20,
) -> tuple[list[str], dict[str, float]]:
    """Select words stratified across rating range.

    Returns:
        train_words: List of selected words
        diagnostics: Dict with distribution statistics
    """
    valid_ratings = ratings[["word", rating_col]].dropna()
    valid_ratings = valid_ratings[valid_ratings["word"].isin(word_to_idx)]

    if len(valid_ratings) < n_bins * n_per_bin:
        raise ValueError(f"Not enough rated words for {rating_col}")

    valid_ratings["bin"] = pd.qcut(
        valid_ratings[rating_col], q=n_bins, labels=False, duplicates="drop"
    )

    train_words = []
    rng = np.random.RandomState(42)

    for bin_idx in range(n_bins):
        bin_words = valid_ratings[valid_ratings["bin"] == bin_idx]["word"].tolist()
        if len(bin_words) >= n_per_bin:
            sampled = rng.choice(bin_words, size=n_per_bin, replace=False).tolist()
        else:
            sampled = bin_words
        train_words.extend(sampled)

    rng.shuffle(train_words)

    train_set = set(train_words)
    test_ratings = valid_ratings[~valid_ratings["word"].isin(train_set)]

    diagnostics = {
        "n_bins": n_bins,
        "train_rating_mean": float(
            valid_ratings[valid_ratings["word"].isin(train_set)][rating_col].mean()
        ),
        "train_rating_std": float(
            valid_ratings[valid_ratings["word"].isin(train_set)][rating_col].std()
        ),
        "test_rating_mean": float(test_ratings[rating_col].mean())
        if len(test_ratings) > 0
        else 0.0,
        "test_rating_std": float(test_ratings[rating_col].std())
        if len(test_ratings) > 0
        else 0.0,
    }

    return train_words, diagnostics


def validate_unsupervised(
    word_embedding: np.ndarray,
    vocabulary: list[str],
    word_to_idx: dict[str, int],
    ratings: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Validate using axes derived from extreme rated words.

    For each rating, selects top/bottom N words, computes axis, predicts
    rating for other words.
    """
    results = []

    for axis_name, rating_col in RATINGS.items():
        if rating_col not in ratings.columns:
            continue

        try:
            high_words, low_words = select_pole_words(
                ratings, rating_col, word_to_idx, top_n
            )
        except ValueError:
            continue

        pole_word_set = set(high_words + low_words)

        high_idx = [word_to_idx[w] for w in high_words]
        low_idx = [word_to_idx[w] for w in low_words]

        axis = compute_axis(word_embedding[high_idx], word_embedding[low_idx])

        projections = word_embedding @ axis
        projection_df = pd.DataFrame({
            "word": [w.lower() for w in vocabulary],
            "projection": projections,
        })

        test_ratings = ratings[~ratings["word"].isin(pole_word_set)].copy()
        merged = test_ratings.merge(projection_df, on="word", how="inner")
        valid = merged[["projection", rating_col]].dropna()

        if len(valid) < 10:
            continue

        r, p = spearmanr(valid["projection"], valid[rating_col])

        results.append({
            "axis": axis_name,
            "method": "unsupervised",
            "r": r,
            "p": p,
            "n_train": len(pole_word_set),
            "n_test": len(valid),
        })

    return pd.DataFrame(results)


def validate_supervised(
    word_embedding: np.ndarray,
    vocabulary: list[str],
    word_to_idx: dict[str, int],
    ratings: pd.DataFrame,
    use_stratified: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train Ridge on sampled words, predict on other words.

    Args:
        use_stratified: If True, use stratified sampling; else use extreme poles

    Returns:
        Validation results, predictions, and diagnostics
    """
    alphas = np.logspace(-2, 3, 20)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    all_predictions = {"word": [word.lower() for word in vocabulary]}
    all_diagnostics = {}

    for axis_name, rating_col in RATINGS.items():
        if rating_col not in ratings.columns:
            all_predictions[f"{axis_name}_predicted"] = [np.nan] * len(vocabulary)
            continue

        try:
            if use_stratified:
                train_words, diagnostics = select_stratified_words(
                    ratings, rating_col, word_to_idx, n_bins=5, n_per_bin=20
                )
                all_diagnostics[axis_name] = diagnostics
            else:
                high_words, low_words = select_pole_words(
                    ratings, rating_col, word_to_idx, top_n=50
                )
                train_words = high_words + low_words
        except ValueError:
            all_predictions[f"{axis_name}_predicted"] = [np.nan] * len(vocabulary)
            continue

        train_word_set = set(train_words)

        train_data = ratings[ratings["word"].isin(train_words)].copy()
        X_train = word_embedding[[word_to_idx[w] for w in train_data["word"]]]
        y_train = train_data[rating_col].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RidgeCV(alphas=alphas, cv=cv, scoring="r2")
        model.fit(X_train_scaled, y_train)

        X_all_scaled = scaler.transform(word_embedding)
        predictions = model.predict(X_all_scaled)
        all_predictions[f"{axis_name}_predicted"] = predictions

        prediction_df = pd.DataFrame({
            "word": [w.lower() for w in vocabulary],
            "prediction": predictions,
        })

        test_ratings = ratings[~ratings["word"].isin(train_word_set)].copy()
        merged = test_ratings.merge(prediction_df, on="word", how="inner")
        valid = merged[["prediction", rating_col]].dropna()

        if len(valid) < 10:
            continue

        r, p = spearmanr(valid["prediction"], valid[rating_col])

        results.append({
            "axis": axis_name,
            "method": "supervised_stratified" if use_stratified else "supervised_extreme",
            "r": r,
            "p": p,
            "alpha": model.alpha_,
            "n_train": len(train_words),
            "n_test": len(valid),
        })

    return pd.DataFrame(results), pd.DataFrame(all_predictions), all_diagnostics


def run(context: ExperimentContext, params: Params) -> Path:
    word_embedding = np.load(params.embedding_dir / "word_embedding.npy")
    vocabulary = (
        (params.embedding_dir / "vocabulary.txt").read_text().strip().split("\n")
    )
    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}

    ratings = load_behavioral_ratings(params.ratings_data_dir)

    unsup_results = validate_unsupervised(
        word_embedding, vocabulary, word_to_idx, ratings
    )

    sup_extreme_results, _, _ = validate_supervised(
        word_embedding, vocabulary, word_to_idx, ratings, use_stratified=False
    )

    sup_stratified_results, predictions, diagnostics = validate_supervised(
        word_embedding, vocabulary, word_to_idx, ratings, use_stratified=True
    )

    context.logger.info("\n=== Unsupervised (Pole-based Axes) ===")
    for _, row in unsup_results.iterrows():
        context.logger.info(
            f"{row['axis']}: r={row['r']:.3f}, p={row['p']:.3e}, "
            f"n_train={row['n_train']}, n_test={row['n_test']}"
        )

    context.logger.info("\n=== Supervised (Extreme Poles Only) ===")
    for _, row in sup_extreme_results.iterrows():
        context.logger.info(
            f"{row['axis']}: r={row['r']:.3f}, p={row['p']:.3e}, "
            f"alpha={row['alpha']:.2e}, n_train={row['n_train']}, n_test={row['n_test']}"
        )

    context.logger.info("\n=== Supervised (Stratified Sampling) ===")
    for _, row in sup_stratified_results.iterrows():
        axis = row['axis']
        context.logger.info(
            f"{axis}: r={row['r']:.3f}, p={row['p']:.3e}, "
            f"alpha={row['alpha']:.2e}, n_train={row['n_train']}, n_test={row['n_test']}"
        )
        if axis in diagnostics:
            diag = diagnostics[axis]
            context.logger.info(
                f"  Train: mean={diag['train_rating_mean']:.2f}, std={diag['train_rating_std']:.2f} | "
                f"Test: mean={diag['test_rating_mean']:.2f}, std={diag['test_rating_std']:.2f}"
            )

    results = pd.concat(
        [unsup_results, sup_extreme_results, sup_stratified_results], ignore_index=True
    )
    context.save_csv(results, "validation")
    context.save_csv(predictions, "predicted_ratings")

    plot_grouped_bars(
        data=results,
        category_col="axis",
        value_col="r",
        group_col="method",
        title="Semantic Axis Validation: Comparison of Methods",
        ylabel="Correlation (r)",
        output_path=context.run_dir / "plots/validation_comparison.png",
    )

    return context.run_dir
