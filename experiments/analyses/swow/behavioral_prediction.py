from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_word_to_idx(vocabulary: list[str]) -> dict[str, int]:
    """Create word to index mapping.

    Args:
        vocabulary: List of words

    Returns:
        Mapping from word to embedding index
    """
    return {word.lower(): i for i, word in enumerate(vocabulary)}


def extract_rating_arrays(
    ratings: pd.DataFrame,
    dimension: str,
    word_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned arrays from ratings DataFrame.

    Args:
        ratings: DataFrame with 'word' and dimension columns
        dimension: Rating column name
        word_to_idx: Word to index mapping

    Returns:
        word_indices: Embedding indices for rated words
        rating_values: Corresponding rating values
    """
    valid = ratings[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]

    words = valid["word"].values
    word_indices = np.array([word_to_idx[w] for w in words])
    rating_values = valid[dimension].to_numpy()

    return word_indices, rating_values


def select_pole_indices(
    rating_values: np.ndarray,
    word_indices: np.ndarray,
    n: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Select indices for high/low pole words.

    Args:
        rating_values: Rating values
        word_indices: Corresponding word indices
        n: Number of words per pole

    Returns:
        high_indices: Indices of high pole words
        low_indices: Indices of low pole words
    """
    if len(rating_values) < 2 * n:
        raise ValueError(f"Need {2 * n} words, found {len(rating_values)}")

    sorted_idx = np.argsort(rating_values)
    low_indices = word_indices[sorted_idx[:n]]
    high_indices = word_indices[sorted_idx[-n:]]

    rng = np.random.RandomState(42)
    rng.shuffle(low_indices)
    rng.shuffle(high_indices)

    return high_indices, low_indices


def compute_semantic_axis(
    high_embeddings: np.ndarray,
    low_embeddings: np.ndarray,
    normalize_poles: bool = True,  # Toggle this based on model type
) -> np.ndarray:

    high_proto = np.mean(high_embeddings, axis=0)
    low_proto = np.mean(low_embeddings, axis=0)

    if normalize_poles:
        # Crucial for NMF: Make sure "Big" and "Small" have equal weight
        high_proto = high_proto / (np.linalg.norm(high_proto) + 1e-10)
        low_proto = low_proto / (np.linalg.norm(low_proto) + 1e-10)

    axis = high_proto - low_proto
    return axis / (np.linalg.norm(axis) + 1e-10)


def project_onto_axis(
    embeddings: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    embeddings_norm = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    )
    return embeddings_norm @ axis


def compute_correlation(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float]:
    valid_mask = ~(np.isnan(predictions) | np.isnan(ground_truth))
    if np.sum(valid_mask) < 10:
        return np.nan, np.nan

    r, p = spearmanr(predictions[valid_mask], ground_truth[valid_mask])
    return r, p


def evaluate_gland_projection_method(
    embedding: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    pole_size: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate semantic projection method on all dimensions.

    Args:
        embeddings: Word embeddings (vocab_size x dim)
        vocabulary: Word list
        ratings: Ground truth ratings
        pole_size: Number of words per pole

    Returns:
        results: Evaluation metrics per dimension
        projections: Projection scores for all words
    """
    word_to_idx = prepare_word_to_idx(vocabulary)
    dimensions = [col for col in ratings.columns if col != "word"]

    results = []
    all_projections = {"word": [w.lower() for w in vocabulary]}

    for dimension in dimensions:
        word_indices, rating_values = extract_rating_arrays(
            ratings, dimension, word_to_idx
        )

        high_idx, low_idx = select_pole_indices(rating_values, word_indices, pole_size)

        axis = compute_semantic_axis(embedding[high_idx], embedding[low_idx])
        scores = project_onto_axis(embedding, axis)

        all_projections[dimension] = scores

        test_mask = ~np.isin(word_indices, np.concatenate([high_idx, low_idx]))
        test_indices = word_indices[test_mask]
        test_ratings = rating_values[test_mask]

        test_predictions = scores[test_indices]
        r, p = compute_correlation(test_predictions, test_ratings)

        results.append(
            {
                "dimension": dimension,
                "method": "Gland_Projection",
                "correlation": r,
                "pvalue": p,
                "n_train": len(high_idx) + len(low_idx),
                "n_test": len(test_indices),
            }
        )

    return pd.DataFrame(results), pd.DataFrame(all_projections)


# =============================================================================
# SUPERVISED BASELINE
# =============================================================================


def train_ridge_with_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> np.ndarray:
    """Train ridge regression with nested cross-validation.

    Args:
        X: Features (n_samples x n_features)
        y: Targets (n_samples,)
        n_outer_folds: Outer CV folds
        n_inner_folds: Inner CV folds

    Returns:
        Out-of-fold predictions (n_samples,)
    """
    predictions = np.full(len(X), np.nan)
    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RidgeCV(
            alphas=np.logspace(-4, 2, 13),
            cv=KFold(n_splits=n_inner_folds, shuffle=True, random_state=42),
        )
        model.fit(X_train_scaled, y_train)

        predictions[test_idx] = model.predict(X_test_scaled)

    return predictions


def evaluate_ridge_encoding(
    embeddings: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate supervised ridge regression with nested CV.

    Args:
        embeddings: Word embeddings (vocab_size x dim)
        vocabulary: Word list
        ratings: Ground truth ratings
        n_outer_folds: Outer CV folds
        n_inner_folds: Inner CV folds

    Returns:
        results: Evaluation metrics per dimension
        predictions: Out-of-fold predictions for all words
    """
    word_to_idx = prepare_word_to_idx(vocabulary)
    dimensions = [col for col in ratings.columns if col != "word"]

    results = []
    all_predictions = {"word": [w.lower() for w in vocabulary]}

    for dimension in dimensions:
        try:
            word_indices, rating_values = extract_rating_arrays(
                ratings, dimension, word_to_idx
            )

            if len(word_indices) < 50:
                raise ValueError(f"Insufficient data: {len(word_indices)} words")

            X = embeddings[word_indices]
            y = rating_values

            oof_predictions = train_ridge_with_nested_cv(
                X, y, n_outer_folds, n_inner_folds
            )

            full_predictions = np.full(len(embeddings), np.nan)
            full_predictions[word_indices] = oof_predictions

            all_predictions[dimension] = full_predictions

            r, p = compute_correlation(oof_predictions, y)

            results.append(
                {
                    "dimension": dimension,
                    "method": "Ridge_CV",
                    "correlation": r,
                    "pvalue": p,
                    "n_train": int(
                        len(word_indices) * (n_outer_folds - 1) / n_outer_folds
                    ),
                    "n_test": int(len(word_indices) / n_outer_folds),
                }
            )

        except (ValueError, KeyError):
            continue

    return pd.DataFrame(results), pd.DataFrame(all_predictions)


def evaluate_lasso_encoding(
    embeddings: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    n_outer_folds: int = 5,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates embeddings using Lasso (L1) regression to respect sparsity.

    This method assumes that behavioral variables (like Concreteness)
    are composed of a sparse sum of the NMF topics.
    """
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}
    dimensions = [col for col in ratings.columns if col != "word"]

    results = []
    # Store predictions for error analysis
    all_predictions = {"word": [w.lower() for w in vocabulary]}

    # We want to track how "sparse" the solution is (interpretability check)
    sparsity_stats = {}

    for dimension in dimensions:
        # 1. Data Extraction
        valid = ratings[["word", dimension]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]

        if len(valid) < 50:
            print(f"Skipping {dimension}: insufficient data ({len(valid)})")
            continue

        word_indices = np.array([word_to_idx[w] for w in valid["word"]])
        X = embeddings[word_indices]
        y = valid[dimension].to_numpy()

        # 2. Nested CV with Lasso
        # LassoCV handles the inner loop for hyperparameter tuning automatically
        kf = KFold(n_splits=n_outer_folds, shuffle=True, random_state=random_seed)

        oof_preds = np.full_like(y, np.nan)
        coef_sparsity = []  # Track how many dims are used

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standard scaling is usually required for convergence,
            # even with NMF, though it temporarily breaks non-negativity
            # *during* the fitting process.
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # LassoCV: L1 penalty. favors sparse coefficients.
            model = LassoCV(cv=5, random_state=random_seed, max_iter=10_000)
            model.fit(X_train_scaled, y_train)

            oof_preds[test_idx] = model.predict(X_test_scaled)

            # Count non-zero coefficients (how many NMF dims explain this trait?)
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
            coef_sparsity.append(n_nonzero)

        # 3. Metrics
        r, p = spearmanr(oof_preds, y)
        avg_active_dims = np.mean(coef_sparsity)

        # Store predictions
        full_preds = np.full(len(embeddings), np.nan)
        full_preds[word_indices] = oof_preds
        all_predictions[dimension] = full_preds

        results.append(
            {
                "dimension": dimension,
                "method": "Lasso_CV",
                "correlation": r,
                "pvalue": p,
                "avg_active_dims": avg_active_dims,  # Key metric for your specific embedding
                "total_dims": X.shape[1],
                "sparsity_ratio": 1.0 - (avg_active_dims / X.shape[1]),
            }
        )

    return pd.DataFrame(results), pd.DataFrame(all_predictions)
