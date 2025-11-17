import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler


def load_glove_vectors(
    glove_path: Path, vocabulary: set[str], dim: int = 300
) -> dict[str, np.ndarray]:
    vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocabulary:
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == dim:
                        vectors[word] = vector
                except (ValueError, IndexError):
                    continue
    return vectors


def load_srf_embeddings(embedding_dir: Path) -> tuple[np.ndarray, list[str]]:
    w = np.load(embedding_dir / "srf_factors.npy")
    vocabulary = Path(embedding_dir / "vocabulary.txt").read_text().strip().split("\n")
    return w, vocabulary


def align_embeddings(
    srf_embeddings: np.ndarray,
    srf_vocab: list[str],
    glove_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    aligned = [
        (srf_embeddings[i], glove_vectors[word], word)
        for i, word in enumerate(srf_vocab)
        if word in glove_vectors
    ]
    srf, glove, vocab = zip(*aligned)
    return np.array(srf), np.array(glove), list(vocab)


from sklearn.preprocessing import StandardScaler


def preprocess_data(
    glove_train: np.ndarray,
    srf_train: np.ndarray,
    glove_test: np.ndarray,
    srf_test: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler
]:
    glove_scaler = StandardScaler()
    srf_scaler = StandardScaler(with_std=False)

    return (
        glove_scaler.fit_transform(glove_train),
        srf_scaler.fit_transform(srf_train),
        glove_scaler.transform(glove_test),
        srf_scaler.transform(srf_test),
        glove_scaler,
        srf_scaler,
    )


def uncenter_and_clip(
    srf_centered: np.ndarray, srf_scaler: StandardScaler
) -> np.ndarray:
    srf = srf_centered + srf_scaler.mean_
    srf[srf < 0] = 0
    return srf


def fit_single_dimension(
    dim: int,
    glove_train: np.ndarray,
    srf_train: np.ndarray,
    glove_test: np.ndarray,
    srf_test: np.ndarray,
    alphas: np.ndarray,
) -> tuple[int, np.ndarray, float, float, float]:
    ridge = RidgeCV(alphas=alphas, cv=10, scoring="r2")
    ridge.fit(glove_train, srf_train[:, dim])

    train_r2 = r2_score(srf_train[:, dim], glove_train @ ridge.coef_)
    test_r2 = r2_score(srf_test[:, dim], glove_test @ ridge.coef_)

    return dim, ridge.coef_, float(train_r2), float(test_r2), float(ridge.alpha_)


def learn_mapping(
    glove_train: np.ndarray,
    srf_train: np.ndarray,
    glove_test: np.ndarray,
    srf_test: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, dict]:
    results = Parallel(n_jobs=-1)(
        delayed(fit_single_dimension)(
            dim, glove_train, srf_train, glove_test, srf_test, alphas
        )
        for dim in range(srf_train.shape[1])
    )

    results.sort(key=lambda x: x[0])

    return (
        np.column_stack([r[1] for r in results]),
        {
            "dimension": [r[0] for r in results],
            "train_r2": [r[2] for r in results],
            "test_r2": [r[3] for r in results],
            "best_alpha": [r[4] for r in results],
        },
    )


def validate_semantic_axes(
    glove_test: np.ndarray,
    srf_test: np.ndarray,
    transform_matrix: np.ndarray,
    embedding_dir: Path,
    srf_scaler: StandardScaler,
) -> dict:
    semantic_axes = np.load(embedding_dir / "semantic_axes.npy")
    axis_names = (
        Path(embedding_dir / "semantic_axis_names.txt").read_text().strip().split("\n")
    )

    srf_pred = uncenter_and_clip(glove_test @ transform_matrix, srf_scaler)
    srf_actual = uncenter_and_clip(srf_test, srf_scaler)

    srf_pred_norm = srf_pred / np.linalg.norm(srf_pred, axis=1, keepdims=True)
    srf_actual_norm = srf_actual / np.linalg.norm(srf_actual, axis=1, keepdims=True)

    actual_scores = srf_actual_norm @ semantic_axes.T
    predicted_scores = srf_pred_norm @ semantic_axes.T

    correlations = [
        {
            "axis": name,
            "correlation": float(
                np.corrcoef(actual_scores[:, i], predicted_scores[:, i])[0, 1]
            ),
        }
        for i, name in enumerate(axis_names)
    ]

    return {
        "axis_correlations": correlations,
        "average_correlation": float(np.mean([c["correlation"] for c in correlations])),
    }


def compute_word_validation(
    glove_test: np.ndarray,
    srf_test: np.ndarray,
    transform_matrix: np.ndarray,
    test_vocab: list[str],
    srf_scaler: StandardScaler,
) -> pd.DataFrame:
    srf_pred = uncenter_and_clip(glove_test @ transform_matrix, srf_scaler)
    srf_actual = uncenter_and_clip(srf_test, srf_scaler)

    mse = np.mean((srf_actual - srf_pred) ** 2, axis=1)
    cosine_sim = np.sum(srf_actual * srf_pred, axis=1) / (
        np.linalg.norm(srf_actual, axis=1) * np.linalg.norm(srf_pred, axis=1) + 1e-10
    )

    return pd.DataFrame(
        {"word": test_vocab, "mse": mse, "cosine_similarity": cosine_sim}
    )


def save_results(
    transform_matrix: np.ndarray,
    dimension_metadata: dict,
    axis_validation: dict,
    word_validation: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "glove_to_srf_transform.npy", transform_matrix)

    with open(output_dir / "transform_metadata.json", "w") as f:
        json.dump(
            {
                "config": config,
                "dimension_metrics": dimension_metadata,
                "axis_validation": axis_validation,
                "transform_shape": list(transform_matrix.shape),
            },
            f,
            indent=2,
        )

    word_validation.to_csv(output_dir / "validation_results.csv", index=False)
    pd.DataFrame(dimension_metadata).to_csv(
        output_dir / "dimension_metrics.csv", index=False
    )


def main(
    embedding_dir: Path | str,
    glove_path: Path | str,
    output_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    embedding_dir = Path(embedding_dir)
    glove_path = Path(glove_path)
    output_dir = Path(output_dir) if output_dir else embedding_dir

    srf_embeddings, srf_vocab = load_srf_embeddings(embedding_dir)
    glove_vectors = load_glove_vectors(glove_path, set(srf_vocab))
    srf_aligned, glove_aligned, aligned_vocab = align_embeddings(
        srf_embeddings, srf_vocab, glove_vectors
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(aligned_vocab)), test_size=test_size, random_state=random_state
    )

    glove_train_z, srf_train_c, glove_test_z, srf_test_c, glove_scaler, srf_scaler = (
        preprocess_data(
            glove_aligned[train_idx],
            srf_aligned[train_idx],
            glove_aligned[test_idx],
            srf_aligned[test_idx],
        )
    )

    alphas = np.logspace(np.log10(0.01), np.log10(10000.0), 50)

    transform_matrix, dimension_metadata = learn_mapping(
        glove_train_z, srf_train_c, glove_test_z, srf_test_c, alphas
    )
    axis_validation = validate_semantic_axes(
        glove_test_z, srf_test_c, transform_matrix, embedding_dir, srf_scaler
    )
    word_validation = compute_word_validation(
        glove_test_z,
        srf_test_c,
        transform_matrix,
        [aligned_vocab[i] for i in test_idx],
        srf_scaler,
    )

    save_results(
        transform_matrix,
        dimension_metadata,
        axis_validation,
        word_validation,
        output_dir,
        {
            "embedding_dir": str(embedding_dir),
            "glove_path": str(glove_path),
            "test_size": test_size,
            "random_state": random_state,
            "glove_mean": glove_scaler.mean_.tolist(),
            "glove_std": glove_scaler.scale_.tolist(),
            "srf_mean": srf_scaler.mean_.tolist(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn mapping from GloVe to SRF embeddings"
    )
    parser.add_argument("--embedding-dir", type=str, required=True)
    parser.add_argument("--glove-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    main(
        args.embedding_dir,
        args.glove_path,
        args.output_dir,
        args.test_size,
        args.random_state,
    )
