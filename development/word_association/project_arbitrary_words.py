import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings("ignore")


def preprocess_glove(
    glove: np.ndarray,
    glove_mean: np.ndarray,
    glove_std: np.ndarray,
) -> np.ndarray:
    """Z-score normalize GloVe vectors using training statistics."""
    return (glove - glove_mean) / glove_std


def load_glove_vocabulary(glove_path: Path, max_words: int | None = None) -> list[str]:
    print(f"Loading GloVe vocabulary from {glove_path.name}...")
    vocabulary = []

    with open(glove_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            word = line.split(" ", 1)[0]
            vocabulary.append(word)

            if max_words and len(vocabulary) >= max_words:
                break

            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i+1:,} words")

    print(f"  Total: {len(vocabulary):,} words")
    return vocabulary


def load_glove_vectors_for_words(
    glove_path: Path, words: list[str], dim: int = 300
) -> dict[str, np.ndarray]:
    print(f"\nLoading GloVe vectors for {len(words):,} words...")
    word_set = set(words)
    vectors = {}

    with open(glove_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            word = parts[0]

            if word in word_set:
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == dim:
                        vectors[word] = vector
                except (ValueError, IndexError):
                    continue

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1:,} lines, found {len(vectors):,} matches")

            if len(vectors) == len(word_set):
                break

    print(f"  Found vectors for {len(vectors):,} / {len(words):,} words")
    return vectors


def load_transform_and_axes(transform_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    transform_matrix = np.load(transform_dir / "glove_to_srf_transform.npy")

    semantic_axes = np.load(transform_dir / "semantic_axes.npy")

    with open(transform_dir / "semantic_axis_names.txt", "r") as f:
        axis_names = [line.strip() for line in f]

    with open(transform_dir / "transform_metadata.json", "r") as f:
        metadata = json.load(f)
        config = metadata.get("config", {})
        
    glove_mean = np.array(config["glove_mean"])
    glove_std = np.array(config["glove_std"])
    srf_mean = np.array(config["srf_mean"])

    print(f"\nLoaded transformation:")
    print(f"  Transform matrix: {transform_matrix.shape}")
    print(f"  Semantic axes: {semantic_axes.shape}")
    print(f"  Axis names: {', '.join(axis_names)}")
    print(f"\nPreprocessing: Z-score GloVe, center SRF, clip to non-negative")

    return transform_matrix, semantic_axes, axis_names, glove_mean, glove_std, srf_mean


def load_swow_vocabulary(transform_dir: Path) -> set[str]:
    with open(transform_dir / "vocabulary.txt", "r") as f:
        vocabulary = {line.strip() for line in f}
    return vocabulary


def project_words_onto_axes(
    glove_vectors: dict[str, np.ndarray],
    transform_matrix: np.ndarray,
    semantic_axes: np.ndarray,
    axis_names: list[str],
    glove_mean: np.ndarray,
    glove_std: np.ndarray,
    srf_mean: np.ndarray,
) -> pd.DataFrame:
    print(f"\nProjecting {len(glove_vectors):,} words onto semantic axes...")

    words = list(glove_vectors.keys())
    glove_matrix = np.array([glove_vectors[w] for w in words])

    glove_matrix_z = preprocess_glove(glove_matrix, glove_mean, glove_std)

    srf_predicted_c = glove_matrix_z @ transform_matrix
    
    srf_predicted = srf_predicted_c + srf_mean
    srf_predicted[srf_predicted < 0] = 0

    axis_scores = srf_predicted @ semantic_axes.T

    score_max = np.abs(axis_scores).max(axis=0, keepdims=True)
    score_max[score_max == 0] = 1
    axis_scores_normalized = axis_scores / score_max

    projections = {"word": words}
    for i, axis_name in enumerate(axis_names):
        projections[axis_name] = axis_scores_normalized[:, i]

    return pd.DataFrame(projections)


def compare_with_swow_words(
    projections: pd.DataFrame,
    swow_vocab: set[str],
    transform_dir: Path,
) -> pd.DataFrame:
    swow_projections = pd.read_csv(transform_dir / "word_projections.csv")

    swow_words_in_proj = projections[projections["word"].isin(swow_vocab)]

    if len(swow_words_in_proj) == 0:
        return pd.DataFrame()

    comparison_data = []

    axis_names = [col for col in projections.columns if col != "word"]

    for _, row in swow_words_in_proj.iterrows():
        word = row["word"]

        swow_row = swow_projections[swow_projections["word"] == word]
        if len(swow_row) == 0:
            continue

        swow_row = swow_row.iloc[0]

        for axis_name in axis_names:
            predicted_score = row[axis_name]
            actual_score = swow_row[axis_name]
            error = predicted_score - actual_score

            comparison_data.append(
                {
                    "word": word,
                    "axis": axis_name,
                    "predicted": float(predicted_score),
                    "actual": float(actual_score),
                    "error": float(error),
                    "abs_error": float(abs(error)),
                }
            )

    return pd.DataFrame(comparison_data)


def find_nearest_swow_neighbors(
    word: str,
    glove_vector: np.ndarray,
    swow_vocab: set[str],
    glove_vectors: dict[str, np.ndarray],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    if word in swow_vocab:
        return []

    neighbors = []
    word_norm = np.linalg.norm(glove_vector)

    for swow_word in swow_vocab:
        if swow_word in glove_vectors:
            swow_vec = glove_vectors[swow_word]
            swow_norm = np.linalg.norm(swow_vec)

            if word_norm > 0 and swow_norm > 0:
                cosine_sim = np.dot(glove_vector, swow_vec) / (word_norm * swow_norm)
                neighbors.append((swow_word, float(cosine_sim)))

    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:top_k]


def analyze_novel_words(
    projections: pd.DataFrame,
    swow_vocab: set[str],
    glove_vectors: dict[str, np.ndarray],
    n_examples: int = 20,
) -> pd.DataFrame:
    novel_words = projections[~projections["word"].isin(swow_vocab)].copy()

    if len(novel_words) == 0:
        return pd.DataFrame()

    print(f"\nAnalyzing {len(novel_words):,} novel words (not in SWOW)...")

    examples = novel_words.sample(min(n_examples, len(novel_words)), random_state=42)

    novel_analysis = []

    for _, row in examples.iterrows():
        word = row["word"]
        glove_vec = glove_vectors[word]

        neighbors = find_nearest_swow_neighbors(
            word, glove_vec, swow_vocab, glove_vectors, top_k=5
        )

        neighbor_words = [n[0] for n in neighbors]
        neighbor_sims = [n[1] for n in neighbors]

        axis_scores = {
            col: float(row[col])
            for col in projections.columns
            if col != "word"
        }

        novel_analysis.append(
            {
                "word": word,
                "nearest_swow_neighbors": ", ".join(neighbor_words),
                "neighbor_similarities": ", ".join([f"{s:.3f}" for s in neighbor_sims]),
                **axis_scores,
            }
        )

    return pd.DataFrame(novel_analysis)


def save_results(
    projections: pd.DataFrame,
    comparison: pd.DataFrame,
    novel_analysis: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    projections.to_csv(output_dir / "extended_word_projections.csv", index=False)

    if len(comparison) > 0:
        comparison.to_csv(output_dir / "prediction_quality.csv", index=False)

    if len(novel_analysis) > 0:
        novel_analysis.to_csv(output_dir / "novel_word_analysis.csv", index=False)

    print("\nSaved outputs:")
    print(f"  - extended_word_projections.csv: {len(projections):,} words")
    if len(comparison) > 0:
        print(f"  - prediction_quality.csv: {len(comparison):,} comparisons")
    if len(novel_analysis) > 0:
        print(f"  - novel_word_analysis.csv: {len(novel_analysis):,} novel words")


def print_word_results(
    projections: pd.DataFrame,
    axis_names: list[str],
    swow_vocab: set[str],
) -> None:
    print("\n" + "=" * 70)
    print("Semantic Axis Projections")
    print("=" * 70)

    for _, row in projections.iterrows():
        word = row["word"]
        is_swow = word in swow_vocab

        print(f"\n{word.upper()} {'(in SWOW)' if is_swow else '(novel)'}")
        print("-" * 70)

        for axis_name in axis_names:
            score = row[axis_name]
            print(f"  {axis_name:15s}: {score:6.3f}")


def main(
    transform_dir: Path | str,
    glove_path: Path | str,
    words: list[str] | None = None,
    all_vocab: bool = False,
    output_dir: Path | str | None = None,
    max_words: int | None = None,
) -> None:
    transform_dir = Path(transform_dir)
    glove_path = Path(glove_path)

    if output_dir is None:
        output_dir = transform_dir
    else:
        output_dir = Path(output_dir)

    print("=" * 70)
    print("Project Arbitrary Words onto Semantic Axes")
    print("=" * 70)
    print(f"Transform directory: {transform_dir}")
    print(f"GloVe file: {glove_path}")
    print(f"Output directory: {output_dir}")

    transform_matrix, semantic_axes, axis_names, glove_mean, glove_std, srf_mean = load_transform_and_axes(transform_dir)

    swow_vocab = load_swow_vocabulary(transform_dir)
    print(f"\nSWOW vocabulary: {len(swow_vocab):,} words")

    if all_vocab:
        target_words = load_glove_vocabulary(glove_path, max_words=max_words)
    elif words:
        target_words = words
    else:
        raise ValueError("Must specify either --words or --all-vocab")

    print(f"\nTarget words: {len(target_words):,}")

    glove_vectors = load_glove_vectors_for_words(glove_path, target_words)

    projections = project_words_onto_axes(
        glove_vectors, transform_matrix, semantic_axes, axis_names, 
        glove_mean, glove_std, srf_mean
    )

    comparison = compare_with_swow_words(projections, swow_vocab, transform_dir)

    if len(comparison) > 0:
        print(f"\n{len(comparison):,} axis comparisons for SWOW words")
        print(f"  Mean absolute error: {comparison['abs_error'].mean():.3f}")
        print(f"  Median absolute error: {comparison['abs_error'].median():.3f}")

        axis_performance = comparison.groupby("axis")["abs_error"].mean().sort_values()
        print("\n  Best performing axes (lowest error):")
        for axis, error in axis_performance.head(5).items():
            print(f"    {axis}: {error:.3f}")

    novel_analysis = analyze_novel_words(
        projections, swow_vocab, glove_vectors, n_examples=20
    )

    save_results(projections, comparison, novel_analysis, output_dir)

    if not all_vocab and len(projections) <= 20:
        print_word_results(projections, axis_names, swow_vocab)

    print("\n" + "=" * 70)
    print("Projection complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project arbitrary words onto semantic axes using GloVe→SRF mapping"
    )
    parser.add_argument(
        "--transform-dir",
        type=str,
        required=True,
        help="Directory containing transformation matrix and semantic axes",
    )
    parser.add_argument(
        "--glove-path",
        type=str,
        required=True,
        help="Path to GloVe embeddings file",
    )
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Comma-separated list of words to project",
    )
    parser.add_argument(
        "--all-vocab",
        action="store_true",
        help="Project all words in GloVe vocabulary",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to transform-dir)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Maximum number of words to process (for --all-vocab)",
    )

    args = parser.parse_args()

    words_list = None
    if args.words:
        words_list = [w.strip() for w in args.words.split(",")]

    main(
        transform_dir=args.transform_dir,
        glove_path=args.glove_path,
        words=words_list,
        all_vocab=args.all_vocab,
        output_dir=args.output_dir,
        max_words=args.max_words,
    )

