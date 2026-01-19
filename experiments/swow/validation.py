import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


def load_behavioral_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Load Glasgow Norms for behavioral prediction.

    The Glasgow Norms (Scott et al., 2019) provide ratings for:
    - Arousal, Valence, Dominance (affective dimensions)
    - Concreteness, Imageability (semantic dimensions)
    - Age of Acquisition (AoA)
    - Size, Familiarity, Gender
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "semantic_norms/glasgow_norms.csv")
    df["word"] = df["word"].str.lower()

    column_mapping = {
        "AROU": "arousal",
        "VAL": "valence",
        "DOM": "dominance",
        "CNC": "concreteness",
        "IMAG": "imageability",
        "AOA": "aoa",
        "SIZE": "size",
        "FAM": "familiarity",
    }

    df = df.rename(columns=column_mapping)
    return df[["word"] + list(column_mapping.values())]


def match_words_to_ratings(
    projections_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    projections_df = projections_df.copy()
    projections_df["word"] = projections_df["word"].str.lower()

    return projections_df.merge(ratings_df, on="word", how="inner")


def convert_to_unipolar(
    merged_df: pd.DataFrame,
    axis_columns: list[str],
) -> pd.DataFrame:
    merged_df = merged_df.copy()

    for axis in axis_columns:
        if axis in merged_df.columns:
            merged_df[axis] = merged_df[axis] - merged_df[axis].min()

    return merged_df


def compute_axis_correlations(
    merged_df: pd.DataFrame,
    axis_rating_pairs: dict[str, str],
) -> pd.DataFrame:
    results = []

    for axis_col, rating_col in axis_rating_pairs.items():
        valid_data = merged_df[[axis_col, rating_col]].dropna()

        if len(valid_data) < 10:
            results.append(
                {
                    "axis": axis_col,
                    "rating": rating_col,
                    "n_words": len(valid_data),
                    "correlation": np.nan,
                    "pvalue": np.nan,
                }
            )
            continue

        corr, pval = spearmanr(valid_data[axis_col], valid_data[rating_col])

        results.append(
            {
                "axis": axis_col,
                "rating": rating_col,
                "n_words": len(valid_data),
                "correlation": corr,
                "pvalue": pval,
            }
        )

    return pd.DataFrame(results)


def get_extreme_words(
    df: pd.DataFrame,
    axis_column: str,
    n_top: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_data = df[["word", axis_column]].dropna()

    top_words = valid_data.nlargest(n_top, axis_column)
    bottom_words = valid_data.nsmallest(n_top, axis_column)

    return top_words, bottom_words


def load_semantic_poles(pole_file):
    with open(pole_file, "r") as f:
        lines = f.read().strip().split("\n")

    poles = {}
    current_axis = None

    for line in lines:
        if ":" in line and not line.startswith(" "):
            current_axis = line.split(":")[0].strip()
            poles[current_axis] = {"negative": [], "positive": []}
        elif line.strip() and current_axis:
            words = [w.strip() for w in line.split(",") if w.strip()]
            if len(poles[current_axis]["negative"]) == 0:
                poles[current_axis]["negative"] = words
            else:
                poles[current_axis]["positive"] = words

    return poles
