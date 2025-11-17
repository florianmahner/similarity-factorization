import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


def load_behavioral_ratings(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)

    size_df = pd.read_csv(data_dir / "things/size_ratings.csv")
    size_df = size_df.rename(columns={"Word": "word", "Size_mean": "size_rating"})
    size_df["word"] = size_df["word"].str.lower()
    size_df = size_df[["word", "size_rating"]]

    concrete_df = pd.read_excel(data_dir / "concreteness_ratings_brysbaert.xlsx")
    concrete_df = concrete_df.rename(
        columns={"Word": "word", "Conc.M": "concreteness_rating"}
    )
    concrete_df["word"] = concrete_df["word"].str.lower()
    concrete_df = concrete_df[["word", "concreteness_rating"]]

    things_props = pd.read_csv(data_dir / "things/things_property_ratings.csv")
    things_props = things_props.rename(columns={"Word": "word"})
    things_props["word"] = things_props["word"].str.lower()

    property_cols = {
        "pleasant_mean": "valence_rating",
        "heavy_mean": "heaviness_rating",
        "lives_mean": "animacy_rating",
    }

    things_props = things_props.rename(columns=property_cols)
    things_props = things_props[["word"] + list(property_cols.values())]

    ratings = size_df.merge(concrete_df, on="word", how="outer")
    ratings = ratings.merge(things_props, on="word", how="outer")
    return ratings


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
