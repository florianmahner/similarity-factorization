import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from typing import Iterable
import re
from collections import defaultdict


def rows_to_paths(
    rows: np.ndarray,
    csv_path: str | Path = "image_filenames.csv",
) -> np.ndarray:
    tbl = load_table(csv_path)
    return tbl["path"].to_numpy()[rows]


def filenames_to_paths(
    filenames: Iterable[str],
    csv_path: str | Path = "image_filenames.csv",
) -> np.ndarray:
    rows = filenames_to_rows(filenames, csv_path)
    return rows_to_paths(rows, csv_path)


def mask_filenames(
    filenames: np.ndarray,
    plus: bool = False,
    behavior: bool = False,
    regex: str | None = None,
) -> np.ndarray:
    # Start with all True
    # make filenames str
    filenames = np.array(filenames, dtype=str)
    mask = np.ones(filenames.shape[0], dtype=bool)
    if plus:
        mask &= np.char.find(filenames, "plus") != -1
    if behavior:
        mask &= np.char.endswith(filenames, "01b.jpg")
    if regex is not None:
        pat = re.compile(regex)
        mask &= np.vectorize(lambda s: bool(pat.search(s)))(filenames)
    return mask, filenames[mask]


def limit_per_category(
    filepaths: Iterable[str], num_per_category: int = 10
) -> np.ndarray:
    """
    Given a list of filepaths, limit the number of categories to the top num_per_category.
    """
    file_list = list(filepaths)
    index_map: dict[str, int] = {p: i for i, p in enumerate(file_list)}

    sorted_paths = sorted(file_list, key=lambda p: (Path(p).parent.name, Path(p).name))

    grouped: dict[str, list[str]] = defaultdict(list)
    for p in sorted_paths:
        cat = Path(p).parent.name
        grouped[cat].append(p)

    selected_paths: list[str] = []
    selected_indices: list[int] = []

    for paths in grouped.values():
        # always-include
        always = [p for p in paths if p.endswith("_01b.jpg") or p.endswith("_plus.jpg")]
        # the rest
        others = [p for p in paths if p not in always]

        # how many more we need
        slots = max(num_per_category - len(always), 0)
        chosen = others[:slots]

        # final picks, sorted by filename
        picked = sorted(always + chosen, key=lambda p: Path(p).name)

        for p in picked:
            selected_paths.append(p)
            selected_indices.append(index_map[p])

    return np.array(selected_indices, dtype=int), np.array(selected_paths, dtype=str)


def categories_to_rows(
    categories: Iterable[str],
    csv_path: str = "image_filenames.csv",
) -> np.ndarray:
    """
    Given a list of category names (no '.jpg'), pick the FIRST row
    in image_filenames.csv whose filename starts with that category + '_'.
    """
    df = pd.read_csv(csv_path)
    # extract category by chopping off the last '_...' suffix
    df["category"] = df["filename"].str.rsplit("_", 1).str[0]
    # for each category, take the first row index
    first_rows = df.groupby("category", sort=False)["row"].first()
    try:
        return first_rows.loc[list(categories)].to_numpy(dtype=np.int32)
    except KeyError as e:
        missing = set(categories) - set(first_rows.index)
        raise ValueError(f"No images found for categories: {missing}") from e


def filter_items_to_rows(
    items: Iterable[str],
    csv_path: str = "image_filenames.csv",
) -> np.ndarray:
    """
    If all items look like filenames (end with '.jpg'), map directly.
    Otherwise treat them as categories and append '_01b.jpg' to each.
    """
    items = list(items)
    # check that every item ends with '.jpg'
    if items and all(item.endswith(".jpg") for item in items):
        return filenames_to_rows(items, csv_path)
    else:
        return categories_to_rows(items, csv_path)


def filenames_to_rows(
    filenames: Iterable[str], csv_path: str | Path = "image_filenames.csv"
) -> np.ndarray:
    tbl = load_table(csv_path)
    name_to_row = dict(zip(tbl.filename, tbl.row))

    return np.array([name_to_row[f] for f in filenames], dtype=np.int32)


def rows_to_paths(
    rows: np.ndarray,
    csv_path: str | Path = "image_filenames.csv",
) -> np.ndarray:
    tbl = load_table(csv_path)
    return tbl["path"].to_numpy()[rows]


def filenames_to_paths(
    filenames: Iterable[str],
    csv_path: str | Path = "image_filenames.csv",
) -> np.ndarray:
    rows = filenames_to_rows(filenames, csv_path)
    return rows_to_paths(rows, csv_path)


@lru_cache(maxsize=1)
def load_table(
    csv_path: str | Path = "image_filenames.csv",
) -> pd.DataFrame:  # noqa: D401
    """Return the master image table (cached)."""
    return pd.read_csv(csv_path)


def get_feature_path(feature_path: str, row: pd.Series) -> Path:
    return (
        Path(feature_path) / row.comparison / row.weights / row.name / row.penultimate
    )


def load_features(
    feature_path: str,
    model_info_csv: str = "model_info.csv",
    image_info: str = "image_info.csv",
    filter_from_filenames: list[str] | None = None,
    behavior: bool = False,
    plus: bool = False,
    num_per_category: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    base_path = Path(feature_path)

    if num_per_category and filter_from_filenames is not None:
        raise ValueError(
            "num_per_category and filter_from_filenames cannot be used together"
        )

    model_info_csv = Path(feature_path) / model_info_csv
    image_info = Path(feature_path) / image_info
    info = load_table(image_info)
    rows = info.row.values
    filenames = info.filename.values
    full_paths = info.path.values

    if filter_from_filenames is not None:
        rows = filter_items_to_rows(filter_from_filenames, image_info)
    elif num_per_category is not None:
        rows, _ = limit_per_category(filenames, num_per_category)
    else:
        mask_bool, _ = mask_filenames(filenames, plus=plus, behavior=behavior)
        rows = rows[mask_bool]

    info = pd.read_csv(model_info_csv, comment="#")
    feats: dict[str, np.ndarray] = {}
    for rec in info.itertuples(index=False, name="ModelInfo"):
        feat_path = get_feature_path(base_path, rec) / "features.npy"
        arr = np.load(feat_path)[rows]
        feats[f"{rec.name}.{rec.weights}.{rec.comparison}"] = arr

    return dict(feats), filenames[rows], full_paths[rows]


@register_dataset("dnn")
class DNN(BaseDatasetLoader):
    def __init__(self, root: str = None, **kwargs):
        super().__init__("dnn", root, **kwargs)
        self.kwargs = kwargs

    def load(self) -> BaseDataset:
        features, filenames, image_paths = load_features(
            self.root,
            **self.kwargs,
        )

        return BaseDataset(
            name="dnn",
            data=features,
            filenames=filenames,
            image_paths=image_paths,
        )
