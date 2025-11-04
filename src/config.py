"""Configuration for dataset paths."""

import os
from pathlib import Path

DATASET_PATHS = {
    "peterson-various": os.getenv(
        "DATASET_PETERSON_VARIOUS", "/SSD/datasets/similarity_datasets/peterson/various"
    ),
    "peterson-animals": os.getenv(
        "DATASET_PETERSON_ANIMALS", "/SSD/datasets/similarity_datasets/peterson/animals"
    ),
    "nsd": os.getenv("DATASET_NSD", "/LOCAL/LABSHARE/natural-scenes-dataset"),
    "cichy118": os.getenv(
        "DATASET_CICHY118", "/SSD/datasets/similarity_datasets/cichy118"
    ),
    "mur92": os.getenv("DATASET_MUR92", "/SSD/datasets/similarity_datasets/mur92"),
    "things-monkey-22k": os.getenv("DATASET_MONKEY_22K", "/ptmp/fmahner/monkey-22k/f"),
    "vit": os.getenv(
        "DATASET_VIT",
        "/SSD/projects/deepsim/raw/features/dataset/openai/ViT-L-14/visual/features.npy",
    ),
    "things-ooo": os.getenv(
        "DATASET_THINGS_OOO", "/LOCAL/fmahner/similarity-factorization/data/things"
    ),
}


def get_dataset_path(name: str) -> str:
    """
    Retrieve the default path for a dataset.

    Parameters
    ----------
    name : str
        Dataset name

    Returns
    -------
    str
        Path to dataset root directory

    Raises
    ------
    ValueError
        If dataset name not registered
    """
    if name not in DATASET_PATHS:
        raise ValueError(
            f"No root path registered for dataset '{name}'. "
            f"Available: {list(DATASET_PATHS.keys())}"
        )
    return DATASET_PATHS[name]
