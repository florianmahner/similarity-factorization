"""This is a base configuration for datasets.

It is used to map dataset names to their storage paths.

To add a new dataset, you need to:
1. Add the dataset name and path to the DATASET_PATHS dictionary.
2. Add the dataset loading function to the registry.py file.
3. Add the dataset to the __init__.py file.
"""

DATASET_PATHS = {
    "mnist": "/SSD/datasets/mnist",
    "iris": "/SSD/datasets/iris",
    "diabetes": "/SSD/datasets/diabetes",
    "digits": "/SSD/datasets/digits",
    "peterson-various": "/SSD/datasets/similarity_datasets/peterson/various",
    "peterson-animals": "/SSD/datasets/similarity_datasets/peterson/animals",
    "nsd": "/LOCAL/LABSHARE/natural-scenes-dataset/",
    "haxby-faces": "/SSD/datasets/haxby-2001",
    "cichy118": "/SSD/datasets/similarity_datasets/cichy118",
    "mur92": "/SSD/datasets/similarity_datasets/mur92",
    "orl": "/SSD/datasets/orl",
    "things-monkey-2k": "/LOCAL/fmahner/monkey-dimensions/data/monkey/2k/f",
    "things-monkey-22k": "/LOCAL/fmahner/monkey-dimensions/data/monkey/22k/f",
    # Additional tabular datasets (sklearn built-in, using cache directory)
    "wine": "/SSD/datasets/sklearn_cache",
    "breast_cancer": "/SSD/datasets/sklearn_cache",
    # Text datasets (sklearn fetch functions use their own cache)
    "20newsgroups": "/SSD/datasets/20newsgroups",
    "20newsgroups_full": "/SSD/datasets/20newsgroups_full",
    "dnn": "/SSD/projects/deepsim/raw/features/",
}


def get_dataset_path(name: str) -> str:
    """Retrieve the default path for a dataset, or raise an error if not found."""
    if name not in DATASET_PATHS:
        raise ValueError(
            f"No root path is registered for dataset '{name}'. Available: {list(DATASET_PATHS.keys())}"
        )

    return DATASET_PATHS.get(name, None)
