import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}


def load_shared_data(
    things_data: Path | str,
    things_images_path: Path,
    num_dims: int,
) -> tuple:
    """Load all shared data for experiments."""

    things_data = Path(things_data)

    spose_embedding = load_spose_embedding(num_dims=num_dims)
    indices_48 = load_concept_mappings(
        things_data / "words48.csv", things_images_path, CATEGORY_REPLACEMENTS
    )
    rsm_48_true = load_words48(things_data / "rdm48_human.mat")
    return spose_embedding, indices_48, rsm_48_true


def load_triplets(things_data: Path | str) -> np.ndarray:
    things_data = Path(things_data)
    train_triplets = np.loadtxt(things_data / "triplets" / "trainset.txt").astype(int)
    validation_triplets = np.loadtxt(
        things_data / "triplets" / "validationset.txt"
    ).astype(int)
    return train_triplets, validation_triplets


def load_spose_embedding(max_objects=None, max_dims=None, num_dims=66):
    if num_dims == 66:
        path = Path("data/things/spose_embedding_66d.txt")
    elif num_dims == 49:
        path = Path("data/things/spose_embedding_49d.txt")
    else:
        raise ValueError(f"Invalid number of dimensions: {num_dims}")
    x = np.maximum(np.loadtxt(path), 0)

    if max_objects:
        objects = np.arange(x.shape[0])
        random_objects = np.random.choice(objects, size=max_objects, replace=False)
        x = x[random_objects]
    if max_dims:
        x = x[:, :max_dims]
    return x


def load_concept_mappings(
    words_path: str, things_image_path: str, replacements: dict
) -> list:
    words48 = pd.read_csv(words_path)
    cls_names = [replacements.get(name, name) for name in words48["Word"].values]

    images = load_things_image_data(things_image_path, filter_behavior=True)
    categories = [" ".join(Path(f).stem.split("_")[0:-1]) for f in images]

    return [categories.index(c) for c in cls_names]


def load_words48(words48_path: str) -> np.ndarray:
    rdm = loadmat(words48_path)["RDM48_triplet"]
    return 1 - rdm


def load_things_image_data(
    img_root: str,
    filter_behavior: bool = False,
    filter_plus: bool = False,
    return_indices: bool = False,
    filter_from_filenames: list[str] | None = None,
    single_category: bool = False,
) -> list[int]:
    """Load image data from a folder"""

    def filter_image_names(filter_criterion, img_names):
        return [i for i, img in enumerate(img_names) if filter_criterion in img]

    image_paths = glob.glob(os.path.join(img_root, "**", "*.jpg"))

    image_paths = sorted(
        image_paths, key=lambda x: (str(Path(x).parent), str(os.path.basename(x)))
    )

    indices = np.arange(len(image_paths))
    img_names = [os.path.basename(img) for img in image_paths]

    if filter_from_filenames is not None:
        indices = [
            i
            for i, path in enumerate(image_paths)
            if Path(path).name in filter_from_filenames
        ]

        # img_names = np.array(img_names)[indices]

        if single_category:
            image_paths = np.array(image_paths)[indices]
            # i have a list of categories eg aardvark01s, aardvark01b, aardvark01c, antilope01s, antilope01b, antilope01c
            # and i want to get only a single image from each category

            # If single_category is True, take only the first image from each category
            categories = dict.fromkeys(
                Path(img).name.rsplit("_", 1)[0] for img in image_paths
            )

            cat_to_idx = {}
            for i, img in enumerate(image_paths):
                cat = Path(img).name.rsplit("_", 1)[0]
                if cat not in cat_to_idx:
                    cat_to_idx[cat] = i

            indices = list(cat_to_idx.values())
            return indices, image_paths[indices]

    elif filter_behavior:
        indices = filter_image_names("01b", img_names)

    elif filter_plus:
        indices = filter_image_names("plus", img_names)

    image_paths = np.array(image_paths)[indices]

    if return_indices:
        return indices, image_paths
    else:
        return image_paths
