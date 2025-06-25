import glob
import os
from pathlib import Path

import numpy as np


def load_spose_embedding(path: str) -> np.ndarray:
    return np.maximum(np.loadtxt(path), 0)


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
