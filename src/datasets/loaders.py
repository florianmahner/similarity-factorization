"""Dataset loaders for neuroscience and machine learning datasets."""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.datasets import (
    fetch_20newsgroups,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from torchvision.datasets import MNIST as MNIST_torch

from config import get_dataset_path
from tools.rsa import compute_similarity

from .base import DatasetResult
from .nsd_utils import (
    get_available_subjects,
    get_roi,
    load_nsd_betas,
    load_nsd_images,
)

ndarray = np.ndarray


def group_level_rsa(
    data: list[ndarray], metric: str = "cosine"
) -> tuple[ndarray, list[ndarray]]:
    """Compute group-level RSM from multiple subjects."""
    from tools.rsa import compute_rsm
    from tools.stats import apply_transform

    if metric == "linear":
        data = [apply_transform(d, "standardize") for d in data]

    per_subject_rsms = [compute_rsm(s, metric=metric) for s in data]

    if metric == "cosine":
        group_level_rsm = np.mean(per_subject_rsms, axis=0)
    elif metric == "pearson":
        group_level_rsm = _average_pearson_rsm(per_subject_rsms)
    elif metric == "linear":
        group_level_rsm = np.mean(per_subject_rsms, axis=0)
    else:
        raise ValueError(f"Metric {metric} not supported")

    return group_level_rsm, per_subject_rsms


def _fisher_z(r: ndarray, epsilon: float = 1e-6) -> ndarray:
    """Fisher z-transformation for correlation coefficients."""
    r = np.clip(r, -1 + epsilon, 1 - epsilon)
    return np.arctanh(r)


def _inverse_fisher_z(z: ndarray) -> ndarray:
    """Inverse Fisher z-transformation."""
    return np.tanh(z)


def _average_pearson_rsm(rsms: list[ndarray]) -> ndarray:
    """Average Pearson RSMs using Fisher z-transformation."""
    z_list = [_fisher_z(rsm) for rsm in rsms]
    avg_z = np.mean(z_list, axis=0)
    return _inverse_fisher_z(avg_z)


def load_mur92(root: str | None = None) -> DatasetResult:
    """
    Load Mur92 fMRI dataset (92 objects, IT cortex, 15 subjects).

    Parameters
    ----------
    root : str, optional
        Path to dataset directory

    Returns
    -------
    DatasetResult
        Dataset with group_rsm, subject_rsms, mri_data, images
    """
    root = Path(root or get_dataset_path("mur92"))
    image_folder = root / "images"
    images = sorted(glob(f"{image_folder}/*.jpg"))
    mri_dir = root / "fmri_roidata_new_all"
    mri_file = mri_dir / "92_fmri_hvc_raw_new_unconstrained_single.mat"
    mri_data = loadmat(mri_file)["data"].ravel()
    mri_data = [d.T for d in mri_data]

    group_level_rsm, per_subject_rsms = group_level_rsa(mri_data, metric="pearson")

    return DatasetResult(
        name="mur92",
        rsm=group_level_rsm,
        metadata={
            "subject_rsms": per_subject_rsms,
            "mri_data": mri_data,
            "images": images,
        },
    )


def load_cichy118(root: str | None = None) -> DatasetResult:
    """
    Load Cichy118 MEG dataset (118 objects, 15 subjects).

    Parameters
    ----------
    root : str, optional
        Path to dataset directory

    Returns
    -------
    DatasetResult
        Dataset with group_rsm, subject_rsms
    """
    root = Path(root or get_dataset_path("cichy118"))
    mat_file = root / "cichy_118_rdms.mat"
    data = loadmat(mat_file)["data"].ravel()
    data = [d.T for d in data]

    group_level_rsm, per_subject_rsms = group_level_rsa(data, metric="pearson")

    return DatasetResult(
        name="cichy118",
        rsm=group_level_rsm,
        metadata={"subject_rsms": per_subject_rsms},
    )


def load_peterson(root: str | None = None, variant: str = "animals") -> DatasetResult:
    """
    Load Peterson fMRI dataset (animals or various objects).

    Parameters
    ----------
    root : str, optional
        Path to dataset directory
    variant : str, default='animals'
        Either 'animals' or 'various'

    Returns
    -------
    DatasetResult
        Dataset with rsm
    """
    if variant == "animals":
        root = Path(root or get_dataset_path("peterson-animals"))
    elif variant == "various":
        root = Path(root or get_dataset_path("peterson-various"))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    rsm_file = root / "rsm.npy"
    if rsm_file.exists():
        rsm = np.load(rsm_file)
    else:
        mat_file = root / f"peterson_rdm_{variant}_all.mat"
        data = loadmat(mat_file)
        rsm = data["RSM_4dim_merged"]

    return DatasetResult(name=f"peterson-{variant}", rsm=rsm)


def load_nsd(
    root: str | None = None,
    subject_id: int = 1,
    zscore_betas: bool = True,
    roi_name: str = "streams",
    space: str = "func1pt8mm",
) -> DatasetResult:
    """
    Load Natural Scenes Dataset (NSD) fMRI data.

    Parameters
    ----------
    root : str, optional
        Path to NSD dataset directory
    subject_id : int, default=1
        Subject ID (1-8)
    zscore_betas : bool, default=True
        Whether to z-score beta values
    roi_name : str, default='streams'
        ROI name (e.g., 'streams', 'floc-faces')
    space : str, default='func1pt8mm'
        Brain space

    Returns
    -------
    DatasetResult
        Dataset with betas, images, categories
    """
    root = Path(root or get_dataset_path("nsd"))

    subjects = get_available_subjects(root)
    if subject_id not in subjects:
        raise ValueError(f"Subject {subject_id} not found. Available: {subjects}")

    roi = get_roi(subject_id, roi_name, root, space=space)
    betas, trials_with_betas = load_nsd_betas(
        subject_id, zscore_betas, root, space, voxel_indices=roi
    )
    images, categories = load_nsd_images(trials_with_betas, root)

    return DatasetResult(
        name="nsd",
        data=betas,
        metadata={"images": images, "categories": categories},
    )


def _get_monkey_channel_mask(monkey_type: str, roi: str | None = None):
    """Get channel mask for monkey data by ROI."""
    if roi is None:
        return slice(None)

    if monkey_type == "N":
        masks = {"v1": slice(0, 512), "v4": slice(512, 768), "it": slice(768, 1024)}
    else:
        masks = {"v1": slice(0, 512), "it": slice(512, 832), "v4": slice(832, 1024)}

    return masks[roi.lower()]


def load_things_monkey(
    root: str | None = None,
    monkey_type: str = "F",
    roi: str | None = "it",
    min_reliab: float = 0.6,
) -> DatasetResult:
    """
    Load THINGS monkey neural data (22k images).

    Parameters
    ----------
    root : str, optional
        Path to monkey data directory
    monkey_type : str, default='F'
        Monkey identifier ('F' or 'N')
    roi : str, optional, default='it'
        ROI name ('v1', 'v4', 'it')
    min_reliab : float, default=0.6
        Minimum reliability threshold for channels

    Returns
    -------
    DatasetResult
        Dataset with neural data, rsm, filenames
    """
    import h5py
    
    root = Path(root or get_dataset_path("things-monkey-22k"))
    mat_path = root / "THINGS_normMUA_raw.mat"
    
    with h5py.File(mat_path, 'r') as f:
        data_key = f"data_{roi}"
        reliab_key = f"reliab_{roi}"
        
        data = f[data_key][:].astype("float32")
        reliab = f[reliab_key][:].mean(axis=0)

    if min_reliab is not None:
        reliab_mask = reliab >= min_reliab
        data = data[:, reliab_mask]

    rsm = compute_similarity(data, data, "gaussian_kernel")

    return DatasetResult(
        name="things-monkey-22k",
        data=data,
        rsm=rsm,
        metadata={"it": data},
    )


def load_iris(root: str | None = None) -> DatasetResult:
    """Load Iris dataset."""
    iris = load_iris()
    rsm = np.corrcoef(iris.data)
    return DatasetResult(name="iris", data=iris.data, targets=iris.target, rsm=rsm)


def load_diabetes(root: str | None = None) -> DatasetResult:
    """Load Diabetes dataset."""
    diabetes = load_diabetes(as_frame=True)
    return DatasetResult(
        name="diabetes",
        data=diabetes.data.to_numpy(),
        targets=diabetes.target.to_numpy(),
    )


def load_digits(root: str | None = None) -> DatasetResult:
    """Load Digits dataset."""
    digits = load_digits()
    return DatasetResult(name="digits", data=digits.data, targets=digits.target)


def load_mnist(root: str | None = None) -> DatasetResult:
    """Load MNIST dataset."""
    root = root or get_dataset_path("mnist")
    train = MNIST_torch(root, train=True, download=True)
    test = MNIST_torch(root, train=False, download=True)

    return DatasetResult(
        name="mnist",
        metadata={
            "train_data": train.data.numpy(),
            "train_targets": train.targets.numpy(),
            "test_data": test.data.numpy(),
            "test_targets": test.targets.numpy(),
        },
    )


def load_wine(root: str | None = None) -> DatasetResult:
    """Load Wine dataset."""
    wine = load_wine()
    rsm = np.corrcoef(wine.data)
    return DatasetResult(name="wine", data=wine.data, targets=wine.target, rsm=rsm)


def load_breast_cancer(root: str | None = None) -> DatasetResult:
    """Load Breast Cancer dataset."""
    cancer = load_breast_cancer()
    return DatasetResult(name="breast_cancer", data=cancer.data, targets=cancer.target)


def load_orl(root: str | None = None) -> DatasetResult:
    """Load ORL faces dataset."""
    root = Path(root or get_dataset_path("orl"))
    file = root / "ORL.mat"
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")
    data = loadmat(file)
    return DatasetResult(name="orl", data=data["data"], targets=data["label"].squeeze())


def load_20newsgroups(
    root: str | None = None,
    categories: list[str] | None = None,
    max_features: int = 1000,
) -> DatasetResult:
    """Load 20 Newsgroups text dataset (4 categories by default)."""
    categories = categories or [
        "alt.atheism",
        "comp.graphics",
        "sci.space",
        "talk.religion.misc",
    ]

    newsgroups = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features, stop_words="english", max_df=0.95, min_df=2
    )
    X = vectorizer.fit_transform(newsgroups.data).toarray()

    return DatasetResult(
        name="20newsgroups",
        data=X,
        targets=newsgroups.target,
        metadata={
            "feature_names": vectorizer.get_feature_names_out(),
            "target_names": newsgroups.target_names,
        },
    )


def load_20newsgroups_full(
    root: str | None = None, max_features: int = 2000
) -> DatasetResult:
    """Load 20 Newsgroups text dataset (all 20 categories)."""
    newsgroups = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features, stop_words="english", max_df=0.95, min_df=2
    )
    X = vectorizer.fit_transform(newsgroups.data).toarray()

    return DatasetResult(
        name="20newsgroups_full",
        data=X,
        targets=newsgroups.target,
        metadata={
            "feature_names": vectorizer.get_feature_names_out(),
            "target_names": newsgroups.target_names,
        },
    )


DATASETS = {
    "mur92": load_mur92,
    "cichy118": load_cichy118,
    "peterson-animals": lambda **kwargs: load_peterson(variant="animals", **kwargs),
    "peterson-various": lambda **kwargs: load_peterson(variant="various", **kwargs),
    "nsd": load_nsd,
    "things-monkey-22k": load_things_monkey,
    "iris": load_iris,
    "diabetes": load_diabetes,
    "digits": load_digits,
    "mnist": load_mnist,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "orl": load_orl,
    "20newsgroups": load_20newsgroups,
    "20newsgroups_full": load_20newsgroups_full,
}


def load_dataset(name: str, **kwargs) -> DatasetResult:
    """
    Load dataset by name.

    Parameters
    ----------
    name : str
        Dataset name
    **kwargs : dict
        Dataset-specific parameters

    Returns
    -------
    DatasetResult
        Loaded dataset

    Raises
    ------
    ValueError
        If dataset name not registered

    Examples
    --------
    >>> ds = load_dataset('mur92')
    >>> ds.rsm.shape
    (92, 92)

    >>> ds = load_dataset('nsd', subject_id=2, roi_name='streams')
    >>> ds.data.shape
    (9841, 426)
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: '{name}'. Available: {list(DATASETS.keys())}"
        )
    return DATASETS[name](**kwargs)
