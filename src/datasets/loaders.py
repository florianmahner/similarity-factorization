"""Dataset loaders for neuroscience and machine learning datasets."""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn import datasets as sklearn_datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.rsa import compute_similarity

from .base import DatasetResult
from .nsd_utils import (
    get_available_subjects,
    load_nsd_data,
)
from .swow import load_swow_ppmi, load_swow_similarity

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
    root = Path(root)
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
    root = Path(root)
    mat_file = (
        root / "fmri_roidata_new_all" / "118_fmri_hvc_raw_new_unconstrained_single.mat"
    )
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
    root = Path(root) / variant

    rsm = np.ascontiguousarray(np.load(root / "rsm.npy"))
    images = sorted(glob(f"{root}/images/*.png"))

    return DatasetResult(
        name=f"peterson-{variant}", rsm=rsm, metadata={"images": images}
    )


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
    root = Path(root)

    subjects = get_available_subjects(root)
    if subject_id not in subjects:
        raise ValueError(f"Subject {subject_id} not found. Available: {subjects}")

    betas, images = load_nsd_data(
        subject_id, roi_name, space, zscore_betas, return_images=True, nsd_dir=root
    )

    return DatasetResult(
        name="nsd",
        data=betas,
        metadata={"images": images},
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
    roi: str = "it",
    min_reliab: float = 0.6,
) -> DatasetResult:
    """
    Load THINGS monkey neural data (22k images).

    Parameters
    ----------
    root : str, optional
        Path to monkey data directory (default: /SSD/fmahner/macaque_florian/22k)
    monkey_type : str, default='F'
        Monkey identifier ('F' or 'N')
    roi : str, default='it'
        ROI name ('v1', 'v4', 'it')
    min_reliab : float, default=0.6
        Minimum reliability threshold for channels

    Returns
    -------
    DatasetResult
        Dataset with neural data, filenames
    """
    import h5py

    if root is None:
        root = Path("/SSD/fmahner/macaque_florian/22k")
    else:
        root = Path(root)

    data_dir = root / monkey_type.lower()
    mat_path = data_dir / "THINGS_normMUA_raw.mat"

    with h5py.File(mat_path, "r") as f:
        data_key = f"data_{roi}"
        reliab_key = f"reliab_{roi}"

        data = f[data_key][:].astype("float32")
        reliab = f[reliab_key][:].mean(axis=0)

    if min_reliab is not None:
        reliab_mask = reliab >= min_reliab
        data = data[:, reliab_mask]

    filenames = np.loadtxt(data_dir / "index_to_image.txt", dtype=str)

    return DatasetResult(
        name="things-monkey-22k",
        data=data,
        rsm=None,
        metadata={
            "filenames": filenames,
            "monkey_type": monkey_type,
            "roi": roi,
            "n_channels": data.shape[1],
        },
    )


def load_things_monkey_2k(
    root: str | None = None,
    recording: str = "N_combined",
    roi: str = "it",
) -> DatasetResult:
    """
    Load preprocessed THINGS monkey 2k data (1854 stimuli).

    Parameters
    ----------
    root : str, optional
        Path to processed data directory
    recording : str, default='N_combined'
        Recording to load: 'N1', 'N2', 'F', or 'N_combined'
    roi : str, default='it'
        ROI name ('v1', 'v4', 'it')

    Returns
    -------
    DatasetResult
        Dataset with neural data (n_stimuli, n_channels), stimuli names
    """
    if root is None:
        root = Path(__file__).parent.parent.parent / "data" / "things-monkey" / "2k" / "processed"
    else:
        root = Path(root)

    path = root / "final" / f"{recording}_{roi}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            f"Run preprocessing first: experiments/preprocessing/monkey_2k/"
        )

    data = np.load(path)
    neural_data = data["data"]
    rsm = data["rsm"]
    stimuli = data["stimuli"]
    reliability = data["reliability"]

    return DatasetResult(
        name="things-monkey-2k",
        data=neural_data,
        rsm=rsm,
        metadata={
            "stimuli": stimuli,
            "reliability": reliability,
            "recording": recording,
            "roi": roi,
            "n_channels": neural_data.shape[1],
        },
    )


def load_iris(root: str | None = None) -> DatasetResult:
    """Load Iris dataset."""
    iris = sklearn_datasets.load_iris()
    rsm = np.corrcoef(iris.data)
    return DatasetResult(name="iris", data=iris.data, targets=iris.target, rsm=rsm)


def load_diabetes(root: str | None = None) -> DatasetResult:
    """Load Diabetes dataset."""
    diabetes = sklearn_datasets.load_diabetes(as_frame=True)
    return DatasetResult(
        name="diabetes",
        data=diabetes.data.to_numpy(),
        targets=diabetes.target.to_numpy(),
    )


def load_digits(root: str | None = None) -> DatasetResult:
    """Load Digits dataset."""
    digits = sklearn_datasets.load_digits()
    return DatasetResult(name="digits", data=digits.data, targets=digits.target)


def load_wine(root: str | None = None) -> DatasetResult:
    """Load Wine dataset."""
    wine = sklearn_datasets.load_wine()
    rsm = np.corrcoef(wine.data)
    return DatasetResult(name="wine", data=wine.data, targets=wine.target, rsm=rsm)


def load_breast_cancer(root: str | None = None) -> DatasetResult:
    """Load Breast Cancer dataset."""
    cancer = sklearn_datasets.load_breast_cancer()
    return DatasetResult(name="breast_cancer", data=cancer.data, targets=cancer.target)


def load_orl(root: str | None = None) -> DatasetResult:
    """Load ORL faces dataset."""
    root = Path(root)
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


def load_swow_data_paper_format(data_dir, use_all_responses=True):
    """
    Load SWOW data following the paper's preprocessing pipeline.

    This loads the strength file (preprocessed cue-response associations)
    and returns it in long format for graph construction.

    Parameters
    ----------
    data_dir : Path
        Directory containing SWOW data files
    use_all_responses : bool
        If True, use R123 (all 3 responses). If False, use R1 only.

    Returns
    -------
    df_long : DataFrame
        Long format data with columns [cue, response, count, N, strength]
    """
    data_dir = Path(data_dir)

    # Use R123 (all 3 responses) for richer associations
    if use_all_responses:
        strength_file = data_dir / "strength.SWOW-EN.R123.20180827.csv"
        print(f"Loading SWOW-EN R123 data (all 3 responses)")
    else:
        strength_file = data_dir / "strength.SWOW-EN.R1.20180827.csv"
        print(f"Loading SWOW-EN R1 data (first response only)")

    print(f"Reading from {strength_file}")
    df = pd.read_csv(strength_file, sep="\t")

    print(f"Loaded {len(df)} cue-response associations")
    print(f"  Unique cues: {df['cue'].nunique()}")
    print(f"  Unique responses: {df['response'].nunique()}")

    # Standardize column names
    if "R123.Strength" in df.columns:
        df["strength"] = df["R123.Strength"]
        df["count"] = df["R123"]
    elif "R1.Strength" in df.columns:
        df["strength"] = df["R1.Strength"]
        df["count"] = df["R1"]
    else:
        raise ValueError("Cannot find strength column in SWOW data")

    return df


def load_swow(
    root: str | None = None,
    similarity_method: str = "ppmi",
    use_all_responses: bool = False,
    top_n_words: int | None = None,
    min_word_length: int = 1,
    symmetrization: str = "sum",
    bidirectional_only: bool = False,
    alpha: float = 0.75,
) -> DatasetResult:
    """
    Load SWOW word association data as similarity matrix.

    Parameters
    ----------
    root : str
        Path to SWOW data directory
    similarity_method : str
        Similarity method: 'ppmi' (local) or 'rw' (random walk, global)
    use_all_responses : bool
        If True, use R123 (all responses). If False, use R1 only.
    top_n_words : int | None
        Keep only top N words by degree (None = all)
    min_word_length : int
        Minimum word length filter
    symmetrization : str
        Method to symmetrize (PPMI only): 'sum', 'mean', 'geometric_mean'
    bidirectional_only : bool
        If True, keep only bidirectional edges (PPMI only)
    alpha : float
        Katz walk damping parameter (RW only, default 0.75)

    Returns
    -------
    DatasetResult
        Dataset with similarity matrix as rsm and vocabulary in metadata
    """
    root = Path(root)
    similarity, vocabulary, metadata = load_swow_similarity(
        root,
        method=similarity_method,
        use_all_responses=use_all_responses,
        top_n_words=top_n_words,
        min_word_length=min_word_length,
        symmetrization=symmetrization,
        bidirectional_only=bidirectional_only,
        alpha=alpha,
    )
    return DatasetResult(
        name="swow",
        rsm=similarity,
        metadata={"vocabulary": vocabulary, **metadata},
    )


def load_dnn_features(
    root: str | None = None,
    layer: str | None = None,
    filter_plus: bool = False,
    image_info_path: str | None = None,
) -> DatasetResult:
    """
    Load DNN features from deepsim feature extraction.

    Parameters
    ----------
    root : str
        Path to model features directory
    layer : str
        Layer to load. If None, auto-detects the only available layer.
    filter_plus : bool
        If True, filter to only THINGS+ images (1854 behavioral subset)
    image_info_path : str
        Path to image_info.csv for filtering

    Returns
    -------
    DatasetResult
        Dataset with features (n_images, n_features)

    Available models (pass as root):
        CLIP models (layer='visual'):
        - /SSD/projects/deepsim/raw/features/dataset/openai/ViT-L-14
        - /SSD/projects/deepsim/raw/features/dataset/laion2b_s32b_b82k/ViT-L-14

        ImageNet models (layer auto-detected):
        - /SSD/projects/deepsim/raw/features/architecture/IMAGENET1K_V1/resnet50
        - /SSD/projects/deepsim/raw/features/architecture/IMAGENET1K_V1/vgg16_bn
        - /SSD/projects/deepsim/raw/features/architecture/IMAGENET1K_V1/convnext_large
        - /SSD/projects/deepsim/raw/features/architecture/IMAGENET1K_V1/swin_b
    """
    import pandas as pd

    root = Path(root)

    if layer is None:
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            layer = subdirs[0].name
        else:
            raise ValueError(f"Multiple layers found in {root}, specify one: {[d.name for d in subdirs]}")

    features = np.load(root / layer / "features.npy")

    metadata = {"layer": layer, "model": root.name}

    if filter_plus:
        if image_info_path is None:
            image_info_path = "/SSD/projects/deepsim/raw/features/image_info.csv"
        info = pd.read_csv(image_info_path)
        plus_mask = info["filename"].str.contains("_plus")
        features = features[plus_mask.values]
        categories = info.loc[plus_mask, "category"].tolist()
        metadata["filenames"] = info.loc[plus_mask, "filename"].tolist()
        metadata["categories"] = categories
        metadata["paths"] = [
            f"/SSD/datasets/things/behav1854/{cat}/{cat}_01b.jpg" for cat in categories
        ]
        metadata["filtered"] = "plus"

    metadata["n_images"] = features.shape[0]
    metadata["n_features"] = features.shape[1]

    return DatasetResult(
        name="dnn",
        data=features,
        metadata=metadata,
    )


# Alias for backward compatibility
load_vit = load_dnn_features


DATASETS = {
    "mur92": load_mur92,
    "cichy118": load_cichy118,
    "vit": load_vit,
    "dnn": load_dnn_features,
    "peterson-animals": lambda **kwargs: load_peterson(variant="animals", **kwargs),
    "peterson-various": lambda **kwargs: load_peterson(variant="various", **kwargs),
    "nsd": load_nsd,
    "swow": load_swow,
    "things-monkey-2k": load_things_monkey_2k,
    "things-monkey-22k": load_things_monkey,
    "iris": load_iris,
    "diabetes": load_diabetes,
    "digits": load_digits,
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
