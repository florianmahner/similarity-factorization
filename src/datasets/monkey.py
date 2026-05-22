"""THINGS macaque neural data utilities.

Provides loaders for both 2k (1854 images × 1 exemplar) and 22k (1854 × 12 exemplars)
datasets from /SSD/datasets/things/macaque/.

Usage:
    from src.datasets.monkey import load_macaque

    # Load 2k IT data
    data, stimuli = load_macaque("2k", "F", roi="it", min_reliab=0.3)

    # Load 22k (category-averaged)
    data, stimuli = load_macaque("22k", "F", roi="it", min_reliab=0.3, average_exemplars=True)
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numba import njit, prange

DATA_ROOT = Path("/SSD/datasets/things/macaque")

ROI_SLICES = {
    "N": {"v1": (0, 512), "v4": (512, 768), "it": (768, 1024)},
    "F": {"v1": (0, 512), "it": (512, 832), "v4": (832, 1024)},
}

# Legacy paths for preprocessing scripts
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "things-monkey" / "2k"
PROCESSED_DIR = DATA_DIR / "processed"

TIME_WINDOWS = {"v1": (25, 125), "v4": (50, 150), "it": (75, 175)}
BASELINE_WINDOW = (-100, 0)

RECORDINGS = {
    "N1": {"folder": "N1", "monkey": "N", "time_resolved": False},
    "N2": {"folder": "N2", "monkey": "N", "time_resolved": True},
    "F": {"folder": "F", "monkey": "F", "time_resolved": True},
}


# =============================================================================
# Stimulus metadata
# =============================================================================


def load_stimulus_classes() -> tuple[np.ndarray, np.ndarray]:
    """Load all 2000 stimulus classes and identify THINGS vs faces.

    Returns
    -------
    classes : array of shape (2000,)
        Stimulus class names
    is_things : array of shape (2000,)
        True for THINGS objects, False for face stimuli
    """
    with h5py.File(DATA_ROOT / "things2_imgs.mat", "r") as f:
        n = f["test_imgs/class"].shape[0]
        classes = []
        for i in range(n):
            ref = f["test_imgs/class"][i, 0]
            name = "".join(chr(int(c)) for c in f[ref][()].flatten() if c < 128)
            classes.append(name)

    classes = np.array(classes)

    # Faces: monkey01-73, person01-73 (but not 'monkey' the animal)
    is_things = np.array([
        not (c.startswith("monkey") and len(c) > 6) and not c.startswith("person")
        for c in classes
    ])

    return classes, is_things


def get_things_classes() -> np.ndarray:
    """Get sorted list of 1854 THINGS class names (excluding faces)."""
    classes, is_things = load_stimulus_classes()
    return np.sort(classes[is_things])


# =============================================================================
# Path helpers
# =============================================================================


def get_intermediate_path(recording: str, roi: str) -> Path:
    return PROCESSED_DIR / "intermediate" / f"{recording}_{roi}.npz"


def get_reliability_path(recording: str, roi: str) -> Path:
    return PROCESSED_DIR / "reliability" / f"{recording}_{roi}.npy"


def get_output_path(recording: str, roi: str) -> Path:
    return PROCESSED_DIR / "final" / f"{recording}_{roi}.npz"


# =============================================================================
# Data loading
# =============================================================================


def load_time_resolved(mat_path: Path, monkey: str, roi: str) -> tuple[np.ndarray, np.ndarray]:
    """Load time-resolved data and compute baseline-subtracted time average.

    Returns
    -------
    data : array of shape (n_trials, n_channels)
    stim_ids : array of shape (n_trials,)
        1-indexed stimulus IDs
    """
    ch_start, ch_end = ROI_SLICES[monkey][roi]
    t_start, t_end = TIME_WINDOWS[roi]

    with h5py.File(mat_path, "r") as f:
        tb = np.array(f["tb"]).flatten()
        stim_ids = np.array(f["ALLMAT"][1]).astype(np.int32)

        baseline_idx = np.where((tb >= BASELINE_WINDOW[0]) & (tb < BASELINE_WINDOW[1]))[0]
        response_idx = np.where((tb > t_start) & (tb <= t_end))[0]

        baseline = np.array(f["ALLMUA"][baseline_idx[0]:baseline_idx[-1]+1, :, ch_start:ch_end])
        response = np.array(f["ALLMUA"][response_idx[0]:response_idx[-1]+1, :, ch_start:ch_end])

    data = np.nanmean(response, axis=0) - np.nanmean(baseline, axis=0)
    return data.astype(np.float32), stim_ids


def load_time_averaged(mat_path: Path, roi: str) -> np.ndarray:
    """Load already time-averaged data (monkeyN session1)."""
    with h5py.File(mat_path, "r") as f:
        return np.array(f[f"data_{roi}"]).astype(np.float32)


def load_zi_list(recording: str) -> np.ndarray:
    """Load stimulus labels for a recording."""
    folder = RECORDINGS[recording]["folder"]
    return pd.read_csv(DATA_DIR / folder / "zi_list.csv", header=None)[0].values


def load_intermediate(recording: str, roi: str) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed intermediate data (time-averaged trials)."""
    data = np.load(get_intermediate_path(recording, roi), allow_pickle=True)
    return data["data"], data["stim_labels"]


# =============================================================================
# Trial tensor operations
# =============================================================================


def build_trial_tensor(
    data: np.ndarray,
    labels: np.ndarray,
    target_stimuli: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Organize trials into (n_stimuli, max_reps, n_channels) tensor.

    Parameters
    ----------
    data : array of shape (n_trials, n_channels)
    labels : array of shape (n_trials,)
    target_stimuli : array of shape (n_stimuli,)
        Ordered list of stimuli to include

    Returns
    -------
    tensor : array of shape (n_stimuli, max_reps, n_channels)
    n_reps : array of shape (n_stimuli,)
    """
    n_ch = data.shape[1]
    reps_per_stim = [np.sum(labels == s) for s in target_stimuli]
    max_reps = max(reps_per_stim)

    tensor = np.zeros((len(target_stimuli), max_reps, n_ch), dtype=np.float64)
    n_reps = np.zeros(len(target_stimuli), dtype=np.int32)

    for i, stim in enumerate(target_stimuli):
        trials = data[labels == stim]
        n_reps[i] = len(trials)
        tensor[i, :n_reps[i], :] = trials

    return tensor, n_reps


def combine_tensors(
    t1: np.ndarray, n1: np.ndarray,
    t2: np.ndarray, n2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate trials from two tensors along the repetition axis."""
    n_stim, _, n_ch = t1.shape
    max_reps = n1.max() + n2.max()

    combined = np.zeros((n_stim, max_reps, n_ch), dtype=np.float64)
    for i in range(n_stim):
        combined[i, :n1[i], :] = t1[i, :n1[i], :]
        combined[i, n1[i]:n1[i]+n2[i], :] = t2[i, :n2[i], :]

    return combined, n1 + n2


# =============================================================================
# Reliability computation
# =============================================================================


@njit(parallel=True)
def _split_half_correlations(
    tensor: np.ndarray,
    n_reps: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Compute split-half correlations across random splits (numba-accelerated)."""
    n_stim, _, n_ch = tensor.shape
    correlations = np.zeros((n_splits, n_ch))

    for split in prange(n_splits):
        np.random.seed(seed + split)
        half1 = np.zeros((n_stim, n_ch))
        half2 = np.zeros((n_stim, n_ch))

        # Split trials for each stimulus
        for i in range(n_stim):
            n = n_reps[i]
            idx = np.random.permutation(n)
            if n % 2 != 0:
                n = n - 1

            for ch in range(n_ch):
                sum1, sum2 = 0.0, 0.0
                for j in range(n // 2):
                    sum1 += tensor[i, idx[j], ch]
                    sum2 += tensor[i, idx[n // 2 + j], ch]
                half1[i, ch] = sum1 / (n // 2)
                half2[i, ch] = sum2 / (n // 2)

        # Correlation per channel
        for ch in range(n_ch):
            m1 = np.mean(half1[:, ch])
            m2 = np.mean(half2[:, ch])
            num, d1, d2 = 0.0, 0.0, 0.0
            for i in range(n_stim):
                x = half1[i, ch] - m1
                y = half2[i, ch] - m2
                num += x * y
                d1 += x * x
                d2 += y * y
            correlations[split, ch] = num / (np.sqrt(d1 * d2) + 1e-10)

    return correlations


def compute_reliability(
    tensor: np.ndarray,
    n_reps: np.ndarray,
    n_splits: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Compute split-half reliability with Spearman-Brown correction.

    Parameters
    ----------
    tensor : array of shape (n_stimuli, max_reps, n_channels)
    n_reps : array of shape (n_stimuli,)
    n_splits : int
        Number of random splits
    seed : int
        Random seed for reproducibility

    Returns
    -------
    reliability : array of shape (n_channels,)
        Spearman-Brown corrected split-half reliability
    """
    correlations = _split_half_correlations(
        tensor.astype(np.float64), n_reps, n_splits, seed
    )

    # Fisher z-transform, average, inverse transform
    z = np.arctanh(np.clip(correlations, -0.9999, 0.9999))
    avg_r = np.tanh(z.mean(axis=0))

    # Spearman-Brown correction
    return 2 * avg_r / (1 + avg_r)


# =============================================================================
# Main data loader (uses /SSD/datasets/things/macaque/)
# =============================================================================


def load_macaque(
    dataset: str,
    recording: str,
    root: str | Path | None = None,
    roi: str = "it",
    min_reliab: float | None = 0.3,
    average_exemplars: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load macaque neural data from /SSD/datasets/things/macaque/.

    Parameters
    ----------
    dataset : str
        "2k" (1854 categories, 1 exemplar) or "22k" (1854 categories, 12 exemplars)
    recording : str
        "F" (MonkeyF) or "N" (MonkeyN). For 2k, also "N1" or "N2".
    roi : str
        Brain region: "v1", "v4", or "it"
    min_reliab : float or None
        Filter channels by reliability threshold. None = no filtering.
    average_exemplars : bool
        If True and dataset="22k", average across exemplars per category.

    Returns
    -------
    data : array of shape (n_stimuli, n_channels)
        Neural responses, filtered by reliability if specified.
    stimuli : list[str]
        Stimulus names (category names for 2k, exemplar names for 22k).
    reliab : array of shape (n_channels,)
        Reliability values for returned channels.

    Example
    -------
    >>> data, stimuli, reliab = load_macaque("2k", "F", roi="it", min_reliab=0.3)
    >>> data.shape  # (1854, ~270) - filtered IT channels
    """
    data_root = Path(root) if root is not None else DATA_ROOT
    path = data_root / dataset / "processed" / f"{recording}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    d = np.load(path, allow_pickle=True).item()
    roi_slice = slice(*d["roi_slices"][roi])

    # Get data and reliability for ROI
    data = d["train_MUA"][roi_slice].T  # (n_stimuli, n_channels)
    reliab = d["reliab"]
    if reliab.ndim == 2:
        reliab = reliab.mean(axis=1)  # 22k: average across splits
    reliab = reliab[roi_slice]

    stimuli = list(d["stimuli"])

    # Average across exemplars if requested (22k only)
    if average_exemplars and dataset == "22k":
        categories = sorted(set(stimuli))
        data_avg = np.zeros((len(categories), data.shape[1]), dtype=data.dtype)
        for i, cat in enumerate(categories):
            mask = np.array([s == cat for s in stimuli])
            data_avg[i] = data[mask].mean(axis=0)
        data = data_avg
        stimuli = categories

    # Filter by reliability
    if min_reliab is not None:
        mask = reliab >= min_reliab
        data = data[:, mask]
        reliab = reliab[mask]

    return data, stimuli, reliab
