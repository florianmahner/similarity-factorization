#!/usr/bin/env python3
"""Preprocessing utilities for THINGS-2k monkey neural data."""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class NeuralData:
    """Container for neural recording data."""
    data: np.ndarray  # (n_trials, n_channels)
    labels: np.ndarray  # (n_trials,) stimulus labels
    name: str

    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_stimuli(self) -> int:
        return len(np.unique(self.labels))

    def __repr__(self) -> str:
        return f"NeuralData({self.name}: {self.n_trials} trials, {self.n_channels} ch, {self.n_stimuli} stimuli)"


def load_recording(path: Path, roi: str = "it", name: str = "") -> NeuralData:
    """Load a single recording from THINGS-2k format."""
    mat_path = path / "THINGS_normMUA_raw.mat"
    zi_path = path / "zi_list.csv"

    if not mat_path.exists():
        raise FileNotFoundError(f"Data file not found: {mat_path}")
    if not zi_path.exists():
        raise FileNotFoundError(f"Label file not found: {zi_path}")

    with h5py.File(mat_path, "r") as f:
        if f"data_{roi}" not in f:
            raise KeyError(f"ROI '{roi}' not found. Available: {list(f.keys())}")
        data = np.array(f[f"data_{roi}"])

    labels = pd.read_csv(zi_path, header=None)[0].values

    if len(labels) != data.shape[0]:
        raise ValueError(f"Label count ({len(labels)}) != trial count ({data.shape[0]})")

    return NeuralData(data=data, labels=labels, name=name or path.name)


def remove_nan_trials(recording: NeuralData) -> NeuralData:
    """Remove trials containing any NaN values."""
    valid = ~np.isnan(recording.data).any(axis=1)
    n_removed = (~valid).sum()

    if n_removed > 0:
        print(f"  [{recording.name}] Removed {n_removed} NaN trials ({n_removed/len(valid)*100:.1f}%)")

    return NeuralData(
        data=recording.data[valid],
        labels=recording.labels[valid],
        name=recording.name
    )


def filter_to_stimuli(recording: NeuralData, stimuli: np.ndarray) -> NeuralData:
    """Filter recording to only include specified stimuli."""
    mask = np.isin(recording.labels, stimuli)
    n_removed = (~mask).sum()

    if n_removed > 0:
        print(f"  [{recording.name}] Filtered to {len(stimuli)} stimuli, removed {n_removed} trials")

    return NeuralData(
        data=recording.data[mask],
        labels=recording.labels[mask],
        name=recording.name
    )


def get_common_stimuli(*recordings: NeuralData) -> np.ndarray:
    """Get stimuli present in all recordings."""
    if len(recordings) == 0:
        raise ValueError("No recordings provided")

    common = set(np.unique(recordings[0].labels))
    for rec in recordings[1:]:
        common &= set(np.unique(rec.labels))

    return np.array(sorted(common))


def zscore_channels(recording: NeuralData) -> NeuralData:
    """Z-score normalize each channel (zero mean, unit variance)."""
    mean = recording.data.mean(axis=0)
    std = recording.data.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero

    data_z = (recording.data - mean) / std

    return NeuralData(data=data_z, labels=recording.labels, name=recording.name)


def average_per_stimulus(recording: NeuralData, stimuli: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Average trials per stimulus to get (n_stimuli, n_channels)."""
    if stimuli is None:
        stimuli = np.array(sorted(np.unique(recording.labels)))

    n_stimuli = len(stimuli)
    n_channels = recording.n_channels

    averaged = np.zeros((n_stimuli, n_channels))
    n_reps = np.zeros(n_stimuli, dtype=int)

    for i, stim in enumerate(stimuli):
        mask = recording.labels == stim
        n_reps[i] = mask.sum()
        if n_reps[i] > 0:
            averaged[i] = recording.data[mask].mean(axis=0)
        else:
            averaged[i] = np.nan

    return averaged, stimuli, n_reps


def combine_recordings(*recordings: NeuralData, name: str = "combined") -> NeuralData:
    """Combine multiple recordings by stacking trials."""
    if len(recordings) == 0:
        raise ValueError("No recordings provided")

    data = np.vstack([r.data for r in recordings])
    labels = np.concatenate([r.labels for r in recordings])

    return NeuralData(data=data, labels=labels, name=name)


# === SANITY CHECKS ===

def check_zscore(data: np.ndarray, name: str = "", tol: float = 1e-5) -> bool:
    """Verify z-scoring: mean~0, std~1 per channel."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    mean_ok = np.abs(mean).max() < tol
    std_ok = np.abs(std - 1).max() < tol

    if not mean_ok:
        print(f"  WARNING [{name}]: Mean not zero, max|mean|={np.abs(mean).max():.6f}")
    if not std_ok:
        print(f"  WARNING [{name}]: Std not 1, max|std-1|={np.abs(std - 1).max():.6f}")

    return mean_ok and std_ok


def check_stimulus_alignment(X1: np.ndarray, X2: np.ndarray, stimuli: np.ndarray, name1: str, name2: str) -> float:
    """Check correlation between averaged responses from two recordings."""
    from scipy.stats import pearsonr

    # Flatten and correlate
    r_flat = pearsonr(X1.flatten(), X2.flatten())[0]

    # Per-channel correlations
    r_per_ch = [pearsonr(X1[:, ch], X2[:, ch])[0] for ch in range(X1.shape[1])]
    r_median = np.median(r_per_ch)

    print(f"  Correlation [{name1} vs {name2}]: r_flat={r_flat:.3f}, r_median_ch={r_median:.3f}")

    return r_flat


def check_no_nan(data: np.ndarray, name: str = "") -> bool:
    """Check for NaN values."""
    n_nan = np.isnan(data).sum()
    if n_nan > 0:
        print(f"  WARNING [{name}]: {n_nan} NaN values found")
        return False
    return True


def check_positive(data: np.ndarray, name: str = "") -> bool:
    """Check that all values are non-negative."""
    n_neg = (data < 0).sum()
    if n_neg > 0:
        print(f"  WARNING [{name}]: {n_neg} negative values found (min={data.min():.4f})")
        return False
    return True


def summarize_recording(recording: NeuralData):
    """Print summary statistics for a recording."""
    print(f"\n=== {recording.name} ===")
    print(f"  Shape: {recording.data.shape}")
    print(f"  Stimuli: {recording.n_stimuli}")
    print(f"  Trials per stimulus: {recording.n_trials / recording.n_stimuli:.1f} avg")
    print(f"  Data range: [{recording.data.min():.3f}, {recording.data.max():.3f}]")
    print(f"  Mean: {recording.data.mean():.4f}, Std: {recording.data.std():.4f}")
    print(f"  NaN count: {np.isnan(recording.data).sum()}")


def summarize_averaged(X: np.ndarray, name: str = ""):
    """Print summary statistics for averaged data."""
    print(f"\n=== {name} (averaged) ===")
    print(f"  Shape: {X.shape}")
    print(f"  Data range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Mean per channel: [{X.mean(axis=0).min():.4f}, {X.mean(axis=0).max():.4f}]")
    print(f"  Overall mean: {X.mean():.6f}")
    print(f"  NaN count: {np.isnan(X).sum()}")
