from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_axes(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load semantic axis definitions from JSON file."""
    return json.loads(Path(path).read_text())


def compute_axis(high_words: np.ndarray, low_words: np.ndarray) -> np.ndarray:
    """Compute normalized axis vector from high/low pole words."""
    axis = np.mean(high_words, axis=0) - np.mean(low_words, axis=0)
    norm = np.linalg.norm(axis)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Zero axis vector")

    axis /= norm

    if np.mean(high_words @ axis) <= np.mean(low_words @ axis):
        axis = -axis

    return axis


def compute_axes(
    axes_config: dict[str, dict[str, list[str]]],
    w: np.ndarray,
    word_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    """Compute axis vectors for all configured axes."""
    axes = {}

    for name, poles in axes_config.items():
        high_idx = [word_to_idx[w] for w in poles["high"] if w in word_to_idx]
        low_idx = [word_to_idx[w] for w in poles["low"] if w in word_to_idx]

        if len(high_idx) < 2 or len(low_idx) < 2:
            continue

        axes[name] = compute_axis(w[high_idx], w[low_idx])

    return axes


def project_words(
    w: np.ndarray, vocabulary: list[str], axes: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Project word embeddings onto semantic axes."""
    if not axes:
        raise ValueError("No axes provided")

    axis_matrix = np.column_stack(list(axes.values()))
    projections = w @ axis_matrix

    return pd.DataFrame({
        "word": [word.lower() for word in vocabulary],
        **{name: projections[:, i] for i, name in enumerate(axes.keys())}
    })
