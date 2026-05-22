"""Low-data THINGS behavioral experiments."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

VICE_MODEL_DIR = Path(__file__).resolve().parent / "outputs" / "models"


def get_vice_dims(
    model_dir: Path = VICE_MODEL_DIR,
    aggregation: str = "median",
) -> dict[int, int]:
    """Read VICE learned dimensions from training logs.

    Parses the final "Dim:" value from each VICE training log and
    aggregates across seeds and partitions per percentage.

    Parameters
    ----------
    model_dir : Path
        Directory containing vice_{pct}pct_part{i}_seed{j}/ subdirectories.
    aggregation : str
        How to aggregate across runs: "median" (default) or "mean".

    Returns
    -------
    dict mapping percentage -> aggregated dimension (int).
    """
    dims_by_pct: dict[int, list[int]] = {}

    for d in model_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"vice_(\d+)pct_part(\d+)_seed(\d+)", d.name)
        if not m:
            continue
        pct = int(m.group(1))

        logs = list(d.rglob("training.log"))
        if not logs:
            continue

        for line in reversed(logs[0].read_text().strip().split("\n")):
            dm = re.search(r"Dim: (\d+)", line)
            if dm:
                dims_by_pct.setdefault(pct, []).append(int(dm.group(1)))
                break

    result = {}
    for pct, vals in sorted(dims_by_pct.items()):
        arr = np.array(vals)
        if aggregation == "median":
            result[pct] = int(np.median(arr))
        else:
            result[pct] = int(np.round(np.mean(arr)))

    return result
