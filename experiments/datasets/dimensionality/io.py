"""Read dimensionality outputs.

Per-dataset layout:
    outputs/<dataset>/coherence_estimate.json
    outputs/<dataset>/cross_validation.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
COHERENCE_FILE = "coherence_estimate.json"
CROSS_VALIDATION_FILE = "cross_validation.json"
LEGACY_RESULT_FILES = ("rank_estimation.json", "result.json")


def dataset_dir(dataset: str, output_dir: Path = OUTPUT_DIR) -> Path:
    return output_dir / dataset


def coherence_path(dataset: str, output_dir: Path = OUTPUT_DIR) -> Path:
    return dataset_dir(dataset, output_dir) / COHERENCE_FILE


def cross_validation_path(dataset: str, output_dir: Path = OUTPUT_DIR) -> Path:
    return dataset_dir(dataset, output_dir) / CROSS_VALIDATION_FILE


def legacy_result_paths(dataset: str, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    ds_dir = dataset_dir(dataset, output_dir)
    return [ds_dir / name for name in LEGACY_RESULT_FILES]


def result_path(dataset: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Backward-compatible alias for the coherence estimate path."""
    return coherence_path(dataset, output_dir)


def summary_path(stage: str, output_dir: Path, only=None) -> Path:
    if only is None:
        return output_dir / f"{stage}_summary.json"
    if isinstance(only, str):
        names = [part.strip() for part in only.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in only if str(part).strip()]
    slug = "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", name) for name in sorted(names))
    return output_dir / f"{stage}_summary_{slug}.json"


def cv_csv_path(
    dataset: str,
    output_dir: Path = OUTPUT_DIR,
    variant: str | None = None,
) -> Path:
    if variant is None:
        return dataset_dir(dataset, output_dir) / "cv.csv"
    return dataset_dir(dataset, output_dir) / f"cv_{variant}.csv"


def cv_csv_paths(dataset: str, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    ds_dir = dataset_dir(dataset, output_dir)
    return [ds_dir / "cv.csv", *sorted(ds_dir.glob("cv_*.csv"))]


def plot_path(dataset: str, output_dir: Path = OUTPUT_DIR) -> Path:
    return dataset_dir(dataset, output_dir) / "plot.pdf"


def _read(dataset: str, output_dir: Path) -> dict:
    path = coherence_path(dataset, output_dir)
    if not path.exists():
        for legacy_path in legacy_result_paths(dataset, output_dir):
            if legacy_path.exists():
                path = legacy_path
                break
    if not path.exists():
        raise FileNotFoundError(
            f"No dimensionality result for {dataset!r}: {path}. "
            f"Run: python -m experiments.datasets.dimensionality.run "
            f"mode=estimate only=[{dataset}]"
        )
    return json.loads(path.read_text())


def read_coherence(dataset: str, output_dir: Path = OUTPUT_DIR) -> dict:
    return _read(dataset, output_dir)


def read_cross_validation(dataset: str, output_dir: Path = OUTPUT_DIR) -> dict:
    path = cross_validation_path(dataset, output_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_estimate(dataset: str, output_dir: Path = OUTPUT_DIR) -> dict:
    payload = _read(dataset, output_dir)
    if "estimate" not in payload:
        raise KeyError(f"'estimate' block missing for {dataset!r}")
    return payload["estimate"]


def load_rank(dataset: str, output_dir: Path = OUTPUT_DIR) -> int:
    return int(load_estimate(dataset, output_dir)["rank"])


def load_sampling_fraction(dataset: str, output_dir: Path = OUTPUT_DIR) -> float:
    return float(load_estimate(dataset, output_dir)["sampling_fraction"])


def _all_results(output_dir: Path):
    """Iterate over (dataset_name, payload) for every coherence JSON in output_dir."""
    if not output_dir.exists():
        return
    for sub in sorted(output_dir.iterdir()):
        if not sub.is_dir():
            continue
        path = coherence_path(sub.name, output_dir)
        if not path.exists():
            legacy = [p for p in legacy_result_paths(sub.name, output_dir) if p.exists()]
            path = legacy[0] if legacy else path
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        yield data.get("dataset", sub.name), data


def load_ranks(output_dir: Path = OUTPUT_DIR) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, data in _all_results(output_dir):
        est = data.get("estimate")
        if isinstance(est, dict) and "rank" in est:
            result[name] = int(est["rank"])
    return result


def load_sampling_fractions(output_dir: Path = OUTPUT_DIR) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, data in _all_results(output_dir):
        est = data.get("estimate")
        if isinstance(est, dict) and "sampling_fraction" in est:
            result[name] = float(est["sampling_fraction"])
    return result
