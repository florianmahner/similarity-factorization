from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pysrf import SRF


@dataclass(frozen=True, slots=True)
class SimplePriorConfig:
    weight_mode: str
    prior_strength: float
    rho: float
    init: str


def load_triplets_file(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(np.int32)


def triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    idx = triplets.astype(np.int32, copy=False)
    ei = embedding[idx[:, 0]]
    ej = embedding[idx[:, 1]]
    ek = embedding[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_srf_embedding(
    similarity: np.ndarray,
    rank: int,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> np.ndarray:
    model = SRF(
        rank=rank,
        rho=rho,
        init=init,
        random_state=seed,
        max_outer=max_outer,
        max_inner=max_inner,
        tol=tol,
        verbose=0,
    )
    return model.fit_transform(similarity)


def fit_and_score_srf(
    similarity: np.ndarray,
    triplets: np.ndarray,
    rank: int,
    config: SimplePriorConfig,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> dict[str, float | int | str]:
    embedding = fit_srf_embedding(
        similarity=similarity,
        rank=rank,
        rho=config.rho,
        init=config.init,
        seed=seed,
        max_outer=max_outer,
        max_inner=max_inner,
        tol=tol,
    )
    return {
        "weight_mode": config.weight_mode,
        "prior_strength": config.prior_strength,
        "rho": config.rho,
        "init": config.init,
        "seed": seed,
        "val_acc": triplet_accuracy(embedding, triplets),
    }


def _find_project_root(start: Path) -> Path:
    """Walk up from ``start`` until a ``pyproject.toml`` is found."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate pyproject.toml walking up from {start}"
    )


def load_optimal_rank(
    dataset_name: str,
    kind: str = "argmin",
    project_root: Path | None = None,
) -> int:
    """Load optimal rank from a dimensionality CV JSON.

    Reads ``<project_root>/experiments/datasets/dimensionality/outputs/<dataset_name>/cross_validation.json``
    and returns ``payload["validations"][payload["primary_variant"]][f"{kind}_rank"]``.

    Parameters
    ----------
    dataset_name : str
        Dataset name (folder under ``dimensionality/outputs/``).
    kind : {"argmin", "one_se"}
        Which rank to load from the primary CV variant.
    project_root : Path or None
        Project root. Defaults to walking up from this file until a
        ``pyproject.toml`` is found.
    """
    if project_root is None:
        project_root = _find_project_root(Path(__file__).resolve().parent)
    path = (
        project_root
        / "experiments"
        / "datasets"
        / "dimensionality"
        / "outputs"
        / dataset_name
        / "cross_validation.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Dimensionality CV results not found: {path}\n"
            f"Run first: poetry run python experiments/datasets/dimensionality/run.py "
            f"dataset={dataset_name}"
        )
    payload = json.loads(path.read_text())
    dataset = payload.get("dataset")
    if dataset is not None and dataset != dataset_name:
        raise ValueError(
            f"Expected dataset {dataset_name!r}, found {dataset!r} in {path}"
        )
    primary = payload["primary_variant"]
    return int(payload["validations"][primary][f"{kind}_rank"])
