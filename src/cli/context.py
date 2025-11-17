from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump


@dataclass(slots=True)
class ExperimentContext:
    experiment: str
    mode: str  # "dev" or "stable"
    run_dir: Path
    logger: logging.Logger
    rng: np.random.Generator

    def task_dir(self, task: str, *tags: str) -> Path:
        d = self.run_dir / task
        for t in tags:
            if t:
                d = d / t
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_csv(self, df: pd.DataFrame, name: str, subdir: Path | None = None) -> Path:
        base = subdir if subdir is not None else self.run_dir
        p = base / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def save_npy(self, arr: np.ndarray, name: str, subdir: Path | None = None) -> Path:
        base = subdir if subdir is not None else self.run_dir
        p = base / f"{name}.npy"
        np.save(p, arr)
        return p

    def save_joblib(self, obj: object, name: str, subdir: Path | None = None) -> Path:
        base = subdir if subdir is not None else self.run_dir
        p = base / f"{name}.joblib"
        dump(obj, p)
        return p

    def save_plot(self, fig: Any, name: str, subdir: Path | None = None) -> Path:
        base = subdir if subdir is not None else self.run_dir
        p = base / f"{name}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        return p

    def track_metric(
        self, key: str, value: float | int | str, step: int | None = None
    ) -> None:
        row = {"step": step, "key": key, "value": value}
        path = self.run_dir / "metrics.csv"
        header = not path.exists()
        pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=header)


def make_run_dir(
    experiment: str,
    mode: str,
    repo_root: Path,
    analysis_name: str | None = None,
    task_name: str | None = None,
) -> Path:
    if mode == "dev":
        date = time.strftime("%y%m%d")
        ts = time.strftime("%H%M%S")
        return (
            repo_root
            / "experiments"
            / "development"
            / experiment
            / "outputs"
            / date
            / ts
        )

    # Stable mode: use --name if provided, otherwise use task name
    output_name = analysis_name or task_name
    if not output_name:
        raise ValueError(
            "--name or --task required when --mode stable "
            "(or specify task in TOML config)"
        )
    return repo_root / "experiments" / experiment / "outputs" / output_name


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(run_dir.as_posix())
    logger.setLevel(logging.INFO)
    logger.handlers = []
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in [logging.FileHandler(run_dir / "run.log"), logging.StreamHandler()]:
        h.setFormatter(f)
        logger.addHandler(h)
    return logger


def write_metadata(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
