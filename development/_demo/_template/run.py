"""Development experiment template.

Usage:
    # Single run
    poetry run python run.py

    # Override parameters
    poetry run python run.py seed=123 param1=value

    # Parallel sweep (joblib)
    poetry run python run.py --multirun seed=0,1,2,3,4

    # SLURM sweep
    poetry run python run.py --multirun hydra/launcher=submitit_slurm seed=0,1,2,3,4
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Outputs automatically go to outputs/<YYMMDD>/<HHMMSS>/
    output_dir = Path.cwd()

    # Your experiment code here
    print(f"Running with seed={cfg.seed}")
    print(f"Output directory: {output_dir}")

    # Example: save results
    # import pandas as pd
    # results = pd.DataFrame(...)
    # results.to_csv(output_dir / "results.csv", index=False)


if __name__ == "__main__":
    main()
