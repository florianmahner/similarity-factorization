from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class MergeResultsCallback(Callback):
    """
    Callback to merge all 'results.json' files from a multirun sweep into a single CSV.
    Assumes jobs are stored in a .tmp/ directory and cleans it up afterwards.
    """

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Run after all jobs in the sweep have completed.
        """
        sweep_dir = Path(config.hydra.sweep.dir)
        log.info(f"Merging results from {sweep_dir}...")

        # Find all individual result.json files recursively
        # We look for ANY results.json, assuming they are in the .tmp subfolders
        all_files = list(sweep_dir.glob("**/results.json"))

        if not all_files:
            log.warning(f"No results.json files found in {sweep_dir} to merge.")
            # Still try to cleanup .tmp if it exists
            self._cleanup(sweep_dir)
            return

        # Read and Concatenate
        dfs = []
        for f in all_files:
            try:
                df = pd.read_json(f, orient="records")
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                log.warning(f"Failed to read {f}: {e}")

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            output_path = sweep_dir / "results.csv"
            merged_df.to_csv(output_path, index=False)
            log.info(f"Successfully merged {len(dfs)} runs into {output_path}")
        else:
            log.warning("No valid results to merge.")

        # Cleanup the temporary directory
        self._cleanup(sweep_dir)

    def _cleanup(self, sweep_dir: Path) -> None:
        tmp_dir = sweep_dir / ".tmp"
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
                log.info(f"Cleaned up temporary directory: {tmp_dir}")
            except Exception as e:
                log.warning(f"Failed to delete temporary directory {tmp_dir}: {e}")
