#!/usr/bin/env python3
"""
Wrapper script to run production tasks with Hydra.
This replaces the internal logic of the old `scripts/experiment`.
"""

import sys
import logging
import importlib
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def _merge_cfg(cfg: DictConfig) -> DictConfig:
    """
    Merges top-level experiment config with the specific task section.
    Preserves the logic from the original experiment runner.
    """
    exp_name = cfg.get("experiment_name")
    task_name = cfg.get("task")
    
    if not exp_name or exp_name == "default":
        raise ValueError("Experiment name not set.")
    if not task_name:
        raise ValueError("Task name not set.")

    # 1. Get Task Config
    if task_name not in cfg:
         available = [k for k, v in cfg.items() if isinstance(v, DictConfig) and k not in ["hydra", "paths"]]
         raise ValueError(f"Task '{task_name}' not found in experiment '{exp_name}'. Available: {available}")
    
    task_cfg = OmegaConf.to_container(cfg[task_name], resolve=True)

    # 2. Get Shared Config (everything except other tasks and system nodes)
    shared_cfg = {}
    ignore_keys = {"hydra", "paths", "experiment_name", "task", task_name}
    
    for key, value in cfg.items():
        if key not in ignore_keys:
            shared_cfg[key] = value

    # Promote common block if present
    if "common" in shared_cfg and isinstance(shared_cfg["common"], DictConfig):
        common_cfg = OmegaConf.to_container(shared_cfg.pop("common"), resolve=True)
        shared_cfg.update(common_cfg)

    # 3. Merge: Shared < Task (Task overrides shared)
    merged = {**shared_cfg, **task_cfg}
    
    # Restore metadata
    merged["experiment_name"] = exp_name
    merged["task"] = task_name
    
    # If datasets are in the root (from common includes), keep them
    if "datasets" in cfg:
        merged["datasets"] = cfg.datasets

    return OmegaConf.create(merged)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Merge Logic
    cfg = _merge_cfg(cfg)

    # 2. Logging
    log = logging.getLogger(__name__)
    log.info(f"Experiment: {cfg.experiment_name}")
    log.info(f"Task:       {cfg.task}")
    log.info(f"Output:     {Path.cwd()}")

    # 3. Import Task Module
    # Structure: experiments.<experiment_name>.tasks.<task_name>
    module_path = f"experiments.{cfg.experiment_name}.tasks.{cfg.task}"
    try:
        task_module = importlib.import_module(module_path)
    except ImportError as e:
        log.error(f"❌ Could not import task: {module_path}")
        log.error(f"Ensure 'experiments/{cfg.experiment_name}/tasks/{cfg.task}.py' exists.")
        raise e

    # 4. Run
    task_module.run(cfg)

if __name__ == "__main__":
    main()
