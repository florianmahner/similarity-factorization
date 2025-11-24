from __future__ import annotations

import logging
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf


def _merge_cfg(cfg: DictConfig) -> DictConfig:
    exp_name = cfg.get("experiment_name")
    task_name = cfg.get("task")
    if not exp_name or exp_name == "default":
        raise ValueError("Set experiment with `experiment=<name>`")
    if not task_name:
        raise ValueError("Set task with `<task>=...` or `task=<name>`")

    # Get task-specific config
    task_cfg_raw = cfg.get(task_name)
    if not isinstance(task_cfg_raw, DictConfig):
        available = [k for k, v in cfg.items() if isinstance(v, DictConfig)]
        raise ValueError(
            f"Task '{task_name}' missing from experiment config. "
            f"Define a '{task_name}:' section in configs/hydra/experiment/{exp_name}.yaml. "
            f"Available sections: {available}"
        )
    task_cfg = OmegaConf.to_container(task_cfg_raw, resolve=True)

    # Get all top-level config (shared settings)
    # Exclude: task-specific configs, Hydra configs, experiment metadata
    shared_cfg = {}
    for key, value in cfg.items():
        if key in ["task", "experiment_name", "hydra", "paths", task_name]:
            continue
        # Skip other task configs
        if key in ["link_prediction", "corum_validation", "node_classification"] and key != task_name:
            continue
        if not isinstance(value, DictConfig):
            shared_cfg[key] = value

    # Merge: shared settings + task-specific (task-specific overrides shared)
    merged = {**shared_cfg, **task_cfg}
    merged["experiment_name"] = exp_name
    merged["task"] = task_name

    if "datasets" in cfg:
        merged["datasets"] = OmegaConf.to_container(cfg.datasets, resolve=True)

    return OmegaConf.create(merged)


def _load_task_module(experiment: str, task: str):
    module_path = f"experiments.{experiment}.tasks.{task}"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Task module not found: {module_path}") from exc


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    """
    Main entry point for experiments.
    """
    # Merge config
    cfg = _merge_cfg(cfg)

    # Set up logging format
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        force=True
    )
    log = logging.getLogger(__name__)
    log.info(f"Starting experiment: {cfg.experiment_name}")
    log.info(f"Task: {cfg.task}")
    
    # Get the task module
    try:
        task_module = importlib.import_module(f"experiments.{cfg.experiment_name}.tasks.{cfg.task}")
    except ImportError as e:
        log.error(f"Could not import task module: experiments.{cfg.experiment_name}.tasks.{cfg.task}")
        raise e

    # Run the task
    task_module.run(cfg)


if __name__ == "__main__":
    run()
