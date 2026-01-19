#!/usr/bin/env python3
"""
Task runner with automatic config-based sweeps.

Sweeps are triggered by list values for SWEEP_PARAMS.
Other list values (like 'methods') are passed through as-is.

Usage:
    ./scripts/submit experiments/ppi/link_prediction.py
    ./scripts/submit experiments/ppi/link_prediction.py dataset=huri
"""

import json
import logging
import importlib
import os
import signal
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import hydra
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_output_capture

log = logging.getLogger(__name__)

# Global reference to status directory for signal handlers
_status_dir: Path | None = None

# Parameters that trigger parallel sweeps when given as lists
SWEEP_PARAMS = {"dataset", "seed", "subject_id", "percentage"}


def init_job_status(name: str, script: str, args: list[str]) -> Path:
    """Initialize .status/ directory and job.json for tracking.

    The log file is always .status/log relative to the output directory.
    This is where Hydra's file handler and stdout/stderr capture both write.
    """
    status_dir = Path.cwd() / ".status"
    status_dir.mkdir(exist_ok=True)

    job_data = {
        "name": name,
        "script": script,
        "args": args,
        "started": datetime.now().isoformat(),
        "ended": None,
        "status": "running",
        "pid": os.getpid(),
        "output_dir": str(Path.cwd()),
        "log_file": str(status_dir / "log"),
        "error": None,
        "signal": None,
    }

    job_file = status_dir / "job.json"
    job_file.write_text(json.dumps(job_data, indent=2))

    return status_dir


def update_job_status(
    status: str,
    error: str | None = None,
    sig: str | None = None,
) -> None:
    """Update job.json with new status."""
    if _status_dir is None:
        return

    job_file = _status_dir / "job.json"
    if not job_file.exists():
        return

    try:
        job_data = json.loads(job_file.read_text())
        job_data["status"] = status
        job_data["ended"] = datetime.now().isoformat()
        if error:
            job_data["error"] = error
        if sig:
            job_data["signal"] = sig
        job_file.write_text(json.dumps(job_data, indent=2))
    except Exception:
        pass  # Best effort - don't crash on status update failure


def handle_signal(signum: int, frame) -> None:
    """Handle termination signals by updating status before exit."""
    sig_name = signal.Signals(signum).name
    log.warning(f"Received {sig_name}, shutting down...")
    update_job_status("aborted", sig=sig_name)
    sys.exit(128 + signum)


def expand_sweep(cfg: DictConfig) -> list[DictConfig]:
    """
    Expand config with list-valued sweep params into a grid of configs.

    Only params in SWEEP_PARAMS are expanded. Other lists pass through.

    Example:
        cfg.dataset = ['huri', 'string']
        cfg.seed = [0, 1]
        cfg.methods = ['srf', 'cn']  # Not in SWEEP_PARAMS, passes through

        Returns 4 configs: (huri,0), (huri,1), (string,0), (string,1)
        Each has methods=['srf', 'cn'] unchanged.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Find sweep params with list values
    sweep_lists = {}
    for param in SWEEP_PARAMS:
        if param in cfg_dict and isinstance(cfg_dict[param], list):
            sweep_lists[param] = cfg_dict[param]

    if not sweep_lists:
        return [cfg]

    # Generate all combinations
    keys = list(sweep_lists.keys())
    values = [sweep_lists[k] for k in keys]

    configs = []
    for combo in product(*values):
        new_cfg = cfg_dict.copy()
        for k, v in zip(keys, combo):
            new_cfg[k] = v
        configs.append(OmegaConf.create(new_cfg))

    return configs


def run_single(task_module, cfg: DictConfig, idx: int, total: int) -> None:
    """Run a single task configuration."""
    sweep_info = {p: cfg.get(p) for p in SWEEP_PARAMS if p in cfg}
    log.info(f"[{idx}/{total}] Running with {sweep_info}")
    task_module.run(cfg)


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    global _status_dir

    # Set up stdout/stderr capture to .status/log FIRST
    # This captures all output including progress bars and prints
    setup_output_capture()

    exp_name = cfg.get("experiment_name")
    task_name = cfg.get("task")

    if not exp_name:
        raise ValueError("experiment_name not set in config")
    if not task_name:
        raise ValueError("task not set in config")

    # Initialize job tracking
    job_name = f"{exp_name}_{task_name}"
    script = cfg.get("_module_path", f"experiments/{exp_name}/{task_name}.py")
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    _status_dir = init_job_status(job_name, script, args)

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        _run_task(cfg, exp_name, task_name)
        update_job_status("completed")
    except Exception as e:
        update_job_status("failed", error=str(e))
        raise


def _run_task(cfg: DictConfig, exp_name: str, task_name: str) -> None:
    """Execute the task (separated for cleaner error handling)."""
    # Import task module
    module_path = cfg.get("_module_path")
    if module_path:
        module_paths = [module_path]
    else:
        module_paths = [
            f"experiments.{exp_name}.{task_name}",
            f"experiments.{exp_name}",
        ]

    task_module = None
    for mp in module_paths:
        try:
            candidate = importlib.import_module(mp)
            if hasattr(candidate, "run"):
                task_module = candidate
                break
        except ImportError:
            continue
    if task_module is None:
        log.error(f"Could not import task module with run(). Tried: {module_paths}")
        raise ImportError(f"No module found for experiment={exp_name}, task={task_name}")

    # Expand sweeps
    configs = expand_sweep(cfg)
    n_configs = len(configs)

    if n_configs == 1:
        log.info(f"Experiment: {exp_name}")
        log.info(f"Task: {task_name}")
        log.info(f"Output: {Path.cwd()}")
        task_module.run(configs[0])
    else:
        n_jobs = cfg.get("sweep_jobs", min(n_configs, cfg.get("n_jobs", -1)))
        sweep_params = [p for p in SWEEP_PARAMS if p in OmegaConf.to_container(cfg)
                       and isinstance(OmegaConf.to_container(cfg).get(p), list)]

        log.info(f"Experiment: {exp_name}")
        log.info(f"Task: {task_name}")
        log.info(f"Sweep: {n_configs} combinations over {sweep_params}")
        log.info(f"Parallel jobs: {n_jobs}")
        log.info(f"Output: {Path.cwd()}")

        Parallel(n_jobs=n_jobs)(
            delayed(run_single)(task_module, c, i, n_configs)
            for i, c in enumerate(configs, 1)
        )

        log.info(f"Completed {n_configs} sweep runs")


if __name__ == "__main__":
    main()
