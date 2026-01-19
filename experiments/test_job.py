"""Test job for dashboard testing.

Usage:
    ./scripts/submit experiments/test_job.py --bg
    ./scripts/submit experiments/test_job.py total=10 interval=3 --bg
"""

import logging
import time
import random
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    total = cfg.get("total", 20)
    interval = cfg.get("interval", 5)
    name = cfg.get("name", "test")

    log.info(f"Starting {name} job: {total} steps, {interval}s interval")
    log.info(f"Output directory: {Path.cwd()}")

    for i in range(1, total + 1):
        value = random.random()
        log.info(f"[{i}/{total}] Processing step {i}... value={value:.4f}")
        time.sleep(interval)

    log.info(f"Completed {name} job!")
