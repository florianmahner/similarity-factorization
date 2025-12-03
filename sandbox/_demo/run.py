from __future__ import annotations
import importlib
import hydra
from omegaconf import DictConfig
import os
import sys

# This run.py is specific to the development/demo project
# It mimics the main run.py but loads tasks locally

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # For this demo, we just delegate everything to the single runner task
    # which handles different 'types'
    from tasks import runner
    try:
        runner.run(cfg)
    except Exception as e:
        # We suppress the traceback for the demo if it's the expected failure
        if cfg.get("type") == "failure":
            print(f"Job failed as expected (check dashboard for details): {e}")
            sys.exit(1) # Exit with error code but no traceback
        else:
            raise e

if __name__ == "__main__":
    main()
