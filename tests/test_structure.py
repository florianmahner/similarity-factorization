"""Structural verification tests for repository reorganization."""
import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_tools_imports():
    """Verify tools/ modules are importable."""
    modules = ["tools.rsa", "tools.metrics", "tools.stats"]
    for mod in modules:
        importlib.import_module(mod)


def test_utils_imports():
    """Verify utils/ modules are importable."""
    modules = ["utils.helpers", "utils.graphs", "utils.io", "utils.plotting"]
    for mod in modules:
        importlib.import_module(mod)


def test_datasets_imports():
    """Verify datasets/ modules are importable."""
    modules = ["datasets.loaders", "datasets.base"]
    for mod in modules:
        importlib.import_module(mod)


def test_similarity_imports():
    """Verify similarity/ modules are importable."""
    modules = ["similarity.builder"]
    for mod in modules:
        importlib.import_module(mod)


def test_pysrf_import():
    """Verify pysrf is importable."""
    import pysrf
    from pysrf import SRF

    assert hasattr(SRF, "fit")


def test_hydra_configs_valid():
    """Verify base Hydra config loads."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    assert "seed" in cfg
    assert "project_root" in cfg
