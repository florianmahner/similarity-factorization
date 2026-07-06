"""Add the repository ``configs/`` directory to Hydra's search path.

Task configs live next to each ``run.py`` (``config_path="."``) but inherit
shared defaults (``/base``, ``/dataset``, ...) from the top-level ``configs/``
directory. Hydra auto-discovers any ``hydra_plugins`` namespace package on the
path, so registering this search path here makes every task runnable with a
plain ``poetry run python experiments/<group>/<task>/run.py`` command, with no
``--config-dir`` flag.
"""

from __future__ import annotations

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class SrfConfigSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="srf-configs", path=f"file://{_CONFIGS}")
