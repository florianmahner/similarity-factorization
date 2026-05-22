"""Load rank estimates from dataset rank JSONs.

Single source of truth for downstream callers. Reads two schemas:
  * new (preferred):  payload["estimate"]["rank"]
        from experiments/datasets/dimensionality/outputs/
  * legacy:            payload["k_star_<method>"]
        from experiments/datasets/ranks/{kappa,pct}/outputs/ and data/things/ranks/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Search order: new dimensionality first, legacy locations as fallback.
RANKS_DIRS = [
    PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs",
    PROJECT_ROOT / "data" / "things" / "ranks",
    PROJECT_ROOT / "experiments" / "datasets" / "ranks" / "kappa" / "outputs",
    PROJECT_ROOT / "experiments" / "datasets" / "ranks" / "pct" / "outputs",
]


def _read_rank(data: dict, method: str) -> int | None:
    """Pull a rank out of a JSON payload, preferring the new schema."""
    estimate = data.get("estimate")
    if isinstance(estimate, dict) and "rank" in estimate:
        return int(estimate["rank"])
    legacy_key = f"k_star_{method}"
    if legacy_key in data:
        return int(data[legacy_key])
    return None


def load_kappa_ranks(method: str = "kappa") -> dict[str, int]:
    """Load rank estimates for all datasets.

    Parameters
    ----------
    method : which legacy k_star_<method> to read if the new schema is missing.
        Ignored for files in the new dimensionality format.

    Returns
    -------
    dict mapping dataset name -> rank.
    """
    result: dict[str, int] = {}
    for ranks_dir in RANKS_DIRS:
        if not ranks_dir.exists():
            continue
        for path in sorted(ranks_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            rank = _read_rank(data, method)
            if rank is None:
                continue
            name = data.get("dataset", path.stem)
            # First hit wins (RANKS_DIRS is ordered new-to-old).
            result.setdefault(name, rank)
    return result


def load_things_ranks(method: str = "kappa") -> dict[int, int]:
    """Load THINGS rank estimates keyed by percentage.

    Recognizes the legacy ``things_<pct>pct`` naming and maps the new
    ``things_behavior`` to pct=100.
    """
    result: dict[int, int] = {}
    for ranks_dir in RANKS_DIRS:
        if not ranks_dir.exists():
            continue
        for path in sorted(ranks_dir.glob("things_*.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            rank = _read_rank(data, method)
            if rank is None:
                continue
            name = data.get("dataset", path.stem)
            pct: int | None = None
            m = re.match(r"things_(\d+)pct", name)
            if m:
                pct = int(m.group(1))
            elif name == "things_behavior":
                pct = 100
            if pct is not None:
                result.setdefault(pct, rank)
    return result


def load_things_rank(pct: int = 100, method: str = "kappa") -> int:
    """Rank estimate for a single THINGS percentage (5/10/20/50/100)."""
    ranks = load_things_ranks(method=method)
    if pct not in ranks:
        raise KeyError(f"No rank estimate for {pct}% (available: {sorted(ranks.keys())})")
    return ranks[pct]
