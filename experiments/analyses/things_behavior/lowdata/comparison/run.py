"""Evaluate SRF and VICE embeddings on the truly held-out THINGS validation set.

Single source of truth for lowdata comparison numbers.
Loads stored embeddings, applies the same accuracy function to all.
VICE: ReLU applied to pruned_q_mu (matches VICE's own evaluation).
SRF: embeddings used as-is (already non-negative from SRF constraint).

Val set: data/things/triplets_47/validationset.txt (457k truly held-out triplets).

Outputs:
    lowdata_comparison.csv  -- model, condition, pct, part, seed, rank, val_acc
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VAL_PATH = PROJECT_ROOT / "data" / "things" / "triplets_47" / "validationset.txt"
SRF_BASE = Path(__file__).resolve().parent.parent / "srf" / "outputs"
VICE_MODEL_DIR = Path(__file__).resolve().parent.parent / "vice" / "outputs" / "models"

SRF_NAME_RE = re.compile(r"srf_(\d+)pct_part(\d+)_seed(\d+)\.npz")
VICE_NAME_RE = re.compile(r"vice_(\d+)pct_part(\d+)_seed(\d+)")


def _triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    """Hard argmax accuracy on dot-product similarities."""
    ei = embedding[triplets[:, 0]]
    ej = embedding[triplets[:, 1]]
    ek = embedding[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def _eval_srf(npz_path: Path, val: np.ndarray, condition: str) -> dict | None:
    match = SRF_NAME_RE.match(npz_path.name)
    if not match:
        return None
    embedding = np.load(npz_path)["embedding"]
    return {
        "model": "SRF",
        "condition": condition,
        "pct": int(match.group(1)),
        "part": int(match.group(2)),
        "seed": int(match.group(3)),
        "rank": embedding.shape[1],
        "val_acc": _triplet_accuracy(embedding, val),
    }


def _eval_vice(model_dir: Path, val: np.ndarray) -> dict | None:
    match = VICE_NAME_RE.match(model_dir.name)
    if not match:
        return None
    param_files = list(model_dir.glob("**/parameters.npz"))
    if not param_files:
        return None
    embedding = np.load(param_files[0], allow_pickle=True).get("pruned_q_mu")
    if embedding is None:
        return None
    embedding = np.maximum(embedding, 0)
    return {
        "model": "VICE",
        "condition": "vice",
        "pct": int(match.group(1)),
        "part": int(match.group(2)),
        "seed": int(match.group(3)),
        "rank": embedding.shape[1],
        "val_acc": _triplet_accuracy(embedding, val),
    }


def _find_srf_embeddings() -> list[tuple[Path, str]]:
    """Find SRF embeddings with their condition name."""
    paths = []
    for emb_dir in sorted(SRF_BASE.glob("*/embeddings")):
        condition = emb_dir.parent.name
        found = sorted(emb_dir.glob("srf_*pct_*.npz"))
        if found:
            log.info("SRF embeddings: %s (%d files)", condition, len(found))
            paths.extend((p, condition) for p in found)
    return paths


def run(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    val = np.loadtxt(VAL_PATH, dtype=float).astype(int)
    log.info("Val: %d triplets from %s", len(val), VAL_PATH.name)

    srf_items = _find_srf_embeddings()
    if not srf_items:
        log.warning("No SRF embeddings found in %s/*/embeddings/", SRF_BASE)

    vice_dirs = sorted(
        d for d in VICE_MODEL_DIR.iterdir()
        if d.is_dir() and VICE_NAME_RE.match(d.name)
    )

    n_srf = len(srf_items)
    log.info("Evaluating %d SRF + %d VICE embeddings...", n_srf, len(vice_dirs))

    all_tasks = (
        [("srf", p, cond, val) for p, cond in srf_items]
        + [("vice", d, "vice", val) for d in vice_dirs]
    )

    def _eval(kind, path, cond, v):
        if kind == "srf":
            return _eval_srf(path, v, cond)
        return _eval_vice(path, v)

    records = Parallel(n_jobs=-1, verbose=10)(
        delayed(_eval)(kind, path, cond, val) for kind, path, cond, val in all_tasks
    )
    records = [r for r in records if r is not None]
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "lowdata_comparison.csv", index=False)

    log.info("\nSummary:")
    for condition in sorted(df["condition"].unique()):
        log.info("  --- %s ---", condition)
        sub_c = df[df["condition"] == condition]
        for pct in sorted(sub_c["pct"].unique()):
            sub = sub_c[sub_c["pct"] == pct]
            log.info(
                "  %s %3d%%: %.2f%% +/- %.2f%% (k=%d, n=%d)",
                sub["model"].iloc[0], pct,
                sub["val_acc"].mean() * 100, sub["val_acc"].std() * 100,
                int(sub["rank"].median()), len(sub),
            )

    log.info("Saved %d rows to %s", len(df), output_dir / "lowdata_comparison.csv")


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
