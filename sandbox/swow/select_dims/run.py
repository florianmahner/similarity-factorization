"""Score all SRF dimensions of the SWOW consensus embedding by interpretability.

Combines reliability + concentration + variance to find the dimensions whose
top words tell the cleanest semantic story. Prints top-15 dimensions by score
with their top-10 words for visual inspection.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.swow import load_swow_ppmi
from src.utils import get_output_dir

log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CONSENSUS_DIR = PROJECT_ROOT / "experiments/datasets/consensus/outputs/swow"


def _top_words(values: np.ndarray, vocabulary: list[str], k: int = 10) -> list[str]:
    idx = np.argsort(values)[::-1][:k]
    return [vocabulary[i] for i in idx]


def _concentration(values: np.ndarray, k: int = 10) -> float:
    """Fraction of L1 mass in top-k entries (higher = more concentrated)."""
    pos = np.maximum(values, 0)
    total = pos.sum()
    if total <= 0:
        return 0.0
    sorted_vals = np.sort(pos)[::-1]
    return float(sorted_vals[:k].sum() / total)


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy of normalized positive values (lower = more peaked)."""
    pos = np.maximum(values, 0)
    total = pos.sum()
    if total <= 0:
        return 0.0
    p = pos / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("Loading consensus embedding and reliability...")
    embedding = np.load(CONSENSUS_DIR / "embedding.npy")
    reliability = np.load(CONSENSUS_DIR / "reliability.npy")
    cv_reliability = np.load(CONSENSUS_DIR / "cv_reliability.npy")

    log.info(f"Embedding: {embedding.shape}")
    log.info(f"Reliability: {reliability.shape}, mean={reliability.mean():.3f}")
    log.info(f"CV reliability: {cv_reliability.shape}, mean={cv_reliability.mean():.3f}")

    log.info("Loading SWOW vocabulary...")
    _, vocabulary, _ = load_swow_ppmi(
        DATA_DIR / "small-world-of-words", use_all_responses=True,
    )
    n_dims = embedding.shape[1]

    rows = []
    for d in range(n_dims):
        col = embedding[:, d]
        var = float(np.var(col))
        conc10 = _concentration(col, k=10)
        conc20 = _concentration(col, k=20)
        ent = _entropy(col)
        rel = float(reliability[d])
        cv_rel = float(cv_reliability[d])
        max_val = float(col.max())
        n_active = int((col > 1e-6).sum())

        # Composite score: prefer high reliability, high concentration, moderate variance
        score = cv_rel * conc10
        rows.append({
            "dim": d,
            "score": score,
            "cv_reliability": cv_rel,
            "reliability_naive": rel,
            "concentration_top10": conc10,
            "concentration_top20": conc20,
            "entropy": ent,
            "variance": var,
            "max_val": max_val,
            "n_active": n_active,
            "top10_words": ", ".join(_top_words(col, vocabulary, 10)),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "dim_scores.csv", index=False)
    log.info(f"\nSaved per-dim scores to {OUTPUT_DIR / 'dim_scores.csv'}")

    log.info("\n=== Top 25 dimensions by score (cv_reliability × concentration_top10) ===")
    for _, row in df.head(25).iterrows():
        log.info(f"  Dim {int(row.dim):3d}  "
                 f"score={row.score:.3f}  "
                 f"cv_rel={row.cv_reliability:.3f}  "
                 f"conc10={row.concentration_top10:.3f}  "
                 f"var={row.variance:.4f}")
        log.info(f"            {row.top10_words}")

    log.info("\n=== Bottom 5 by score (worst) ===")
    for _, row in df.tail(5).iterrows():
        log.info(f"  Dim {int(row.dim):3d}  "
                 f"score={row.score:.3f}  cv_rel={row.cv_reliability:.3f}  "
                 f"conc10={row.concentration_top10:.3f}")
        log.info(f"            {row.top10_words}")


if __name__ == "__main__":
    main()
