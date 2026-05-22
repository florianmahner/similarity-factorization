"""Single SRF fit on monkey IT data at kappa rank k*=40 for visual inspection."""

import logging

import numpy as np
from pysrf import SRF

from similarity import build_similarity
from datasets import load_dataset
from src.utils import get_output_dir

log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

RANK = 40


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("Loading monkey IT similarity matrix...")
    ds = load_dataset("things-monkey-22k", root="data/things/macaque",
                      monkey_type="F", roi="it", min_reliab=0.3)
    features = ds.data
    log.info(f"Features shape: {features.shape}")

    from src.tools.metrics import gaussian_kernel_similarity
    s = gaussian_kernel_similarity(features, features)
    log.info(f"Similarity matrix: {s.shape}, mean={s.mean():.4f}")

    log.info(f"Fitting SRF at rank={RANK}...")
    model = SRF(rank=RANK, max_outer=5, max_inner=50, random_state=42)
    model.fit(s)
    w = model.w_
    log.info(f"Embedding shape: {w.shape}")
    log.info(f"Sparsity: {(w == 0).mean():.3f}")

    np.save(OUTPUT_DIR / "embedding.npy", w)
    np.save(OUTPUT_DIR / "similarity.npy", s)
    log.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
