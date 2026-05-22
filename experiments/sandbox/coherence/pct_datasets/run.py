"""Run PCT on dinov3, swow, nsd_subj01, mur92, peterson_animals, peterson_various."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from similarity import build_similarity
from src.coherence import permutation_coherence_test
from src.utils import get_output_dir

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path("/LOCAL/fmahner/similarity-factorization")


def load_dataset_cfg(config_name):
    raw = OmegaConf.load(PROJECT_ROOT / "configs" / "dataset" / f"{config_name}.yaml")
    parent = OmegaConf.create({
        "paths": {"data_dir": str(PROJECT_ROOT / "data")},
        "dataset": raw,
    })
    OmegaConf.resolve(parent)
    return parent.dataset


DATASETS = [
    ("mur92", "mur92", None, 80),
    ("peterson_animals", "peterson_animals", None, 100),
    ("peterson_various", "peterson_various", None, 100),
    ("dinov3", "dinov3", None, 120),
    ("swow", "swow", None, 200),
    ("nsd_subj01", "nsd", 1, 200),
]


def main():
    for name, config, subject_id, k_max in DATASETS:
        log.info(f"\n{'='*50}")
        log.info(f"Dataset: {name}")
        log.info(f"{'='*50}")

        t0 = time.time()
        cfg = load_dataset_cfg(config)
        s = build_similarity(cfg, subject_id=subject_id)
        log.info(f"RSM: {s.shape}, load time: {time.time()-t0:.1f}s")

        t1 = time.time()
        result = permutation_coherence_test(
            s, k_max=k_max, p_list=np.linspace(0.3, 0.95, 10),
            B=15, J=50, alpha=0.05, random_state=42, show_progress=True,
        )
        elapsed = time.time() - t1

        pv = result["pvalues"]
        n_sig = int(np.sum(pv < 0.05))
        log.info(f"PCT: k*={result['k_star']}, n_sig={n_sig}, time={elapsed:.0f}s")

        out = {
            "dataset": name, "n": int(s.shape[0]),
            "k_star_pct": result["k_star"], "n_significant": n_sig,
            "pvalues": pv.tolist(), "runtime_sec": round(elapsed, 1),
        }
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"Saved {name}.json")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("PCT SUMMARY")
    log.info(f"{'='*60}")
    for name, _, _, _ in DATASETS:
        p = OUTPUT_DIR / f"{name}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            log.info(f"  {d['dataset']:>20s}: k*={d['k_star_pct']:>3d}")


if __name__ == "__main__":
    main()
