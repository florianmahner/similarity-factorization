"""Evaluate SRF (kappa ranks) and VICE on validationset.txt (truly held-out)."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
VAL_PATH = PROJECT_ROOT / "data" / "things" / "triplets_47" / "validationset.txt"
SRF_DIR = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "srf" / "outputs" / "kappa_alpha0" / "embeddings"
VICE_DIR = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "vice" / "outputs" / "models"


def triplet_acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def _eval_srf(path, val):
    d = np.load(path)
    w = d["embedding"]
    return triplet_acc(w, val)


def _eval_vice(model_dir, val):
    params = sorted(model_dir.rglob("parameters.npz"))
    if not params:
        return np.nan, 0
    d = np.load(params[0], allow_pickle=True)
    w = np.maximum(d["embedding"], 0)  # embedding + relu, matches VICE's own eval
    return triplet_acc(w, val), w.shape[1]


def main():
    log.info("Loading validationset.txt...")
    val = np.loadtxt(VAL_PATH).astype(int)
    log.info(f"Validation triplets: {len(val):,}")

    records = []

    # SRF kappa
    srf_files = sorted(SRF_DIR.glob("srf_*.npz"))
    log.info(f"\nEvaluating {len(srf_files)} SRF embeddings...")
    srf_accs = Parallel(n_jobs=-1, verbose=10)(
        delayed(_eval_srf)(f, val) for f in srf_files
    )
    for f, acc in zip(srf_files, srf_accs):
        name = f.stem
        parts = name.split("_")
        pct = int(parts[1].replace("pct", ""))
        part = int(parts[2].replace("part", ""))
        seed = int(parts[3].replace("seed", ""))
        rank = np.load(f)["embedding"].shape[1]
        records.append({"model": "SRF", "condition": "kappa_alpha0", "pct": pct,
                        "part": part, "seed": seed, "rank": rank, "val_acc": acc})

    # VICE
    vice_dirs = sorted(VICE_DIR.glob("vice_*"))
    log.info(f"\nEvaluating {len(vice_dirs)} VICE models...")
    vice_accs = Parallel(n_jobs=-1, verbose=10)(
        delayed(_eval_vice)(d, val) for d in vice_dirs
    )
    for d, (acc, rank) in zip(vice_dirs, vice_accs):
        name = d.name
        parts = name.split("_")
        pct = int(parts[1].replace("pct", ""))
        part = int(parts[2].replace("part", ""))
        seed = int(parts[3].replace("seed", ""))
        records.append({"model": "VICE", "condition": "vice", "pct": pct,
                        "part": part, "seed": seed, "rank": rank, "val_acc": acc})

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "validationset_comparison.csv", index=False)
    log.info(f"\nSaved: {OUTPUT_DIR / 'validationset_comparison.csv'}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS (validationset.txt)")
    log.info("=" * 60)
    for model in ["SRF", "VICE"]:
        for pct in [5, 10, 20, 50, 100]:
            sub = df[(df["model"] == model) & (df["pct"] == pct)]
            if len(sub) == 0:
                continue
            log.info(f"  {model:4s} {pct:>3d}%: {sub.val_acc.mean():.4f} +/- {sub.val_acc.std():.4f} "
                     f"(k={int(sub['rank'].median())}, n={len(sub)})")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
