"""Save NSD similarity matrix for quick loading."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets import load_dataset
from src.tools.rsa import compute_similarity


def main():
    print("Loading NSD data for subject 1...")
    ds = load_dataset("nsd", root="/LOCAL/LABSHARE/natural-scenes-dataset", subject_id=1, roi_name="nsdgeneral", space="func1pt8mm")
    print(f"Data shape: {ds.data.shape}")

    print("Computing similarity matrix...")
    S = compute_similarity(ds.data, ds.data, "gaussian_kernel")
    print(f"RSM shape: {S.shape}")

    out_path = Path("/tmp/nsd_subj1_rsm.npy")
    np.save(out_path, S)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
