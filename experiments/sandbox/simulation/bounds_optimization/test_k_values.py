"""Test pmax with k=1 vs k=eff_dim on mur92."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets import load_dataset
from pysrf.bounds import p_upper_only_k_fast

MUR92_PATH = "/SSD/datasets/similarity_datasets/mur92"


def main():
    print("Loading mur92 dataset...")
    ds = load_dataset("mur92", root=MUR92_PATH)
    S = ds.rsm
    print(f"Shape: {S.shape}")

    eff_dim = np.ceil((np.linalg.norm(S, "fro") / np.linalg.norm(S, 2)) ** 2).astype(int)
    print(f"Effective dimension: {eff_dim}")

    print("\n--- k=1 (current code) ---")
    pmax_k1 = p_upper_only_k_fast(S, k=1, verbose=True, seed=0)
    print(f"pmax = {pmax_k1:.6f}")

    print(f"\n--- k={eff_dim} (old code, k=eff_dim) ---")
    pmax_keff = p_upper_only_k_fast(S, k=eff_dim, verbose=True, seed=0)
    print(f"pmax = {pmax_keff:.6f}")

    print("\n--- Summary ---")
    print(f"k=1:       pmax = {pmax_k1:.6f}")
    print(f"k={eff_dim} (old): pmax = {pmax_keff:.6f}")
    print(f"Stored:    pmax = 0.909812")


if __name__ == "__main__":
    main()
