#!/usr/bin/env python3
"""Validate CORUM complex recovery from SRF embedding."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analyses.ppi import (
    load_corum,
    validate_embedding_against_corum,
    map_string_ids_to_genes,
)


def main():
    root_dir = Path(__file__).parent.parent.parent
    output_dir = root_dir / "experiments/ppi/outputs/corum_validation"

    embedding = np.load(output_dir / "embedding.npy")
    proteins = np.loadtxt(output_dir / "proteins.txt", dtype=str).tolist()

    mapped_proteins, n_mapped = map_string_ids_to_genes(
        proteins, root_dir / "data/ppi/protein_info_900.csv"
    )

    corum_complexes = load_corum(root_dir / "data/ppi/corum_complexes.txt")
    results_df = validate_embedding_against_corum(
        embedding, mapped_proteins, corum_complexes, top_n=10
    )

    results_df.to_csv(output_dir / "validation_results.csv", index=False)

    corum_proteins = set().union(*corum_complexes.values())
    overlap = set(mapped_proteins) & corum_proteins

    print(f"Embedding: {embedding.shape[0]} proteins x {embedding.shape[1]} dimensions")
    print(f"Mapped: {n_mapped} ENSP IDs to gene names")
    print(
        f"CORUM coverage: {len(overlap)}/{len(corum_proteins)} ({len(overlap)/len(corum_proteins)*100:.1f}%)"
    )
    print(f"\nValidation results (top 5 by F1):")
    print(
        results_df.nlargest(5, "f1")[
            ["dimension", "best_complex", "f1", "precision", "recall"]
        ]
    )
    print(f"\nAverage F1: {results_df['f1'].mean():.3f}")
    print(f"Saved: {output_dir / 'validation_results.csv'}")


if __name__ == "__main__":
    main()
