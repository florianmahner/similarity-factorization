from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pysrf import SRF

from ..lib.corum import (
    load_corum,
    validate_embedding_against_corum,
    map_string_ids_to_genes,
)


def run(cfg: DictConfig) -> None:
    if not cfg.string_data:
        raise ValueError("string_data required for corum_validation task")

    string_data = Path(cfg.string_data)

    if string_data.suffix == ".csv":
        from ..lib.utils import load_network, build_adjacency_with_nan

        g, _ = load_network(string_data)
        nodes = sorted(g.nodes())
        adj = build_adjacency_with_nan(g, g, nodes, fill_missing_with_nan=False)
        proteins = nodes
    elif string_data.suffix == ".npy":
        adj = np.load(string_data)
        proteins_file = string_data.parent / "proteins.txt"
        proteins = (
            np.loadtxt(proteins_file, dtype=str).tolist()
            if proteins_file.exists()
            else [f"protein_{i}" for i in range(adj.shape[0])]
        )
    else:
        raise ValueError(f"Unknown file format: {string_data.suffix}")

    model = SRF(
        rank=cfg.rank,
        rho=3.0,
        max_outer=2000,
        max_inner=50,
        tol=1e-4,
        verbose=0,
        init="random_sqrt",
        random_state=cfg.seed,
        missing_values=np.nan,
        loss="frobenius",
    )
    embedding = model.fit_transform(adj)

    np.save(Path.cwd() / "embedding.npy", embedding)
    pd.Series(proteins).to_csv(Path.cwd() / "proteins.txt", index=False, header=False)

    protein_info_file = Path("data/ppi/protein_info_900.csv")
    mapped_proteins, _ = map_string_ids_to_genes(proteins, protein_info_file)

    corum_file = Path("data/ppi/corum_complexes.txt")
    corum_complexes = load_corum(corum_file)

    results_df = validate_embedding_against_corum(
        embedding, mapped_proteins, corum_complexes, top_n=10
    )
    results_df.to_csv(Path.cwd() / "validation_results.csv", index=False)
