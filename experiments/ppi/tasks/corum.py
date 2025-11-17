from __future__ import annotations

from pathlib import Path

import numpy as np
from pysrf import SRF


def run_corum_validation(
    string_data_path: Path,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    if string_data_path.suffix == ".csv":
        from analyses.ppi.graph_utils import load_network
        from analyses.ppi.graph_utils import build_adjacency_with_nan

        g, _ = load_network(string_data_path)
        nodes = sorted(g.nodes())
        adj = build_adjacency_with_nan(g, g, nodes, fill_missing_with_nan=False)
    elif string_data_path.suffix == ".npy":
        adj = np.load(string_data_path)
        proteins_file = string_data_path.parent / "proteins.txt"
        nodes = (
            np.loadtxt(proteins_file, dtype=str).tolist()
            if proteins_file.exists()
            else [f"protein_{i}" for i in range(adj.shape[0])]
        )
    else:
        raise ValueError(f"Unknown file format: {string_data_path.suffix}")

    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=2000,
        max_inner=50,
        tol=1e-4,
        verbose=0,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        loss="frobenius",
    )
    embedding = model.fit_transform(adj)
    return embedding, nodes

