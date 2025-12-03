"""
C. elegans Connectome Factorization with SRF.

Applies SRF to the weighted connectome (synapse counts) to discover
interpretable neural circuit modules.
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

from pysrf import SRF

log = logging.getLogger(__name__)


def load_connectome(data_file: Path, edge_type: str = "all") -> pd.DataFrame:
    """Load edgelist and filter by edge type."""
    df = pd.read_csv(data_file)
    df["Source"] = df["Source"].str.strip()
    df["Target"] = df["Target"].str.strip()

    if edge_type == "chemical":
        df = df[df["Type"] == "chemical"]
    elif edge_type == "electrical":
        df = df[df["Type"] == "electrical"]

    return df


def build_adjacency_matrix(
    df: pd.DataFrame,
    symmetry: str = "symmetrize",
    normalize: str = "none",
) -> tuple[np.ndarray, list[str]]:
    """
    Build adjacency matrix from edgelist.

    Args:
        df: Edgelist with Source, Target, Weight columns
        symmetry: How to handle directed edges
            - "gap_only": Use only electrical (gap junction) edges
            - "symmetrize": (A + A.T) / 2
            - "max": max(A, A.T)
        normalize: Normalization method
            - "none": Raw synapse counts
            - "log": log(1 + count)
            - "max": Divide by max value

    Returns:
        Adjacency matrix and list of neuron names
    """
    neurons = sorted(set(df["Source"].unique()) | set(df["Target"].unique()))
    neuron_to_idx = {n: i for i, n in enumerate(neurons)}
    n = len(neurons)

    A = np.zeros((n, n), dtype=np.float64)

    for _, row in df.iterrows():
        i = neuron_to_idx[row["Source"]]
        j = neuron_to_idx[row["Target"]]
        A[i, j] += row["Weight"]

    if symmetry == "gap_only":
        df_gap = df[df["Type"] == "electrical"]
        A = np.zeros((n, n), dtype=np.float64)
        for _, row in df_gap.iterrows():
            i = neuron_to_idx[row["Source"]]
            j = neuron_to_idx[row["Target"]]
            A[i, j] += row["Weight"]
        A = (A + A.T) / 2
    elif symmetry == "symmetrize":
        A = (A + A.T) / 2
    elif symmetry == "max":
        A = np.maximum(A, A.T)

    if normalize == "log":
        A = np.log1p(A)
    elif normalize == "max":
        if A.max() > 0:
            A = A / A.max()

    # Set diagonal to NaN (self-connections are trivial)
    np.fill_diagonal(A, np.nan)

    return A, neurons


def run(cfg: DictConfig) -> None:
    """Main entry point."""
    log.info(f"Loading connectome from {cfg.data_file}")

    df = load_connectome(Path(cfg.data_file), cfg.edge_type)
    log.info(f"Loaded {len(df)} edges")

    A, neurons = build_adjacency_matrix(df, cfg.symmetry, cfg.normalize)
    log.info(f"Built {A.shape[0]}x{A.shape[1]} adjacency matrix")
    log.info(f"Non-zero entries: {np.count_nonzero(A)}")
    log.info(f"Value range: {A.min():.2f} - {A.max():.2f}")

    model = SRF(
        rank=cfg.rank,
        max_outer=50,
        max_inner=30,
        tol=1e-4,
        verbose=1,
        random_state=cfg.seed,
    )

    log.info(f"Fitting SRF with rank={cfg.rank}")
    W = model.fit_transform(A)
    log.info(f"Converged in {model.n_iter_} iterations")

    reconstruction = W @ W.T

    # Compute metrics ignoring diagonal (NaN)
    off_diag = ~np.isnan(A)
    mse = np.mean((A[off_diag] - reconstruction[off_diag]) ** 2)
    observed_mask = off_diag & (A > 0)
    mse_observed = np.mean((A[observed_mask] - reconstruction[observed_mask]) ** 2)

    log.info(f"Reconstruction MSE (all): {mse:.6f}")
    log.info(f"Reconstruction MSE (observed): {mse_observed:.6f}")

    output_dir = Path.cwd()
    np.save(output_dir / "W.npy", W)
    np.save(output_dir / "adjacency.npy", A)

    with open(output_dir / "neurons.json", "w") as f:
        json.dump(neurons, f)

    results = {
        "n_neurons": len(neurons),
        "n_edges": len(df),
        "rank": cfg.rank,
        "symmetry": cfg.symmetry,
        "normalize": cfg.normalize,
        "edge_type": cfg.edge_type,
        "n_iter": model.n_iter_,
        "mse_all": float(mse),
        "mse_observed": float(mse_observed),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Saved results to {output_dir}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
