"""Quick debug: Check if OpenNE methods are learning/converging."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ppi.embeddings import fit_openne
from experiments.ppi.node_classification_sweep import load_ppi, get_subgraph

N_NODES = 200
RANK = 32
SEED = 42


def main():
    print("Loading PPI...")
    g_full, _, _ = load_ppi()
    g = get_subgraph(g_full, N_NODES)
    print(f"Subgraph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    for method in ["deepwalk", "node2vec", "line"]:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()}")
        print("="*50)

        emb = fit_openne(method, g, RANK, SEED)

        if emb:
            W = np.array([emb[n] for n in g.nodes() if n in emb])
            print(f"Embedding shape: {W.shape}")
            print(f"Mean: {W.mean():.4f}, Std: {W.std():.4f}")
            print(f"Min: {W.min():.4f}, Max: {W.max():.4f}")
            print(f"Non-zero: {(W != 0).mean()*100:.1f}%")

            # Check if embeddings are degenerate
            if W.std() < 0.01:
                print("WARNING: Embeddings look degenerate (very low variance)")
            elif np.allclose(W, W[0]):
                print("WARNING: All embeddings identical!")
            else:
                print("OK: Embeddings have reasonable variance")
        else:
            print("ERROR: No embeddings returned")


if __name__ == "__main__":
    main()
