"""
Extract embeddings using OpenNE command-line interface.
Converts STRING network to OpenNE format and runs OpenNE models.
"""

from pathlib import Path
import argparse
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# Project paths
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data/ppi"


def load_string_network(csv_path: Path, n_subset: int = None) -> nx.Graph:
    """Load STRING PPI network."""
    print(f"Loading STRING network from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} edges")

    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_edge(row.iloc[0], row.iloc[1], weight=1.0)

    print(f"Full graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    if n_subset and n_subset < g.number_of_nodes():
        print(f"Taking subset of {n_subset} highest-degree nodes")
        top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:n_subset]
        top_nodes = [n for n, _ in top_nodes]
        g = g.subgraph(top_nodes).copy()

        if not nx.is_connected(g):
            print("Taking largest connected component")
            largest_cc = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest_cc).copy()

        print(f"Subset graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    return g


def save_openne_edgelist(g: nx.Graph, output_path: Path):
    """Save graph in OpenNE edgelist format."""
    print(f"Saving graph to {output_path}")

    # OpenNE expects: node1 node2 [weight]
    with open(output_path, 'w') as f:
        for u, v, data in g.edges(data=True):
            weight = data.get('weight', 1.0)
            f.write(f"{u} {v} {weight}\n")

    print(f"Saved {g.number_of_edges()} edges")


def run_openne_method(method: str, input_file: Path, output_file: Path,
                      dim: int = 128, epochs: int = 5, **kwargs):
    """Run OpenNE using command-line interface."""
    print(f"\n{'='*60}")
    print(f"Running OpenNE: {method}")
    print(f"{'='*60}")

    # Base command
    cmd = [
        "poetry", "run", "python", "-m", "openne",
        "--method", method,
        "--input", str(input_file),
        "--output", str(output_file),
        "--graph-format", "edgelist",
        "--representation-size", str(dim),
        "--weighted",
    ]

    # Method-specific options
    if method in ["deepwalk", "node2vec"]:
        cmd.extend([
            "--number-walks", str(kwargs.get("number_walks", 10)),
            "--walk-length", str(kwargs.get("walk_length", 40)),
            "--window-size", str(kwargs.get("window_size", 10)),
            "--workers", str(kwargs.get("workers", 4)),
        ])

        if method == "node2vec":
            cmd.extend([
                "--p", str(kwargs.get("p", 4.0)),
                "--q", str(kwargs.get("q", 1.0)),
            ])

    elif method == "line":
        cmd.extend([
            "--epochs", str(epochs),
            "--order", str(kwargs.get("order", 2)),
            "--negative-ratio", str(kwargs.get("negative_ratio", 5)),
        ])

    print(f"Command: {' '.join(cmd)}")

    # Run command
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False

    print("✓ Completed successfully")
    return True


def load_openne_embeddings(embedding_file: Path) -> dict:
    """Load embeddings from OpenNE output format."""
    print(f"Loading embeddings from {embedding_file}")

    embeddings = {}
    with open(embedding_file, 'r') as f:
        # First line: num_nodes dim
        first_line = f.readline().strip().split()
        num_nodes, dim = int(first_line[0]), int(first_line[1])
        print(f"Expected: {num_nodes} nodes, {dim} dimensions")

        # Read embeddings
        for line in f:
            parts = line.strip().split()
            node = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[node] = vector

    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def save_embeddings_npz(embeddings: dict, output_path: Path):
    """Save embeddings as .npz file."""
    print(f"Saving to {output_path}")

    nodes = sorted(embeddings.keys(), key=str)
    vectors = np.array([embeddings[n] for n in nodes])

    np.savez(output_path, nodes=np.array([str(n) for n in nodes]), vectors=vectors)
    print(f"Saved {len(nodes)} embeddings with shape {vectors.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract OpenNE embeddings")

    parser.add_argument("--string-csv", type=Path,
                       default=data_dir / "STRING_human_min900_v12.csv",
                       help="Path to STRING CSV file")
    parser.add_argument("--n-subset", type=int, default=None,
                       help="Number of nodes to sample")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["deepwalk", "line", "node2vec"],
                       help="Methods to run")
    parser.add_argument("--dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Training epochs (for LINE)")
    parser.add_argument("--output-dir", type=Path,
                       default=project_root / "outputs/openne_embeddings",
                       help="Output directory")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    g = load_string_network(args.string_csv, args.n_subset)
    graph_name = f"string_n{g.number_of_nodes()}" if args.n_subset else "string_full"

    # Save in OpenNE format
    edgelist_file = args.output_dir / f"{graph_name}.edgelist"
    save_openne_edgelist(g, edgelist_file)

    # Run each method
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Processing: {method.upper()}")
        print(f"{'='*60}")

        # Output files
        openne_output = args.output_dir / f"{graph_name}_{method}_dim{args.dim}.txt"
        npz_output = args.output_dir / f"{graph_name}_{method}_dim{args.dim}.npz"

        # Run OpenNE
        success = run_openne_method(
            method=method,
            input_file=edgelist_file,
            output_file=openne_output,
            dim=args.dim,
            epochs=args.epochs,
        )

        if success and openne_output.exists():
            # Load and convert to .npz
            embeddings = load_openne_embeddings(openne_output)
            save_embeddings_npz(embeddings, npz_output)

            # Clean up text file
            openne_output.unlink()
            print(f"✓ {method} completed successfully\n")
        else:
            print(f"✗ {method} failed\n")

    # Clean up edgelist
    edgelist_file.unlink()

    print("\n" + "="*60)
    print("All done!")
    print(f"Embeddings saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
