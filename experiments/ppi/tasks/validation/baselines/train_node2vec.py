import numpy as np
import pandas as pd
from pathlib import Path
from node2vec import Node2Vec
import networkx as nx
from analyses.ppi.graph_utils import load_network
from analyses.ppi.corum import map_string_ids_to_genes


def train_node2vec(
    g: nx.Graph,
    dimensions: int = 50,
    walk_length: int = 30,
    num_walks: int = 200,
    workers: int = 8,
) -> dict:
    node2vec = Node2Vec(
        g,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=1,
        q=1,
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    return model


def main():
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "validation" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading STRING network...")
    string_path = data_dir / "STRING_human_min900_v12.csv"
    g, _ = load_network(string_path)
    print(f"Network: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    all_nodes = sorted(g.nodes())
    mapping_file = data_dir / "protein_info_900.csv"
    if mapping_file.exists():
        all_nodes, n_mapped = map_string_ids_to_genes(all_nodes, mapping_file)

    print("\nTraining Node2Vec (rank=50)...")
    model = train_node2vec(g, dimensions=50, walk_length=30, num_walks=200, workers=8)

    print("\nExtracting embeddings...")
    embedding = np.zeros((len(all_nodes), 50))
    for i, node in enumerate(all_nodes):
        if node in model.wv:
            embedding[i] = model.wv[node]

    np.save(output_dir / "node2vec_embedding.npy", embedding)

    with open(output_dir / "node2vec_proteins.txt", "w") as f:
        for node in all_nodes:
            f.write(f"{node}\n")

    print(f"\nNode2Vec embedding saved: {embedding.shape}")
    print(f"Output: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
