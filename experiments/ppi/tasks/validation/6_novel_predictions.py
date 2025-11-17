import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from analyses.ppi.corum import load_corum
from analyses.ppi.graph_utils import load_network


CANONICAL_COMPLEXES = [
    "Proteasome",
    "Ribosome",
    "Spliceosome",
    "Mediator complex",
    "Exosome",
    "CCT complex",
    "eIF3 complex",
    "Nuclear pore complex",
]


def find_matching_corum_complexes(corum_complexes: dict, query: str) -> list:
    matches = []
    query_lower = query.lower()
    for name, members in corum_complexes.items():
        if query_lower in name.lower():
            matches.append((name, members))
    return matches


def rank_candidates_for_complex(
    embedding: np.ndarray,
    proteins: np.ndarray,
    complex_members: list[str],
    exclude_members: bool = True,
) -> pd.DataFrame:
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    member_indices = [protein_to_idx[m] for m in complex_members if m in protein_to_idx]

    if len(member_indices) == 0:
        return pd.DataFrame()

    complex_centroid = embedding[member_indices].mean(axis=0, keepdims=True)

    similarities = cosine_similarity(embedding, complex_centroid).flatten()

    candidates = []
    for idx, protein in enumerate(proteins):
        if exclude_members and protein in complex_members:
            continue

        candidates.append({"protein": protein, "similarity": similarities[idx]})

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values("similarity", ascending=False)

    return candidates_df


def get_string_evidence(protein: str, complex_members: list[str], g: nx.Graph) -> dict:
    if protein not in g:
        return {"n_edges": 0, "edges": []}

    edges = []
    for member in complex_members:
        if member in g and g.has_edge(protein, member):
            edges.append(member)

    return {"n_edges": len(edges), "edges": edges}


def analyze_complex(
    complex_name: str,
    complex_members: list[str],
    embedding: np.ndarray,
    proteins: np.ndarray,
    g: nx.Graph,
    n_candidates: int = 50,
) -> pd.DataFrame:
    candidates_df = rank_candidates_for_complex(
        embedding, proteins, complex_members, exclude_members=True
    )

    if len(candidates_df) == 0:
        return pd.DataFrame()

    top_candidates = candidates_df.head(n_candidates).copy()

    string_edges = []
    for _, row in top_candidates.iterrows():
        protein = row["protein"]
        evidence = get_string_evidence(protein, complex_members, g)
        string_edges.append(evidence["n_edges"])

    top_candidates["string_edges_to_complex"] = string_edges
    top_candidates["complex_name"] = complex_name
    top_candidates["complex_size"] = len(complex_members)

    return top_candidates


def plot_candidate_rankings(
    results_df: pd.DataFrame, complex_name: str, output_dir: Path
):
    complex_df = results_df[results_df["complex_name"] == complex_name].head(20)

    if len(complex_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    y_pos = np.arange(len(complex_df))
    ax.barh(y_pos, complex_df["similarity"], alpha=0.7, color="#2E86AB")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(complex_df["protein"], fontsize=9)
    ax.set_xlabel("Similarity to complex centroid")
    ax.set_title(f"{complex_name}: top candidates by embedding similarity")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    ax = axes[1]
    colors = [
        "#2E86AB" if edges > 0 else "#CCCCCC"
        for edges in complex_df["string_edges_to_complex"]
    ]
    ax.barh(y_pos, complex_df["string_edges_to_complex"], alpha=0.7, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(complex_df["protein"], fontsize=9)
    ax.set_xlabel("Number of STRING edges to complex")
    ax.set_title(f"{complex_name}: STRING evidence")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    safe_name = complex_name.replace(" ", "_").replace("/", "_")
    plt.savefig(
        output_dir / f"{safe_name}_candidates.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_summary_table(results_df: pd.DataFrame, output_dir: Path):
    summary = []

    for complex_name in results_df["complex_name"].unique():
        complex_df = results_df[results_df["complex_name"] == complex_name]

        top10 = complex_df.head(10)

        summary.append(
            {
                "complex": complex_name,
                "complex_size": complex_df["complex_size"].iloc[0],
                "candidates_evaluated": len(complex_df),
                "top1_protein": top10.iloc[0]["protein"],
                "top1_similarity": top10.iloc[0]["similarity"],
                "top1_string_edges": top10.iloc[0]["string_edges_to_complex"],
                "top10_with_string_evidence": (
                    top10["string_edges_to_complex"] > 0
                ).sum(),
                "mean_top10_similarity": top10["similarity"].mean(),
                "mean_top10_string_edges": top10["string_edges_to_complex"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("mean_top10_string_edges", ascending=False)

    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return summary_df


def main():
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    validation_output_dir = (
        base_dir
        / "experiments"
        / "ppi"
        / "validation"
        / "outputs"
        / "novel_predictions"
    )
    validation_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading rank-50 SRF embedding...")
    embedding = np.load(output_dir / "embedding.npy")
    print(f"Embedding shape: {embedding.shape}")

    with open(output_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])
    print(f"Number of proteins: {len(proteins)}")

    print("\nLoading CORUM complexes...")
    corum_path = data_dir / "corum_complexes.txt"
    corum_complexes = load_corum(corum_path)
    print(f"Total CORUM complexes: {len(corum_complexes)}")

    print("\nLoading STRING network for evidence...")
    string_path = data_dir / "STRING_human_min900_v12.csv"
    g, _ = load_network(string_path)
    print(f"STRING nodes: {g.number_of_nodes()}")
    print(f"STRING edges: {g.number_of_edges()}")

    print("\nFinding canonical complexes in CORUM...")
    all_results = []

    for canonical_name in CANONICAL_COMPLEXES:
        matches = find_matching_corum_complexes(corum_complexes, canonical_name)

        if len(matches) == 0:
            print(f"  No match for: {canonical_name}")
            continue

        complex_name, complex_members = matches[0]
        print(
            f"  {canonical_name}: found '{complex_name}' with {len(complex_members)} members"
        )

        results = analyze_complex(
            complex_name, list(complex_members), embedding, proteins, g, n_candidates=50
        )

        if len(results) > 0:
            all_results.append(results)

    if len(all_results) == 0:
        print("No results generated")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(validation_output_dir / "all_candidates.csv", index=False)

    print("\nCreating summary table...")
    summary_df = create_summary_table(all_results_df, validation_output_dir)

    print("\nGenerating candidate plots for top 3 complexes...")
    top_complexes = summary_df.head(3)["complex"].tolist()
    for complex_name in top_complexes:
        plot_candidate_rankings(all_results_df, complex_name, validation_output_dir)

    print("\n" + "=" * 80)
    print("NOVEL COMPLEX MEMBER PREDICTIONS")
    print("=" * 80)
    print("\nSummary (top 5 complexes by STRING evidence):")
    print(summary_df.head().to_string(index=False))

    print("\nTop 5 candidates per complex:")
    for complex_name in top_complexes:
        complex_df = all_results_df[all_results_df["complex_name"] == complex_name]
        print(f"\n{complex_name}:")
        print(
            complex_df[["protein", "similarity", "string_edges_to_complex"]]
            .head()
            .to_string(index=False)
        )

    print(f"\nResults saved to: {validation_output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
