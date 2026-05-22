from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import subprocess
import tempfile


BASE = Path("/LOCAL/fmahner/similarity-factorization")
HURI_PATH = Path("/LOCAL/fmahner/PPI-Prediction-Project/Interactomes/HuRI.csv")
PRED_DIR = BASE / "experiments/ppi/outputs/link_prediction/HuRI/topk"
FULL_PRED_DIR = BASE / "experiments/ppi/outputs/link_prediction/HuRI/predictions"
OUTPUT_DIR = BASE / "experiments/ppi/development/outputs"
SPLITS_DIR = BASE / "data/ppi/splits/HuRI"
Y2H_PATH = BASE / "data/ppi/y2h_validation.tsv"
SRF_TOP5000_PATH = PRED_DIR / "srf_top5000.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_huri():
    edges = pd.read_csv(HURI_PATH)
    g = nx.from_pandas_edgelist(edges, "source", "target")
    return g


def load_node_mapping():
    nodes_file = SPLITS_DIR / "nodes.csv"
    nodes = pd.read_csv(nodes_file)["node_id"].values
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    return idx_to_node, node_to_idx


def get_top_predictions(pred_file, n=500):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        cmd = f"(head -1 {pred_file} && tail -n +2 {pred_file} | sort -t, -k4 -gr | head -{n}) > {tmp.name}"
        subprocess.run(cmd, shell=True, check=True)
        df = pd.read_csv(tmp.name)
        Path(tmp.name).unlink()
        return df


def get_degree_distribution(degrees: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    degree_values = np.array(list(degrees.values()))
    unique_degrees, counts = np.unique(degree_values, return_counts=True)
    probabilities = counts / counts.sum()
    return unique_degrees, probabilities


def collect_protein_degrees(
    predictions: pd.DataFrame,
    degrees: dict[str, int],
    count_duplicates: bool = True,
) -> list[int]:
    if count_duplicates:
        protein_degrees = []
        for _, row in predictions.iterrows():
            source, target = row["source"], row["target"]
            if source in degrees:
                protein_degrees.append(degrees[source])
            if target in degrees:
                protein_degrees.append(degrees[target])
    else:
        unique_proteins = set(predictions["source"]) | set(predictions["target"])
        protein_degrees = [degrees[p] for p in unique_proteins if p in degrees]

    return protein_degrees


def find_nearest_probability(
    mean_degree: float,
    unique_degrees: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    nearest_idx = np.argmin(np.abs(unique_degrees - mean_degree))
    return probabilities[nearest_idx]


def plot_figure_4b(count_duplicates: bool = True):
    g = load_huri()
    degrees = dict(g.degree())

    unique_degrees, probabilities = get_degree_distribution(degrees)

    methods = ["srf", "cn", "aa", "ra", "jc", "skipgnn"]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.loglog(
        unique_degrees,
        probabilities,
        "o",
        color="white",
        markeredgecolor="black",
        markersize=5,
        label="HuRI distribution",
    )
    import seaborn as sns

    colors = sns.color_palette("husl", len(methods))

    for i, method in enumerate(methods):
        pred_file = PRED_DIR / f"{method}_top500.csv"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found, skipping {method}")
            continue

        predictions = pd.read_csv(pred_file)
        protein_degrees = collect_protein_degrees(
            predictions, degrees, count_duplicates
        )

        if len(protein_degrees) == 0:
            print(f"Warning: No valid degrees found for {method}")
            continue

        mean_degree = np.mean(protein_degrees)
        prob_at_mean = find_nearest_probability(
            mean_degree, unique_degrees, probabilities
        )

        color = colors[i]
        ax.loglog(
            mean_degree,
            prob_at_mean,
            "o",
            markersize=10,
            color=color,
            label=method.upper(),
        )

        print(
            f"{method.upper()}: mean_degree={mean_degree:.1f}, "
            f"n_proteins={'all mentions' if count_duplicates else len(set(predictions['source']) | set(predictions['target']))}"
        )

    ax.set_xlabel("$k$", fontsize=14)
    ax.set_ylabel("$P(k)$", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "figure_4b.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


def load_validated_ppis() -> set[tuple[str, str]]:
    y2h = pd.read_csv(Y2H_PATH, sep="\t")
    positive_mask = (
        (y2h["score_assay1"] == "Positive")
        | (y2h["score_assay2"] == "Positive")
        | (y2h["score_assay3"] == "Positive")
    )
    validated = y2h[positive_mask][["source", "target"]].values
    validated_set = {tuple(sorted([s, t])) for s, t in validated}
    print(f"Loaded {len(validated_set)} validated PPIs from Y2H")
    return validated_set


def convert_ppis_to_indices(
    ppis: set[tuple[str, str]], node_to_idx: dict[str, int]
) -> set[tuple[int, int]]:
    ppi_indices = set()
    for source, target in ppis:
        if source in node_to_idx and target in node_to_idx:
            idx_pair = tuple(sorted([node_to_idx[source], node_to_idx[target]]))
            ppi_indices.add(idx_pair)
    return ppi_indices


def search_full_predictions_for_method(
    method: str,
    validated_ppi_indices: set[tuple[int, int]],
    idx_to_node: dict[int, str],
) -> set[tuple[str, str]]:
    found_ppis = set()

    for fold in range(10):
        pred_file = FULL_PRED_DIR / f"fold{fold}_{method}_predictions.csv"
        if not pred_file.exists():
            continue

        print(f"  Searching {pred_file.name}...")
        df = pd.read_csv(pred_file)

        df["sorted_pair"] = df.apply(
            lambda row: tuple(sorted([int(row["node_i"]), int(row["node_j"])])), axis=1
        )

        mask = df["sorted_pair"].isin(validated_ppi_indices)
        matches = df[mask]

        for idx_pair in matches["sorted_pair"]:
            source = idx_to_node[idx_pair[0]]
            target = idx_to_node[idx_pair[1]]
            found_ppis.add((source, target))

    return found_ppis


def search_all_methods_full_predictions(
    methods: list[str],
    validated_ppis: set[tuple[str, str]],
) -> dict[str, set[tuple[str, str]]]:
    idx_to_node, node_to_idx = load_node_mapping()
    validated_ppi_indices = convert_ppis_to_indices(validated_ppis, node_to_idx)

    print(f"Converted {len(validated_ppi_indices)} validated PPIs to indices")
    print(f"Searching full prediction files across 10 folds...\n")

    method_predictions = {}
    for method in methods:
        print(f"{method.upper()}:")
        found = search_full_predictions_for_method(
            method, validated_ppi_indices, idx_to_node
        )
        method_predictions[method] = found
        print(f"  Found {len(found)} validated PPIs\n")

    return method_predictions


def load_method_predictions(methods: list[str]) -> dict[str, set[tuple[str, str]]]:
    predictions = {}
    for method in methods:
        pred_file = PRED_DIR / f"{method}_top500.csv"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found, skipping {method}")
            continue

        df = pd.read_csv(pred_file)
        pred_set = {
            tuple(sorted([row["source"], row["target"]])) for _, row in df.iterrows()
        }
        predictions[method] = pred_set
        print(f"{method.upper()}: {len(pred_set)} predictions")

    return predictions


def load_predictions_from_file(
    pred_file: Path, topk: int | None = None
) -> set[tuple[str, str]]:
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    df = pd.read_csv(pred_file)
    if topk is not None:
        df = df.head(topk)
    return {tuple(sorted([row["source"], row["target"]])) for _, row in df.iterrows()}


def find_validated_predictions(
    validated_ppis: set[tuple[str, str]],
    method_predictions: dict[str, set[tuple[str, str]]],
) -> dict[tuple[str, str], list[str]]:
    edge_methods = {}
    for edge in validated_ppis:
        predicting_methods = [
            m for m, preds in method_predictions.items() if edge in preds
        ]
        if predicting_methods:
            edge_methods[edge] = predicting_methods
    return edge_methods


def build_validated_network(edge_methods: dict[tuple[str, str], list[str]]) -> nx.Graph:
    g = nx.Graph()
    for edge, methods_list in edge_methods.items():
        if edge[0] == edge[1]:
            g.add_node(edge[0], methods=methods_list, n_methods=len(methods_list))
            continue
        g.add_edge(edge[0], edge[1], methods=methods_list, n_methods=len(methods_list))
    return g


def plot_validated_network(
    validated_ppis: set[tuple[str, str]],
    method_predictions: dict[str, set[tuple[str, str]]],
    colors: dict[str, str],
    output_filename: str,
    title: str,
    legend_order: list[str] | None = None,
):
    edge_methods = find_validated_predictions(validated_ppis, method_predictions)

    print(f"\nValidated PPIs predicted by at least one method: {len(edge_methods)}")

    g = build_validated_network(edge_methods)

    if g.number_of_nodes() == 0:
        print("No validated PPIs were predicted by any method")
        return

    print(f"Network: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    isolated_nodes = list(nx.isolates(g))
    print(f"Isolated nodes (degree 0): {len(isolated_nodes)}")

    components = list(nx.connected_components(g))
    node_to_component = {
        node: idx for idx, comp in enumerate(components) for node in comp
    }

    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(g, k=0.5, iterations=50, seed=42)

    degrees = dict(g.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [50 + 300 * (degrees[node] / max_degree) for node in g.nodes()]
    node_colors = [node_to_component[node] for node in g.nodes()]

    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.7,
        ax=ax,
    )

    multi_method_present = False
    for edge in g.edges():
        methods_list = g.edges[edge]["methods"]
        n_methods = g.edges[edge]["n_methods"]

        if n_methods == 1:
            edge_color = colors.get(methods_list[0], "#000000")
            width = 1.0
        else:
            edge_color = "black"
            width = 1.0 + n_methods * 0.5
            multi_method_present = True

        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[edge],
            edge_color=edge_color,
            width=width,
            alpha=0.6,
            ax=ax,
        )

    legend_methods = legend_order or list(method_predictions.keys())
    legend_elements = [
        plt.Line2D([0], [0], color=colors.get(m, "#000000"), lw=2, label=m.upper())
        for m in legend_methods
        if m in method_predictions
    ]
    if multi_method_present:
        legend_elements.append(
            plt.Line2D([0], [0], color="black", lw=3, label="Multiple methods")
        )
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", frameon=False, fontsize=10)

    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.tight_layout()

    output_file = OUTPUT_DIR / output_filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {output_file}")


def plot_figure_6(use_full_predictions: bool = True):
    methods = ["srf", "cn", "aa", "ra", "jc", "skipgnn"]
    colors = {
        "srf": "#C0392B",
        "cn": "#95A5A6",
        "aa": "#7F8C8D",
        "ra": "#BDC3C7",
        "jc": "#34495E",
        "skipgnn": "#3498DB",
    }

    validated_ppis = load_validated_ppis()

    if use_full_predictions:
        method_predictions = search_all_methods_full_predictions(
            methods, validated_ppis
        )
    else:
        method_predictions = load_method_predictions(methods)

    plot_validated_network(
        validated_ppis,
        method_predictions,
        colors,
        "figure_6.pdf",
        f"Validated PPIs Network ({len(method_predictions)} methods)",
        legend_order=methods,
    )


def plot_srf_top5000():
    colors = {"srf": "#C0392B"}
    validated_ppis = load_validated_ppis()
    srf_predictions = load_predictions_from_file(SRF_TOP5000_PATH, topk=5000)
    method_predictions = {"srf": srf_predictions}
    plot_validated_network(
        validated_ppis,
        method_predictions,
        colors,
        "figure_6_srf_top5000.pdf",
        "SRF Top-5000 Validated PPIs",
        legend_order=["srf"],
    )


if __name__ == "__main__":
    # plot_figure_4b()
    plot_srf_top5000()
