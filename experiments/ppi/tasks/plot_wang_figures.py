from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import subprocess
import tempfile

HURI_PATH = Path("/LOCAL/fmahner/PPI-Prediction-Project/Interactomes/HuRI.csv")
PRED_DIR = Path(__file__).parent / "outputs/link_prediction/HuRI/predictions"
EXTERNAL_PRED_DIR = Path(__file__).parent / "outputs/link_prediction/HuRI/external"
OUTPUT_DIR = Path(__file__).parent / "outputs"
SPLITS_DIR = Path(__file__).parent.parent.parent / "data/ppi/splits/HuRI"
Y2H_PATH = Path(__file__).parent.parent.parent / "data/ppi/y2h_validation.tsv"


def load_huri():
    edges = pd.read_csv(HURI_PATH)
    g = nx.from_pandas_edgelist(edges, "source", "target")
    return g


def load_node_mapping():
    nodes_file = SPLITS_DIR / "nodes.csv"
    nodes = pd.read_csv(nodes_file)["node_id"].values
    return {i: node for i, node in enumerate(nodes)}


def get_top_predictions(pred_file, n=500):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        cmd = f"(head -1 {pred_file} && tail -n +2 {pred_file} | sort -t, -k4 -gr | head -{n}) > {tmp.name}"
        subprocess.run(cmd, shell=True, check=True)
        df = pd.read_csv(tmp.name)
        Path(tmp.name).unlink()
        return df


def plot_figure_4b(n_folds=10):
    g = load_huri()
    degrees = dict(g.degree())
    deg_vals = np.array(list(degrees.values()))
    unique_deg, counts = np.unique(deg_vals, return_counts=True)
    prob = counts / counts.sum()

    idx_to_node = load_node_mapping()
    methods = ["srf", "cn", "aa", "ra", "jc", "skipgnn"]
    colors = {
        "srf": "#C0392B",
        "cn": "#95A5A6",
        "aa": "#7F8C8D",
        "ra": "#BDC3C7",
        "jc": "#34495E",
        "skipgnn": "#3498DB",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.loglog(
        unique_deg, prob, "o", color="white", markeredgecolor="black", markersize=5
    )

    for method in methods:
        all_deg = []
        for fold in range(n_folds):
            pred_file = PRED_DIR / f"fold{fold}_{method}_predictions.csv"
            if not pred_file.exists():
                continue

            df = get_top_predictions(pred_file, n=500)
            for _, row in df.iterrows():
                node_i = idx_to_node.get(int(row["node_i"]))
                node_j = idx_to_node.get(int(row["node_j"]))
                if node_i and node_i in degrees:
                    all_deg.append(degrees[node_i])
                if node_j and node_j in degrees:
                    all_deg.append(degrees[node_j])

        if len(all_deg) > 0:
            mean_deg = np.mean(all_deg)
            idx = np.argmin(np.abs(unique_deg - mean_deg))
            prob_at = prob[idx]
            ax.loglog(
                mean_deg,
                prob_at,
                "o",
                color=colors.get(method, "#000"),
                markersize=10,
                label=method.upper(),
            )
            print(f"{method.upper()}: mean_degree={mean_deg:.1f}")

    ax.set_xlabel("$k$", fontsize=14)
    ax.set_ylabel("$P(k)$", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "figure_4b.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR}/figure_4b.pdf")


def plot_figure_6():
    y2h = pd.read_csv(Y2H_PATH, sep="\t")
    positive_mask = (
        (y2h["score_assay1"] == "Positive")
        | (y2h["score_assay2"] == "Positive")
        | (y2h["score_assay3"] == "Positive")
    )
    validated_ppis = y2h[positive_mask][["source", "target"]].values
    print(f"Loaded {len(validated_ppis)} validated PPIs")

    methods = ["srf", "cn", "aa", "ra", "jc", "skipgnn"]
    colors = {
        "srf": "#C0392B",
        "cn": "#95A5A6",
        "aa": "#7F8C8D",
        "ra": "#BDC3C7",
        "jc": "#34495E",
        "skipgnn": "#3498DB",
    }

    method_predictions = {}
    for method in methods:
        pred_file = EXTERNAL_PRED_DIR / f"{method}_top500.csv"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found, skipping {method}")
            continue

        df = pd.read_csv(pred_file)
        pred_set = {
            tuple(sorted([row["source"], row["target"]])) for _, row in df.iterrows()
        }
        method_predictions[method] = pred_set
        print(f"{method.upper()}: {len(pred_set)} top-500 predictions loaded")

    edge_methods = {}
    for source, target in validated_ppis:
        edge = tuple(sorted([source, target]))
        methods_predicting = [
            m for m, preds in method_predictions.items() if edge in preds
        ]
        if methods_predicting:
            edge_methods[edge] = methods_predicting

    print(f"\nValidated PPIs predicted by at least one method: {len(edge_methods)}")

    g = nx.Graph()
    for edge, methods_list in edge_methods.items():
        g.add_edge(edge[0], edge[1], methods=methods_list, n_methods=len(methods_list))

    if g.number_of_nodes() == 0:
        print("No validated PPIs were predicted by any method in top-500")
        return

    print(f"Network: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    components = list(nx.connected_components(g))
    node_to_component = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = idx

    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(g, k=0.5, iterations=50, seed=42)

    degrees = dict(g.degree())
    node_sizes = [
        50 + 300 * (degrees[node] / max(degrees.values())) for node in g.nodes()
    ]
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

    for edge in g.edges():
        methods_list = g.edges[edge]["methods"]
        n_methods = g.edges[edge]["n_methods"]

        if n_methods == 1:
            edge_color = colors[methods_list[0]]
            width = 0.5
        else:
            edge_color = "black"
            width = 0.5 + n_methods * 0.3

        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[edge],
            edge_color=edge_color,
            width=width,
            alpha=0.6,
            ax=ax,
        )

    legend_elements = [
        plt.Line2D([0], [0], color=colors[m], lw=2, label=m.upper()) for m in methods
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color="black", lw=2, label="Multiple methods")
    )
    ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=9)

    ax.set_title(f"Network of {g.number_of_edges()} Validated PPIs", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "figure_6.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR}/figure_6.pdf")


if __name__ == "__main__":
    plot_figure_4b()
    plot_figure_6()
