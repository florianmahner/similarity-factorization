"""
Visualize SRF factors for C. elegans connectome.

Compares factor loadings to known neuron types (sensory, interneuron, motor).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Known neuron type prefixes from WormAtlas
# Sensory neurons typically have names starting with these prefixes
SENSORY_PREFIXES = [
    "ADF", "ADL", "AFD", "ALM", "ALN", "AQR", "ASE", "ASG", "ASH", "ASI",
    "ASJ", "ASK", "AVM", "AWA", "AWB", "AWC", "BAG", "CEP", "FLP", "IL1",
    "IL2", "OLL", "OLQ", "PDE", "PHA", "PHB", "PHC", "PLM", "PLN", "PQR",
    "PVD", "PVM", "URX", "URY",
]

# Motor neuron prefixes
MOTOR_PREFIXES = [
    "DA", "DB", "DD", "VA", "VB", "VC", "VD", "AS", "RMD", "RME", "RMF",
    "RMG", "RMH", "SMB", "SMD", "SAA", "SAB", "SIA", "SIB", "URA", "URB",
]

# Pharyngeal neurons
PHARYNGEAL_PREFIXES = ["I1", "I2", "I3", "I4", "I5", "I6", "M1", "M2", "M3", "M4", "M5", "MC", "MI", "NSM", "pm"]


def classify_neuron(name: str) -> str:
    """Classify neuron by name prefix."""
    name_upper = name.upper()

    for prefix in SENSORY_PREFIXES:
        if name_upper.startswith(prefix):
            return "sensory"

    for prefix in MOTOR_PREFIXES:
        if name_upper.startswith(prefix):
            return "motor"

    for prefix in PHARYNGEAL_PREFIXES:
        if name.startswith(prefix):
            return "pharyngeal"

    return "interneuron"


def load_results(folder: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load SRF results from output folder."""
    W = np.load(folder / "W.npy")
    A = np.load(folder / "adjacency.npy")

    with open(folder / "neurons.json") as f:
        neurons = json.load(f)

    with open(folder / "results.json") as f:
        results = json.load(f)

    return W, A, neurons, results


def plot_factor_heatmap(W: np.ndarray, neurons: list[str], output_dir: Path):
    """Plot heatmap of factor loadings sorted by neuron type."""
    types = [classify_neuron(n) for n in neurons]
    type_order = {"sensory": 0, "interneuron": 1, "motor": 2, "pharyngeal": 3}
    sort_idx = np.argsort([type_order[t] for t in types])

    W_sorted = W[sort_idx]
    types_sorted = [types[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(W_sorted, cmap="viridis", ax=ax, cbar_kws={"label": "Factor loading"})

    type_boundaries = []
    current_type = types_sorted[0]
    for i, t in enumerate(types_sorted):
        if t != current_type:
            type_boundaries.append(i)
            current_type = t

    for boundary in type_boundaries:
        ax.axhline(boundary, color="red", linewidth=2)

    ax.set_xlabel("Factor")
    ax.set_ylabel("Neuron (sorted by type)")
    ax.set_title("SRF Factor Loadings by Neuron Type")

    type_positions = []
    prev = 0
    for boundary in type_boundaries + [len(types_sorted)]:
        type_positions.append((prev + boundary) / 2)
        prev = boundary

    unique_types = []
    current = types_sorted[0]
    unique_types.append(current)
    for t in types_sorted:
        if t != current:
            unique_types.append(t)
            current = t

    for pos, label in zip(type_positions, unique_types):
        ax.text(-0.5, pos, label, ha="right", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "factor_heatmap.png", dpi=150)
    plt.close()


def plot_factor_type_distribution(W: np.ndarray, neurons: list[str], output_dir: Path):
    """Plot distribution of factor loadings by neuron type."""
    types = [classify_neuron(n) for n in neurons]
    n_factors = W.shape[1]

    fig, axes = plt.subplots(2, min(5, n_factors), figsize=(15, 6))
    axes = axes.flatten()

    for i in range(min(10, n_factors)):
        ax = axes[i]
        df = pd.DataFrame({"loading": W[:, i], "type": types})
        sns.boxplot(data=df, x="type", y="loading", ax=ax, order=["sensory", "interneuron", "motor", "pharyngeal"])
        ax.set_title(f"Factor {i}")
        ax.set_xlabel("")
        if i % 5 != 0:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "factor_by_type.png", dpi=150)
    plt.close()


def compute_clustering_metrics(W: np.ndarray, neurons: list[str]) -> dict:
    """Compute clustering metrics comparing SRF factors to neuron types."""
    types = [classify_neuron(n) for n in neurons]
    type_to_int = {t: i for i, t in enumerate(set(types))}
    true_labels = [type_to_int[t] for t in types]

    pred_labels = np.argmax(W, axis=1)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return {"ari": ari, "nmi": nmi, "n_types": len(set(types))}


def plot_reconstruction(A: np.ndarray, W: np.ndarray, output_dir: Path):
    """Plot original vs reconstructed adjacency matrix."""
    reconstruction = W @ W.T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    A_plot = A.copy()
    A_plot[np.isnan(A_plot)] = 0

    im0 = axes[0].imshow(A_plot, cmap="hot", aspect="auto")
    axes[0].set_title("Original Adjacency")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(reconstruction, cmap="hot", aspect="auto")
    axes[1].set_title("Reconstruction (W @ W.T)")
    plt.colorbar(im1, ax=axes[1])

    diff = A_plot - reconstruction
    im2 = axes[2].imshow(diff, cmap="RdBu", aspect="auto", vmin=-diff.max(), vmax=diff.max())
    axes[2].set_title("Difference")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to results folder")
    parser.add_argument("--cleanup", action="store_true", help="Remove intermediate files")
    args = parser.parse_args()

    folder = Path(args.folder)
    print(f"Loading results from {folder}")

    W, A, neurons, results = load_results(folder)
    print(f"Loaded W: {W.shape}, A: {A.shape}, neurons: {len(neurons)}")

    print("\n=== Results ===")
    print(f"Rank: {results['rank']}")
    print(f"Symmetry: {results['symmetry']}")
    print(f"MSE (observed): {results['mse_observed']:.4f}")

    types = [classify_neuron(n) for n in neurons]
    type_counts = pd.Series(types).value_counts()
    print(f"\n=== Neuron Types ===")
    print(type_counts)

    metrics = compute_clustering_metrics(W, neurons)
    print(f"\n=== Clustering Metrics ===")
    print(f"ARI (factors vs types): {metrics['ari']:.3f}")
    print(f"NMI (factors vs types): {metrics['nmi']:.3f}")

    print("\n=== Generating plots ===")
    plot_factor_heatmap(W, neurons, folder)
    print("  Saved factor_heatmap.png")

    plot_factor_type_distribution(W, neurons, folder)
    print("  Saved factor_by_type.png")

    plot_reconstruction(A, W, folder)
    print("  Saved reconstruction.png")

    metrics_file = folder / "clustering_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved clustering_metrics.json")


if __name__ == "__main__":
    main()
