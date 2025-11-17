import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from analyses.ppi.corum import load_corum
from analyses.ppi.graph_utils import load_network, build_sparse_adjacency

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2


def evaluate_method_on_complex(
    embedding: np.ndarray,
    complex_idx: list[int],
    held_out_idx: int,
    proteins: np.ndarray,
    method: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    remaining_idx = [idx for idx in complex_idx if idx != held_out_idx]

    if len(remaining_idx) == 0:
        return np.zeros(len(embedding)), np.zeros(len(proteins), dtype=int)

    centroid = embedding[remaining_idx].mean(axis=0, keepdims=True)
    similarities = cosine_similarity(embedding, centroid).flatten()

    labels = np.zeros(len(proteins), dtype=int)
    labels[held_out_idx] = 1

    mask = np.ones(len(proteins), dtype=bool)
    mask[complex_idx] = False
    mask[held_out_idx] = True

    return similarities[mask], labels[mask]


def evaluate_all_methods(
    srf_embedding: np.ndarray,
    node2vec_embedding: np.ndarray,
    adj_sparse: csr_matrix,
    proteins: np.ndarray,
    corum_complexes: dict[str, set[str]],
    min_size: int = 5,
    max_size: int = 50,
) -> pd.DataFrame:
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    valid_complexes = {}
    for name, members in corum_complexes.items():
        present = [m for m in members if m in protein_to_idx]
        if min_size <= len(present) <= max_size:
            valid_complexes[name] = present

    results = []

    cn_matrix = adj_sparse @ adj_sparse

    for complex_name, complex_members in tqdm(
        valid_complexes.items(), desc="Evaluating"
    ):
        complex_idx = [protein_to_idx[m] for m in complex_members]

        for held_out_idx in complex_idx:
            remaining_idx = [idx for idx in complex_idx if idx != held_out_idx]

            srf_scores, labels = evaluate_method_on_complex(
                srf_embedding, complex_idx, held_out_idx, proteins
            )

            n2v_scores, _ = evaluate_method_on_complex(
                node2vec_embedding, complex_idx, held_out_idx, proteins
            )

            cn_scores = np.array(cn_matrix[:, remaining_idx].sum(axis=1)).flatten()
            mask = np.ones(len(proteins), dtype=bool)
            mask[complex_idx] = False
            mask[held_out_idx] = True
            cn_scores = cn_scores[mask]

            if len(np.unique(labels)) == 2:
                srf_auroc = roc_auc_score(labels, srf_scores)
                srf_aupr = average_precision_score(labels, srf_scores)

                n2v_auroc = roc_auc_score(labels, n2v_scores)
                n2v_aupr = average_precision_score(labels, n2v_scores)

                cn_auroc = roc_auc_score(labels, cn_scores)
                cn_aupr = average_precision_score(labels, cn_scores)
            else:
                continue

            results.append(
                {
                    "complex": complex_name,
                    "complex_size": len(complex_idx),
                    "srf_auroc": srf_auroc,
                    "srf_aupr": srf_aupr,
                    "node2vec_auroc": n2v_auroc,
                    "node2vec_aupr": n2v_aupr,
                    "cn_auroc": cn_auroc,
                    "cn_aupr": cn_aupr,
                    "srf_scores": srf_scores,
                    "n2v_scores": n2v_scores,
                    "cn_scores": cn_scores,
                    "labels": labels,
                }
            )

    return pd.DataFrame(results)


def plot_roc_curves(results_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    all_methods = {
        "SRF": ("srf_scores", "srf_auroc", "#2E86AB"),
        "Node2Vec": ("n2v_scores", "node2vec_auroc", "#A23B72"),
        "CN": ("cn_scores", "cn_auroc", "#999999"),
    }

    for method_name, (score_col, auroc_col, color) in all_methods.items():
        all_fpr = []
        all_tpr = []

        for _, row in results_df.iterrows():
            fpr, tpr, _ = roc_curve(row["labels"], row[score_col])
            all_fpr.append(fpr)
            all_tpr.append(tpr)

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fpr, tpr in zip(all_fpr, all_tpr):
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        mean_auroc = results_df[auroc_col].mean()
        std_auroc = results_df[auroc_col].std()

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            lw=2.5,
            label=f"{method_name} (AUC = {mean_auroc:.3f} $\pm$ {std_auroc:.3f})",
        )

        ax.fill_between(
            mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.15
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    ax.set_xlabel("False positive rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=12, fontweight="bold")
    ax.set_title(
        "ROC curves: complex member completion", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(results_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    all_methods = {
        "SRF": ("srf_scores", "srf_aupr", "#2E86AB"),
        "Node2Vec": ("n2v_scores", "node2vec_aupr", "#A23B72"),
        "CN": ("cn_scores", "cn_aupr", "#999999"),
    }

    for method_name, (score_col, aupr_col, color) in all_methods.items():
        all_precision = []
        all_recall = []

        for _, row in results_df.iterrows():
            precision, recall, _ = precision_recall_curve(row["labels"], row[score_col])
            all_precision.append(precision)
            all_recall.append(recall)

        mean_recall = np.linspace(0, 1, 100)
        precisions = []
        for precision, recall in zip(all_precision, all_recall):
            precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))

        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)

        mean_aupr = results_df[aupr_col].mean()
        std_aupr = results_df[aupr_col].std()

        ax.plot(
            mean_recall,
            mean_precision,
            color=color,
            lw=2.5,
            label=f"{method_name} (AP = {mean_aupr:.3f} $\pm$ {std_aupr:.3f})",
        )

        ax.fill_between(
            mean_recall,
            mean_precision - std_precision,
            mean_precision + std_precision,
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title(
        "Precision-recall curves: complex member completion",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplots(results_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data_auroc = [
        results_df["srf_auroc"].values,
        results_df["node2vec_auroc"].values,
        results_df["cn_auroc"].values,
    ]

    data_aupr = [
        results_df["srf_aupr"].values,
        results_df["node2vec_aupr"].values,
        results_df["cn_aupr"].values,
    ]

    colors = ["#2E86AB", "#A23B72", "#999999"]
    labels = ["SRF", "Node2Vec", "CN"]

    bp1 = axes[0].boxplot(
        data_auroc, labels=labels, patch_artist=True, widths=0.6, showfliers=True
    )

    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp1[element], color="black", linewidth=1.2)

    axes[0].set_ylabel("AUROC", fontsize=12, fontweight="bold")
    axes[0].set_title("AUROC distribution", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.5)
    axes[0].set_ylim([0, 1.05])

    bp2 = axes[1].boxplot(
        data_aupr, labels=labels, patch_artist=True, widths=0.6, showfliers=True
    )

    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp2[element], color="black", linewidth=1.2)

    axes[1].set_ylabel("AUPR", fontsize=12, fontweight="bold")
    axes[1].set_title("AUPR distribution", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()


def statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("srf", "node2vec"), ("srf", "cn"), ("node2vec", "cn")]

    stats_results = []

    for metric in ["auroc", "aupr"]:
        for method1, method2 in comparisons:
            data1 = results_df[f"{method1}_{metric}"].values
            data2 = results_df[f"{method2}_{metric}"].values

            t_stat, p_val = stats.wilcoxon(data1, data2, alternative="greater")

            stats_results.append(
                {
                    "metric": metric.upper(),
                    "comparison": f"{method1.upper()} vs {method2.upper()}",
                    "statistic": t_stat,
                    "p_value": p_val,
                    "significant": "Yes" if p_val < 0.05 else "No",
                }
            )

    return pd.DataFrame(stats_results)


def main() -> None:
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    baseline_dir = base_dir / "experiments" / "ppi" / "validation" / "baselines"
    validation_output_dir = (
        base_dir
        / "experiments"
        / "ppi"
        / "validation"
        / "outputs"
        / "complex_completion_comparison"
    )
    validation_output_dir.mkdir(parents=True, exist_ok=True)

    srf_embedding = np.load(output_dir / "embedding.npy")
    node2vec_embedding = np.load(baseline_dir / "node2vec_embedding.npy")

    with open(output_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])

    corum_path = data_dir / "corum_complexes.txt"
    corum_complexes = load_corum(corum_path)

    string_path = data_dir / "STRING_human_min900_v12.csv"
    g, _ = load_network(string_path)
    adj_sparse = build_sparse_adjacency(g, sorted(g.nodes()))

    results_df = evaluate_all_methods(
        srf_embedding,
        node2vec_embedding,
        adj_sparse,
        proteins,
        corum_complexes,
        min_size=5,
        max_size=50,
    )

    results_df_clean = results_df.drop(
        columns=["srf_scores", "n2v_scores", "cn_scores", "labels"]
    )
    results_df_clean.to_csv(validation_output_dir / "results.csv", index=False)

    stats_df = statistical_tests(results_df)
    stats_df.to_csv(validation_output_dir / "statistical_tests.csv", index=False)

    plot_roc_curves(results_df, validation_output_dir)
    plot_pr_curves(results_df, validation_output_dir)
    plot_boxplots(results_df, validation_output_dir)

    print(
        f"SRF AUROC: {results_df['srf_auroc'].mean():.3f} ± {results_df['srf_auroc'].std():.3f}"
    )
    print(
        f"Node2Vec AUROC: {results_df['node2vec_auroc'].mean():.3f} ± {results_df['node2vec_auroc'].std():.3f}"
    )
    print(
        f"CN AUROC: {results_df['cn_auroc'].mean():.3f} ± {results_df['cn_auroc'].std():.3f}"
    )
    print(
        f"SRF AUPR: {results_df['srf_aupr'].mean():.4f} ± {results_df['srf_aupr'].std():.4f}"
    )
    print(
        f"Node2Vec AUPR: {results_df['node2vec_aupr'].mean():.4f} ± {results_df['node2vec_aupr'].std():.4f}"
    )
    print(
        f"CN AUPR: {results_df['cn_aupr'].mean():.4f} ± {results_df['cn_aupr'].std():.4f}"
    )


if __name__ == "__main__":
    main()
