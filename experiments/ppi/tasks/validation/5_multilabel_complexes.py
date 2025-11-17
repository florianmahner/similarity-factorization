import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.preprocessing import normalize
from analyses.ppi.corum import load_corum


def create_multilabel_targets(
    proteins: np.ndarray,
    corum_complexes: dict,
    min_complex_size: int = 5,
    max_complex_size: int = 200,
) -> tuple[np.ndarray, list, np.ndarray]:
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    valid_complexes = []
    for name, members in corum_complexes.items():
        present_members = [m for m in members if m in protein_to_idx]
        if min_complex_size <= len(present_members) <= max_complex_size:
            valid_complexes.append((name, present_members))

    n_complexes = len(valid_complexes)
    n_proteins = len(proteins)

    labels = np.zeros((n_proteins, n_complexes), dtype=int)

    for complex_idx, (name, members) in enumerate(valid_complexes):
        for protein in members:
            protein_idx = protein_to_idx[protein]
            labels[protein_idx, complex_idx] = 1

    complexes_with_members = labels.sum(axis=1) > 0

    return (
        labels[complexes_with_members],
        [name for name, _ in valid_complexes],
        complexes_with_members,
    )


def evaluate_multilabel_prediction(embedding: np.ndarray, labels: np.ndarray) -> dict:
    embedding_norm = normalize(embedding, axis=1)

    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42))

    y_pred = cross_val_predict(clf, embedding_norm, labels, cv=5, n_jobs=-1)

    micro_f1 = f1_score(labels, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
    samples_f1 = f1_score(labels, y_pred, average="samples", zero_division=0)

    micro_precision = precision_score(labels, y_pred, average="micro", zero_division=0)
    macro_precision = precision_score(labels, y_pred, average="macro", zero_division=0)

    micro_recall = recall_score(labels, y_pred, average="micro", zero_division=0)
    macro_recall = recall_score(labels, y_pred, average="macro", zero_division=0)

    hamming = hamming_loss(labels, y_pred)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "samples_f1": samples_f1,
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "hamming_loss": hamming,
    }


def analyze_multi_membership(
    labels: np.ndarray, proteins_with_complex: np.ndarray, complex_names: list
) -> pd.DataFrame:
    n_memberships = labels.sum(axis=1)

    results = {
        "total_proteins": len(labels),
        "proteins_in_1_complex": (n_memberships == 1).sum(),
        "proteins_in_2_complexes": (n_memberships == 2).sum(),
        "proteins_in_3_complexes": (n_memberships == 3).sum(),
        "proteins_in_4plus_complexes": (n_memberships >= 4).sum(),
        "mean_memberships": n_memberships.mean(),
        "max_memberships": n_memberships.max(),
    }

    return pd.DataFrame([results])


def plot_membership_distribution(labels: np.ndarray, output_dir: Path):
    n_memberships = labels.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))

    membership_counts = np.bincount(n_memberships.astype(int))
    x = np.arange(len(membership_counts))

    ax.bar(x, membership_counts, alpha=0.7, color="#2E86AB")
    ax.set_xlabel("Number of complex memberships")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Distribution of multi-complex memberships")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "membership_distribution.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_performance_comparison(results: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_f1 = {
        "Micro": results["micro_f1"],
        "Macro": results["macro_f1"],
        "Samples": results["samples_f1"],
    }

    ax = axes[0]
    ax.bar(metrics_f1.keys(), metrics_f1.values(), alpha=0.7, color="#2E86AB")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 scores")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    metrics_pr = {
        "Precision (Micro)": results["micro_precision"],
        "Recall (Micro)": results["micro_recall"],
    }

    ax = axes[1]
    ax.bar(metrics_pr.keys(), metrics_pr.values(), alpha=0.7, color="#A23B72")
    ax.set_ylabel("Score")
    ax.set_title("Precision and recall")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(["Hamming Loss"], [results["hamming_loss"]], alpha=0.7, color="#F18F01")
    ax.set_ylabel("Loss")
    ax.set_title("Hamming loss (lower is better)")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multilabel_performance.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    validation_output_dir = (
        base_dir / "experiments" / "ppi" / "validation" / "outputs" / "multilabel"
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

    print("\nCreating multilabel targets...")
    labels, complex_names, proteins_with_complex = create_multilabel_targets(
        proteins, corum_complexes, min_complex_size=5, max_complex_size=50
    )
    print(f"Proteins with complex membership: {proteins_with_complex.sum()}")
    print(f"Number of complexes: {len(complex_names)}")
    print(f"Label matrix shape: {labels.shape}")

    print("\nAnalyzing multi-membership statistics...")
    membership_stats = analyze_multi_membership(
        labels, proteins_with_complex, complex_names
    )
    print(membership_stats.to_string(index=False))
    membership_stats.to_csv(validation_output_dir / "membership_stats.csv", index=False)

    print("\nPlotting membership distribution...")
    plot_membership_distribution(labels, validation_output_dir)

    print("\nEvaluating multilabel prediction with SRF embedding...")
    embedding_subset = embedding[proteins_with_complex]
    results = evaluate_multilabel_prediction(embedding_subset, labels)

    results_df = pd.DataFrame([results])
    results_df.to_csv(validation_output_dir / "results.csv", index=False)

    print("\n" + "=" * 80)
    print("MULTILABEL COMPLEX MEMBERSHIP PREDICTION")
    print("=" * 80)
    print("\nMembership Statistics:")
    print(membership_stats.to_string(index=False))
    print("\nPrediction Performance:")
    print(results_df.to_string(index=False))

    print("\nPlotting performance metrics...")
    plot_performance_comparison(results, validation_output_dir)

    print(f"\nResults saved to: {validation_output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
