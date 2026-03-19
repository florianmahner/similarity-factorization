"""Compare OOO prediction accuracy with different Laplace smoothing and ranks."""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from pysrf import SRF
from src.colors import ROSE, TEAL
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

N_OBJECTS = 1854


def build_rsm_laplace(triplets, n, alpha=0.0):
    """Build RSM from triplets with Laplace smoothing."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))

    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            if a != b:
                shown[a, b] += 1
                shown[b, a] += 1
        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1

    if alpha > 0:
        S = (counts + alpha) / (shown + 2 * alpha)
    else:
        with np.errstate(invalid='ignore'):
            S = np.divide(counts, shown, out=np.zeros_like(counts), where=shown != 0)

    np.fill_diagonal(S, 1.0)
    return S


def softmax_triplet_choice(w_i: np.ndarray, w_j: np.ndarray, w_k: np.ndarray) -> bool:
    """Check if softmax over similarities predicts the correct pair (i,j)."""
    similarities = np.array([w_i @ w_j, w_i @ w_k, w_j @ w_k])
    probas = np.exp(similarities) / np.sum(np.exp(similarities))
    return np.argmax(probas) == 0


def compute_triplet_prediction_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    """Compute OOO prediction accuracy on triplets."""
    acc = 0
    for i, j, k in triplets:
        acc += softmax_triplet_choice(embedding[i], embedding[j], embedding[k])
    return acc / len(triplets)


def main():
    print("Loading triplets...")
    train_triplets, test_triplets = load_triplets(Path('data/things'), number='4.7mio')

    print(f"Train triplets: {len(train_triplets):,}")
    print(f"Test triplets: {len(test_triplets):,}")

    # Test different alpha values and ranks
    alphas = [0.0, 1.0]
    ranks = [30, 40, 50]

    results = []

    for alpha in alphas:
        print(f"\nBuilding RSM with α={alpha}...")
        S = build_rsm_laplace(train_triplets, N_OBJECTS, alpha=alpha)

        for rank in ranks:
            print(f"  Fitting SRF (rank={rank})...")
            model = SRF(rank=rank, random_state=42)
            embedding = model.fit_transform(S)

            # Compute accuracy on train and test
            print(f"  Computing prediction accuracy...")
            train_acc = compute_triplet_prediction_accuracy(embedding, train_triplets)
            test_acc = compute_triplet_prediction_accuracy(embedding, test_triplets)

            results.append({
                'alpha': alpha,
                'rank': rank,
                'train_acc': train_acc,
                'test_acc': test_acc,
            })

            print(f"    Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    # Print summary table
    print("\n" + "="*70)
    print("OOO PREDICTION ACCURACY")
    print("="*70)
    print(f"{'Alpha':<10} {'Rank':<10} {'Train Acc':<15} {'Test Acc':<15}")
    print("-"*50)
    for r in results:
        print(f"{r['alpha']:<10} {r['rank']:<10} {r['train_acc']:<15.4f} {r['test_acc']:<15.4f}")

    # =========================================================================
    # Plot: Test accuracy comparison
    # =========================================================================
    fig, ax = create_figure("single")

    # Group by alpha
    x = np.arange(len(ranks))
    width = 0.35

    alpha0_test = [r['test_acc'] for r in results if r['alpha'] == 0.0]
    alpha1_test = [r['test_acc'] for r in results if r['alpha'] == 1.0]

    bars1 = ax.bar(x - width/2, alpha0_test, width, label='α=0 (no smoothing)', color=ROSE, alpha=0.8)
    bars2 = ax.bar(x + width/2, alpha1_test, width, label='α=1 (Laplace)', color=TEAL, alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Test OOO accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.set_ylim(0.5, 0.75)
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "ooo_accuracy_by_alpha.pdf")
    print(f"\nSaved: ooo_accuracy_by_alpha.pdf")

    # =========================================================================
    # Plot: Train vs Test accuracy
    # =========================================================================
    fig, ax = create_figure("wide")

    # Focus on alpha=1.0 results
    alpha1_results = [r for r in results if r['alpha'] == 1.0]
    ranks_a1 = [r['rank'] for r in alpha1_results]
    train_a1 = [r['train_acc'] for r in alpha1_results]
    test_a1 = [r['test_acc'] for r in alpha1_results]

    x = np.arange(len(ranks_a1))
    width = 0.35

    ax.bar(x - width/2, train_a1, width, label='Train', color=TEAL, alpha=0.6)
    ax.bar(x + width/2, test_a1, width, label='Test', color=TEAL, alpha=1.0)

    ax.set_xlabel("Rank")
    ax.set_ylabel("OOO accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks_a1)
    ax.legend()
    ax.set_ylim(0.5, 0.75)
    ax.set_title("α=1.0 (Laplace smoothing)")
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "train_vs_test_accuracy.pdf")
    print(f"Saved: train_vs_test_accuracy.pdf")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "ooo_results.csv", index=False)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
