"""Quick sanity check for link prediction methods on C. elegans."""

from pathlib import Path

from src.utils import get_output_dir
import numpy as np
import json
from sklearn.metrics import roc_auc_score

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.ppi.utils import load_fold_data, evaluate_open_world, FILENAME_MAP
from experiments.ppi.models import get_predictor

DATA_DIR = Path(__file__).parents[2] / "data" / "ppi"
OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET = "c_elegans"
METHODS = ["srf", "deepwalk", "node2vec", "aa", "cn"]
SEED = 42
RANK = 50
N_FOLDS = 3  # Quick test with 3 folds

def main():
    split_dir = DATA_DIR / "splits" / FILENAME_MAP[DATASET]

    results = []

    for method in METHODS:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()}")
        print(f"{'='*50}")

        aurocs = []
        for fold in range(N_FOLDS):
            print(f"\n--- Fold {fold} ---")

            data = load_fold_data(split_dir, DATASET, fold)
            train_edges = data["train_edges"]
            test_pos = data["test_pos_edges"]
            n_nodes = data["n_nodes"]

            print(f"  Train edges: {len(train_edges)}")
            print(f"  Test pos edges: {len(test_pos)}")
            print(f"  N nodes: {n_nodes}")

            # Create config dict
            cfg = {"rank": RANK}

            model = get_predictor(method, SEED + fold, cfg)
            print(f"  Fitting {method}...")
            model.fit(train_edges, n_nodes)

            print(f"  Predicting...")
            score_matrix = model.predict_all()

            # Check score matrix properties
            print(f"  Score matrix shape: {score_matrix.shape}")
            print(f"  Score matrix range: [{score_matrix.min():.4f}, {score_matrix.max():.4f}]")
            print(f"  Score matrix mean: {score_matrix.mean():.4f}")
            print(f"  Non-zero entries: {(score_matrix != 0).sum()}")

            # Evaluate
            metrics = evaluate_open_world(score_matrix, train_edges, test_pos)
            aurocs.append(metrics["auroc"])

            print(f"  AUROC: {metrics['auroc']:.4f}")
            print(f"  AUPRC: {metrics['auprc']:.6f}")

        avg_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        print(f"\n{method.upper()} Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")

        results.append({
            "method": method,
            "auroc_mean": avg_auroc,
            "auroc_std": std_auroc,
            "aurocs": aurocs
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in sorted(results, key=lambda x: -x["auroc_mean"]):
        print(f"{r['method']:12s}: {r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}")

    # Save results
    with open(OUTPUT_DIR / "debug_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'debug_results.json'}")


if __name__ == "__main__":
    main()
