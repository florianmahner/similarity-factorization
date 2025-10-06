#!/usr/bin/env python3
"""
Plotting script for rank detection experiment results

Usage:
    python plot_rank_detection_results.py

Loads CSV files and creates comprehensive plots showing the relationship
between RSM size and optimal training ratio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import argparse


def load_results(data_dir="."):
    """Load experiment results from CSV files"""
    data_dir = Path(data_dir)

    trial_file = data_dir / "rank_detection_trial_results.csv"
    summary_file = data_dir / "rank_detection_summary.csv"

    if not trial_file.exists():
        raise FileNotFoundError(f"Trial results not found: {trial_file}")
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary results not found: {summary_file}")

    trial_df = pd.read_csv(trial_file)
    summary_df = pd.read_csv(summary_file)

    print(f"📊 Loaded {len(trial_df)} individual trials")
    print(f"📈 Loaded {len(summary_df)} condition summaries")

    return trial_df, summary_df


def create_main_plots(summary_df, save_path=None):
    """Create the main visualization plots"""

    plt.style.use("default")
    fig = plt.figure(figsize=(20, 12))

    # Create a 2x3 grid of subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Get unique object counts for analysis
    n_objects_list = sorted(summary_df["n_objects"].unique())

    # 1. Heatmap of accuracy across all conditions
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_data = summary_df.pivot(
        index="n_objects", columns="train_ratio", values="accuracy_mean"
    )
    sns.heatmap(
        pivot_data, annot=False, cmap="viridis", ax=ax1, cbar_kws={"label": "Accuracy"}
    )
    ax1.set_title("Rank Detection Accuracy\n(Heatmap)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Training Ratio")
    ax1.set_ylabel("Number of Objects")

    # 2. Line plot showing optimal ratio for each object count
    ax2 = fig.add_subplot(gs[0, 1])
    optimal_ratios = []
    object_counts = []

    for n_obj in n_objects_list:
        subset = summary_df[summary_df["n_objects"] == n_obj]
        best_ratio = subset.loc[subset["accuracy_mean"].idxmax(), "train_ratio"]
        optimal_ratios.append(best_ratio)
        object_counts.append(n_obj)

    ax2.plot(
        object_counts, optimal_ratios, "o-", linewidth=2, markersize=8, color="red"
    )
    ax2.set_xlabel("Number of Objects")
    ax2.set_ylabel("Optimal Training Ratio")
    ax2.set_title(
        "Optimal Training Ratio vs\nNumber of Objects", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(object_counts, optimal_ratios, 1)
    p = np.poly1d(z)
    ax2.plot(
        object_counts,
        p(object_counts),
        "--",
        alpha=0.8,
        color="blue",
        label=f"Linear fit: slope={z[0]:.4f}",
    )
    ax2.legend()

    # 3. Maximum accuracy achievable for each object count
    ax3 = fig.add_subplot(gs[0, 2])
    max_accuracies = []
    for n_obj in n_objects_list:
        subset = summary_df[summary_df["n_objects"] == n_obj]
        max_acc = subset["accuracy_mean"].max()
        max_accuracies.append(max_acc)

    ax3.plot(
        object_counts, max_accuracies, "s-", linewidth=2, markersize=8, color="green"
    )
    ax3.set_xlabel("Number of Objects")
    ax3.set_ylabel("Maximum Accuracy")
    ax3.set_title(
        "Peak Rank Detection Accuracy\nvs Number of Objects",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. Individual curves for different object counts
    ax4 = fig.add_subplot(gs[1, :])
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_objects_list)))

    for i, n_obj in enumerate(n_objects_list):
        subset = summary_df[summary_df["n_objects"] == n_obj]
        subset_sorted = subset.sort_values("train_ratio")
        ax4.plot(
            subset_sorted["train_ratio"],
            subset_sorted["accuracy_mean"],
            "o-",
            label=f"n={n_obj}",
            linewidth=2,
            markersize=4,
            color=colors[i],
        )

        # Add error bars
        ax4.fill_between(
            subset_sorted["train_ratio"],
            subset_sorted["accuracy_ci_lower"],
            subset_sorted["accuracy_ci_upper"],
            alpha=0.2,
            color=colors[i],
        )

    ax4.set_xlabel("Training Ratio")
    ax4.set_ylabel("Rank Detection Accuracy")
    ax4.set_title(
        "Rank Detection Accuracy vs Training Ratio for Different Object Counts",
        fontsize=14,
        fontweight="bold",
    )
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.1, 0.8])
    ax4.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📁 Main plots saved to: {save_path}")

    plt.show()

    return object_counts, optimal_ratios, max_accuracies


def analyze_relationships(object_counts, optimal_ratios):
    """Analyze the relationship between object count and optimal ratio"""

    # Fit polynomial models of different degrees
    X = np.array(object_counts).reshape(-1, 1)
    y = np.array(optimal_ratios)

    models = {}
    degrees = [1, 2, 3]

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        models[f"poly_{degree}"] = {
            "model": model,
            "poly_features": poly_features,
            "r2": r2,
        }

    # Power law fit: optimal_ratio = a * n_objects^b
    log_objects = np.log(object_counts)
    log_ratios = np.log(optimal_ratios)

    slope, intercept = np.polyfit(log_objects, log_ratios, 1)
    a = np.exp(intercept)
    b = slope

    power_law_pred = a * np.array(object_counts) ** b
    power_law_r2 = r2_score(optimal_ratios, power_law_pred)

    # Statistical analysis
    pearson_corr, pearson_p = pearsonr(object_counts, optimal_ratios)
    spearman_corr, spearman_p = spearmanr(object_counts, optimal_ratios)

    return models, power_law_r2, a, b, pearson_corr, pearson_p


def print_analysis_results(
    models,
    power_law_r2,
    a,
    b,
    pearson_corr,
    pearson_p,
    object_counts,
    optimal_ratios,
    max_accuracies,
):
    """Print comprehensive analysis results"""

    print(f"\n📈 MODEL COMPARISON:")
    print("-" * 50)
    print(f"Linear fit (degree 1):     R² = {models['poly_1']['r2']:.4f}")
    print(f"Quadratic fit (degree 2):  R² = {models['poly_2']['r2']:.4f}")
    print(f"Cubic fit (degree 3):      R² = {models['poly_3']['r2']:.4f}")
    print(f"Power law fit:             R² = {power_law_r2:.4f}")
    print(f"Power law equation: optimal_ratio = {a:.4f} * n_objects^{b:.4f}")

    print(f"\n📊 STATISTICAL ANALYSIS:")
    print("-" * 50)
    print(f"Pearson correlation: r={pearson_corr:.4f}, p={pearson_p:.4f}")

    if pearson_p < 0.05:
        print("✓ Significant linear relationship detected!")
    else:
        print("⚠ No significant linear relationship found.")

    # Determine best model
    best_poly_r2 = max([models[k]["r2"] for k in models.keys()])

    print(f"\n🎯 FINAL CONCLUSIONS:")
    print("-" * 50)

    if power_law_r2 > 0.5:
        print(f"✓ POWER LAW RELATIONSHIP CONFIRMED!")
        print(f"  Formula: optimal_ratio = {a:.4f} * n_objects^{b:.4f}")
        print(f"  R² = {power_law_r2:.3f}")
        if b < 0:
            print(
                f"  ✓ FEWER SAMPLES NEEDED as RSM size increases (exponent = {b:.3f})"
            )
        else:
            print(f"  ⚠ More samples needed as RSM size increases (exponent = {b:.3f})")
    elif best_poly_r2 > 0.5:
        best_poly = max(models.keys(), key=lambda k: models[k]["r2"])
        print(f"✓ POLYNOMIAL RELATIONSHIP CONFIRMED!")
        print(f"  Best fit: {best_poly} with R² = {models[best_poly]['r2']:.3f}")
    else:
        print(f"⚠ No strong relationship found (all R² < 0.5)")

    # Sample efficiency analysis
    print(f"\n💡 SAMPLE EFFICIENCY ANALYSIS:")
    print("-" * 50)
    for i, (n_obj, opt_ratio) in enumerate(zip(object_counts, optimal_ratios)):
        n_samples = int(n_obj * (n_obj - 1) / 2 * opt_ratio)
        print(
            f"n={n_obj:3d}: optimal_ratio={opt_ratio:.3f} → {n_samples:4d} training samples"
        )

    # Accuracy trend
    acc_slope, _ = np.polyfit(object_counts, max_accuracies, 1)
    print(f"\n📈 ACCURACY TRENDS:")
    print("-" * 50)
    print(f"Accuracy trend: slope = {acc_slope:.4f} per additional object")
    if acc_slope > 0:
        print("✓ Larger RSMs achieve better peak rank detection accuracy")
    else:
        print("⚠ No clear improvement in accuracy with larger RSMs")


def main():
    """Main plotting function"""
    parser = argparse.ArgumentParser(
        description="Plot rank detection experiment results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing CSV result files",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        help="Base filename to save plots (e.g., 'results.png')",
    )

    args = parser.parse_args()

    # Load results
    print("📂 Loading experiment results...")
    trial_df, summary_df = load_results(args.data_dir)

    # Create main plots
    print("\n🎨 Creating visualization...")
    object_counts, optimal_ratios, max_accuracies = create_main_plots(
        summary_df, save_path=args.save_plots
    )

    # Analyze relationships
    print("\n🔍 Analyzing relationships...")
    models, power_law_r2, a, b, pearson_corr, pearson_p = analyze_relationships(
        object_counts, optimal_ratios
    )

    # Print comprehensive analysis
    print_analysis_results(
        models,
        power_law_r2,
        a,
        b,
        pearson_corr,
        pearson_p,
        object_counts,
        optimal_ratios,
        max_accuracies,
    )

    print(f"\n🔬 Analysis complete!")


if __name__ == "__main__":
    main()
