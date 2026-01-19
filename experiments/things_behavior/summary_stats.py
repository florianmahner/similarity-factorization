"""Calculate summary statistics for THINGS behavior results."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/things_behavior"


def main():
    stats = {}

    # Dataset info
    n_objects = 1854
    total_possible_triplets = math.comb(n_objects, 3)
    actual_triplets = 4_120_663
    hebart_triplets = 4_700_000

    stats["dataset"] = {
        "n_objects": n_objects,
        "total_possible_triplets": total_possible_triplets,
        "actual_triplets": actual_triplets,
        "hebart_triplets": hebart_triplets,
        "sampling_pct_actual": round(actual_triplets / total_possible_triplets * 100, 2),
        "sampling_pct_hebart": round(hebart_triplets / total_possible_triplets * 100, 2),
    }

    # Cross-validation dimensionality
    cv_summary = pd.read_csv(
        PROJECT_ROOT / "archive/2025-Q3-Q4/things_old/dimension_estimation/dimension_estimation_summary_trainset.csv"
    )
    stats["cross_validation"] = {
        "optimal_rank": int(cv_summary["best_rank"].iloc[0]),
        "best_rmse": float(cv_summary["best_rmse"].iloc[0]),
    }

    # 66-dimension results
    acc66 = pd.read_csv(OUTPUT_DIR / "data/66/accuracy_comparison.csv")
    stats["accuracy_66d"] = {}
    for model in ["SRF", "VICE", "SPoSE"]:
        data = acc66[acc66["model"] == model]
        corr = data["correlation"].mean()
        stats["accuracy_66d"][model] = {
            "accuracy_mean": round(data["accuracy"].mean(), 4),
            "accuracy_std": round(data["accuracy"].std(), 4),
            "accuracy_pct": round(data["accuracy"].mean() * 100, 2),
            "correlation_mean": round(corr, 4),
            "correlation_std": round(data["correlation"].std(), 4),
            "r_squared": round(corr**2, 4),
            "r_squared_pct": round(corr**2 * 100, 1),
        }

    # 49-dimension results
    acc49 = pd.read_csv(OUTPUT_DIR / "data/49/accuracy_comparison.csv")
    stats["accuracy_49d"] = {}
    for model in ["ADMM", "SPoSE"]:
        data = acc49[acc49["model"] == model]
        corr = data["correlation"].mean()
        stats["accuracy_49d"][model] = {
            "accuracy_mean": round(data["accuracy"].mean(), 4),
            "accuracy_std": round(data["accuracy"].std(), 4),
            "accuracy_pct": round(data["accuracy"].mean() * 100, 2),
            "correlation_mean": round(corr, 4),
            "correlation_std": round(data["correlation"].std(), 4),
        }

    # Noise ceiling for triplet accuracy (from Hebart et al. 2020, Nature Human Behaviour)
    noise_ceiling_pct = 67.22
    noise_ceiling_std = 1.04

    stats["triplet_performance"] = {
        "noise_ceiling_pct": noise_ceiling_pct,
        "noise_ceiling_std": noise_ceiling_std,
    }

    for model in ["SRF", "VICE", "SPoSE"]:
        acc_mean = stats["accuracy_66d"][model]["accuracy_mean"] * 100
        acc_std = stats["accuracy_66d"][model]["accuracy_std"] * 100
        pct_ceiling = (acc_mean / noise_ceiling_pct) * 100
        # Error propagation
        pct_ceiling_std = pct_ceiling * np.sqrt(
            (acc_std / acc_mean) ** 2 + (noise_ceiling_std / noise_ceiling_pct) ** 2
        ) if acc_std > 0 else pct_ceiling * (noise_ceiling_std / noise_ceiling_pct)

        stats["triplet_performance"][model] = {
            "accuracy_pct": round(acc_mean, 2),
            "accuracy_std": round(acc_std, 2),
            "pct_of_ceiling": round(pct_ceiling, 2),
            "pct_of_ceiling_std": round(pct_ceiling_std, 2),
        }

    # RSM correlation for 48 held-out objects
    srf_corr = stats["accuracy_66d"]["SRF"]["correlation_mean"]
    stats["rsm_48_correlation"] = {
        "srf_r": round(srf_corr, 4),
        "srf_r_squared_pct": round(srf_corr**2 * 100, 1),
    }

    # Dimension correlations with SPoSE (66d)
    recon66 = pd.read_csv(OUTPUT_DIR / "data/66/pairwise_reconstruction.csv")
    mean_corr66 = recon66.groupby("dimension")["correlation"].mean()
    stats["dimension_correlation_66d"] = {
        "mean": round(mean_corr66.mean(), 4),
        "std": round(mean_corr66.std(), 4),
        "min": round(mean_corr66.min(), 4),
        "max": round(mean_corr66.max(), 4),
        "n_above_0.9": int((mean_corr66 > 0.9).sum()),
        "n_above_0.95": int((mean_corr66 > 0.95).sum()),
        "n_total": len(mean_corr66),
    }

    # Dimension correlations with SPoSE (49d)
    recon49 = pd.read_csv(OUTPUT_DIR / "data/49/pairwise_reconstruction.csv")
    mean_corr49 = recon49.groupby("dimension")["correlation"].mean()
    stats["dimension_correlation_49d"] = {
        "mean": round(mean_corr49.mean(), 4),
        "std": round(mean_corr49.std(), 4),
        "min": round(mean_corr49.min(), 4),
        "max": round(mean_corr49.max(), 4),
        "n_above_0.9": int((mean_corr49 > 0.9).sum()),
        "n_above_0.95": int((mean_corr49 > 0.95).sum()),
        "n_total": len(mean_corr49),
    }

    # Dimension reliability (split-half)
    rel66 = pd.read_csv(OUTPUT_DIR / "data/66/dimension_reliability.csv")
    max_rel = rel66["Reliability"].max()
    stats["dimension_reliability_66d"] = {
        "mean": round(rel66["Reliability"].mean(), 2),
        "std": round(rel66["Reliability"].std(), 2),
        "min": round(rel66["Reliability"].min(), 2),
        "max": math.floor(max_rel * 100) / 100,  # truncate to avoid 0.9954 -> 1.00
        "n_above_0.8": int((rel66["Reliability"] > 0.8).sum()),
        "n_above_0.9": int((rel66["Reliability"] > 0.9).sum()),
    }

    rel49 = pd.read_csv(OUTPUT_DIR / "data/49/dimension_reliability.csv")
    stats["dimension_reliability_49d"] = {
        "mean": round(rel49["Reliability"].mean(), 4),
        "std": round(rel49["Reliability"].std(), 4),
        "min": round(rel49["Reliability"].min(), 4),
        "max": round(rel49["Reliability"].max(), 4),
        "n_above_0.8": int((rel49["Reliability"] > 0.8).sum()),
        "n_above_0.9": int((rel49["Reliability"] > 0.9).sum()),
    }

    # Low data results
    low66 = pd.read_csv(OUTPUT_DIR / "data/66/low_data.csv")
    stats["low_data_66d"] = {}
    for pct in [0.05, 0.1, 0.2, 0.5, 1.0]:
        data = low66[low66["data_percentage"] == pct]
        n_triplets = int(data["n_triplets"].iloc[0])
        stats["low_data_66d"][f"{int(pct*100)}%"] = {
            "n_triplets": n_triplets,
            "n_triplets_formatted": f"{n_triplets:,}",
            "pct_of_possible": round(n_triplets / total_possible_triplets * 100, 4),
            "accuracy_mean": round(data["accuracy"].mean(), 4),
            "accuracy_std": round(data["accuracy"].std(), 4),
            "accuracy_pct": round(data["accuracy"].mean() * 100, 1),
        }

    # Performance relative to full data
    full_acc = stats["low_data_66d"]["100%"]["accuracy_mean"]
    for key in stats["low_data_66d"]:
        acc = stats["low_data_66d"][key]["accuracy_mean"]
        stats["low_data_66d"][key]["pct_of_full"] = round(acc / full_acc * 100, 1)

    # Save to JSON
    output_path = OUTPUT_DIR / "summary_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {output_path}")

    # Print formatted for paper
    print("\n" + "=" * 60)
    print("VALUES FOR PAPER")
    print("=" * 60)

    print(f"\nDataset: {stats['dataset']['hebart_triplets']:,} triplets, "
          f"{stats['dataset']['n_objects']} objects, "
          f"{stats['dataset']['sampling_pct_hebart']}% sampled")

    print(f"\nOptimal dimensionality (CV): {stats['cross_validation']['optimal_rank']} dimensions")

    print(f"\nNoise ceiling: {stats['triplet_performance']['noise_ceiling_pct']}% "
          f"(±{stats['triplet_performance']['noise_ceiling_std']}%)")
    print("\nTriplet prediction accuracy:")
    for model in ["SRF", "VICE", "SPoSE"]:
        m = stats["triplet_performance"][model]
        print(f"  {model}: {m['accuracy_pct']:.2f}% (±{m['accuracy_std']:.2f}%), "
              f"{m['pct_of_ceiling']:.2f}% (±{m['pct_of_ceiling_std']:.2f}%) of ceiling")

    print(f"\nRSM correlation (48 held-out): r = {stats['rsm_48_correlation']['srf_r']}, "
          f"r² = {stats['rsm_48_correlation']['srf_r_squared_pct']}%")

    print(f"\nDimension correlation with SPoSE 66d: "
          f"mean r = {stats['dimension_correlation_66d']['mean']}, "
          f"range {stats['dimension_correlation_66d']['min']}-{stats['dimension_correlation_66d']['max']}, "
          f"{stats['dimension_correlation_66d']['n_above_0.9']}/{stats['dimension_correlation_66d']['n_total']} dims > 0.9")

    print(f"\nSplit-half reliability (66d): "
          f"mean = {stats['dimension_reliability_66d']['mean']}, "
          f"range {stats['dimension_reliability_66d']['min']}-{stats['dimension_reliability_66d']['max']}")

    print("\nAccuracy comparison (66d):")
    for model in ["SRF", "VICE", "SPoSE"]:
        s = stats["accuracy_66d"][model]
        print(f"  {model}: {s['accuracy_pct']}% (r={s['correlation_mean']:.3f})")

    print("\nLow data results:")
    for key in ["5%", "10%", "20%", "50%", "100%"]:
        s = stats["low_data_66d"][key]
        print(f"  {key}: {s['accuracy_pct']}% accuracy, "
              f"{s['n_triplets_formatted']} triplets ({s['pct_of_possible']}% of possible), "
              f"{s['pct_of_full']}% of full-data performance")


if __name__ == "__main__":
    main()
