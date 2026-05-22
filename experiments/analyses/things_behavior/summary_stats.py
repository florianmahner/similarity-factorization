"""Calculate summary statistics for THINGS behavior results."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
NOISE_CEILING_PCT = 67.22
NOISE_CEILING_STD_PCT = 1.04


def main():
    stats = {}

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

    stats["triplet_performance"] = {
        "noise_ceiling_pct": NOISE_CEILING_PCT,
        "noise_ceiling_std": NOISE_CEILING_STD_PCT,
    }

    # 66-dimension results
    acc66 = pd.read_csv(HERE / "similarity_48" / "outputs" / "rank66" / "accuracy_comparison.csv")
    stats["accuracy_66d"] = {}
    for model in ["SRF", "VICE", "SPoSE"]:
        data = acc66[acc66["model"] == model]
        corr = data["correlation"].mean()
        acc_mean = data["accuracy"].mean()
        stats["accuracy_66d"][model] = {
            "accuracy_pct": round(acc_mean * 100, 2),
            "accuracy_std": round(data["accuracy"].std() * 100, 2),
            "correlation_mean": round(corr, 4),
            "r_squared_pct": round(corr**2 * 100, 1),
        }

    # Pairwise dimension correlations
    recon66 = pd.read_csv(HERE / "pairwise" / "outputs" / "pairwise_reconstruction.csv")
    mean_corr66 = recon66.groupby("dimension")["correlation"].mean()
    stats["dimension_correlation_66d"] = {
        "mean": round(mean_corr66.mean(), 4),
        "std": round(mean_corr66.std(), 4),
        "min": round(mean_corr66.min(), 4),
        "max": round(mean_corr66.max(), 4),
        "n_above_0.9": int((mean_corr66 > 0.9).sum()),
        "n_total": len(mean_corr66),
    }

    # Dimension reliability
    rel66 = pd.read_csv(HERE / "reliability" / "outputs" / "dimension_reliability.csv")
    stats["dimension_reliability_66d"] = {
        "mean": round(rel66["Reliability"].mean(), 2),
        "std": round(rel66["Reliability"].std(), 2),
        "min": round(rel66["Reliability"].min(), 2),
        "max": round(rel66["Reliability"].max(), 2),
        "n_above_0.9": int((rel66["Reliability"] > 0.9).sum()),
    }

    # Print
    print(json.dumps(stats, indent=2))

    output_path = HERE / "summary_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
