#!/usr/bin/env python3
"""
Ablation study on triplet-to-RSM construction with the 1.47M dataset.
Analyze NaN patterns, coverage, and performance vs data amount.
"""
import argparse
import logging
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    acc = 0
    for i, j, k in triplets:
        sims = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
        probas = np.exp(sims - sims.max())
        probas /= probas.sum()
        acc += np.argmax(probas) == 0
    return acc / len(triplets)


def build_rsm_with_stats(n: int, triplets: np.ndarray):
    """Build RSM and return detailed statistics."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))

    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    # RSM with NaN for unobserved pairs
    rsm = np.divide(counts, shown, out=np.nan * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm, 1.0)

    return rsm, counts, shown


def analyze_coverage(n: int, shown: np.ndarray):
    """Analyze pair coverage statistics."""
    n_pairs = n * (n - 1) // 2
    triu_idx = np.triu_indices(n, k=1)
    shown_triu = shown[triu_idx]

    stats = {
        "n_items": n,
        "n_possible_pairs": n_pairs,
        "n_observed_pairs": (shown_triu > 0).sum(),
        "n_unobserved_pairs": (shown_triu == 0).sum(),
        "coverage_pct": 100 * (shown_triu > 0).sum() / n_pairs,
        "mean_observations": shown_triu[shown_triu > 0].mean() if (shown_triu > 0).any() else 0,
        "median_observations": np.median(shown_triu[shown_triu > 0]) if (shown_triu > 0).any() else 0,
        "max_observations": shown_triu.max(),
        "min_observations_nonzero": shown_triu[shown_triu > 0].min() if (shown_triu > 0).any() else 0,
    }
    return stats, shown_triu


def plot_coverage_histogram(shown_triu: np.ndarray, title: str, filename: str):
    """Plot histogram of observation counts per pair."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # All pairs (including zeros)
    ax = axes[0]
    ax.hist(shown_triu, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Observations per pair")
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"{title}\n(including unobserved)")
    ax.axvline(x=0.5, color='red', linestyle='--', label=f"Unobserved: {(shown_triu == 0).sum():,}")
    ax.legend()

    # Only observed pairs
    ax = axes[1]
    observed = shown_triu[shown_triu > 0]
    ax.hist(observed, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Observations per pair")
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"{title}\n(observed pairs only)")
    ax.axvline(x=observed.mean(), color='red', linestyle='--', label=f"Mean: {observed.mean():.1f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {filename}")


def plot_rsm_nan_heatmap(rsm: np.ndarray, filename: str):
    """Visualize NaN pattern in RSM."""
    fig, ax = plt.subplots(figsize=(8, 8))
    nan_mask = np.isnan(rsm).astype(float)
    im = ax.imshow(nan_mask, cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel("Item index")
    ax.set_ylabel("Item index")
    ax.set_title(f"RSM NaN pattern\n(red = NaN, green = observed)")
    plt.colorbar(im, ax=ax, label="NaN")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {filename}")


def plot_ablation_curve(results_df: pd.DataFrame, filename: str):
    """Plot performance vs data fraction."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Accuracy vs data fraction
    ax = axes[0]
    for model in results_df["model"].unique():
        subset = results_df[results_df["model"] == model]
        ax.plot(subset["fraction"], subset["val_accuracy"], marker='o', label=model)
    ax.set_xlabel("Data fraction")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs Data Amount")
    ax.axhline(y=0.6667, color='gray', linestyle='--', label="Noise ceiling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Coverage vs data fraction
    ax = axes[1]
    ax.plot(results_df[results_df["model"] == "SRF"]["fraction"],
            results_df[results_df["model"] == "SRF"]["coverage_pct"], marker='o', color='blue')
    ax.set_xlabel("Data fraction")
    ax.set_ylabel("Pair coverage (%)")
    ax.set_title("RSM Coverage vs Data Amount")
    ax.grid(True, alpha=0.3)

    # NaN percentage vs data fraction
    ax = axes[2]
    ax.plot(results_df[results_df["model"] == "SRF"]["fraction"],
            100 - results_df[results_df["model"] == "SRF"]["coverage_pct"], marker='o', color='red')
    ax.set_xlabel("Data fraction")
    ax.set_ylabel("NaN pairs (%)")
    ax.set_title("Missing Pairs vs Data Amount")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {filename}")


def plot_observation_vs_accuracy(triplets: np.ndarray, embedding: np.ndarray, shown: np.ndarray, filename: str):
    """Plot triplet accuracy binned by minimum observation count."""
    # Compute accuracy per observation bin
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 100)]
    bin_results = []

    for lo, hi in bins:
        subset_idx = []
        for idx, (i, j, k) in enumerate(triplets):
            min_obs = min(shown[i, j], shown[i, k], shown[j, k])
            if lo <= min_obs < hi:
                subset_idx.append(idx)

        if subset_idx:
            subset_triplets = triplets[subset_idx]
            acc = compute_triplet_accuracy(embedding, subset_triplets)
            bin_results.append({
                "bin": f"[{lo}, {hi})",
                "n_triplets": len(subset_idx),
                "accuracy": acc,
                "bin_mid": (lo + hi) / 2,
            })

    df = pd.DataFrame(bin_results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(df["bin"], df["accuracy"], alpha=0.7, edgecolor='black')
    ax.set_xlabel("Min observations per pair in triplet")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Observation Count")
    ax.axhline(y=0.6667, color='gray', linestyle='--', label="Noise ceiling")
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    ax.bar(df["bin"], df["n_triplets"], alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel("Min observations per pair in triplet")
    ax.set_ylabel("Number of triplets")
    ax.set_title("Triplet Distribution by Observation Count")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="1.47mio", choices=["1.47mio", "4.7mio"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=66)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    log.info(f"Loading {args.dataset} dataset...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")

    if args.dataset == "1.47mio":
        triplet_dir = data_dir / "triplets_147"
    else:
        triplet_dir = data_dir / "triplets_47"

    train_triplets = np.loadtxt(triplet_dir / "trainset.txt").astype(int)
    val_triplets = np.loadtxt(triplet_dir / "validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    log.info(f"Train triplets: {len(train_triplets):,}")
    log.info(f"Val triplets: {len(val_triplets):,}")

    # =========================================================================
    # PART 1: Analyze full dataset coverage
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("PART 1: Full Dataset Coverage Analysis")
    log.info("="*60)

    rsm, counts, shown = build_rsm_with_stats(n, train_triplets)
    stats, shown_triu = analyze_coverage(n, shown)

    log.info(f"\nCoverage Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            log.info(f"  {key}: {val:.2f}")
        else:
            log.info(f"  {key}: {val:,}")

    # NaN analysis
    triu_idx = np.triu_indices(n, k=1)
    rsm_triu = rsm[triu_idx]
    n_nan = np.isnan(rsm_triu).sum()
    n_total = len(rsm_triu)
    log.info(f"\nRSM NaN Analysis:")
    log.info(f"  Total pairs (upper tri): {n_total:,}")
    log.info(f"  NaN pairs: {n_nan:,} ({100*n_nan/n_total:.2f}%)")
    log.info(f"  Valid pairs: {n_total - n_nan:,} ({100*(n_total-n_nan)/n_total:.2f}%)")

    # Plot coverage histogram
    plot_coverage_histogram(shown_triu, f"Observation counts ({args.dataset})", "coverage_histogram.pdf")

    # Plot NaN heatmap (subsample for visibility)
    plot_rsm_nan_heatmap(rsm[::10, ::10], "nan_heatmap_subsampled.pdf")

    # =========================================================================
    # PART 2: Fit SRF and analyze
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("PART 2: SRF Fitting and Evaluation")
    log.info("="*60)

    from pysrf import SRF

    # Fill NaN with 0.5 for SRF
    rsm_filled = rsm.copy()
    rsm_filled[np.isnan(rsm_filled)] = 0.5

    model = SRF(rank=args.rank, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    srf_emb = model.fit_transform(rsm_filled)

    srf_val_acc = compute_triplet_accuracy(srf_emb, val_triplets)
    spose_val_acc = compute_triplet_accuracy(spose, val_triplets)

    log.info(f"\nValidation Accuracy:")
    log.info(f"  SRF:   {srf_val_acc:.4f}")
    log.info(f"  SPoSE: {spose_val_acc:.4f}")
    log.info(f"  Noise ceiling: 0.6667")

    # Accuracy by observation count
    plot_observation_vs_accuracy(val_triplets, srf_emb, shown, "accuracy_by_obs_count.pdf")

    # =========================================================================
    # PART 3: Ablation - vary data amount
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("PART 3: Data Amount Ablation")
    log.info("="*60)

    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    ablation_results = []

    for frac in fractions:
        log.info(f"\nFraction: {frac:.0%}")

        # Subsample triplets
        n_samples = int(len(train_triplets) * frac)
        idx = np.random.choice(len(train_triplets), size=n_samples, replace=False)
        triplets_sub = train_triplets[idx]

        # Build RSM
        rsm_sub, _, shown_sub = build_rsm_with_stats(n, triplets_sub)
        stats_sub, _ = analyze_coverage(n, shown_sub)

        # Fill NaN
        rsm_sub_filled = rsm_sub.copy()
        n_nan_sub = np.isnan(rsm_sub_filled).sum()
        rsm_sub_filled[np.isnan(rsm_sub_filled)] = 0.5

        # Fit SRF
        model_sub = SRF(rank=args.rank, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        srf_sub = model_sub.fit_transform(rsm_sub_filled)

        srf_acc = compute_triplet_accuracy(srf_sub, val_triplets)

        log.info(f"  Triplets: {n_samples:,}")
        log.info(f"  Coverage: {stats_sub['coverage_pct']:.1f}%")
        log.info(f"  SRF accuracy: {srf_acc:.4f}")

        ablation_results.append({
            "fraction": frac,
            "n_triplets": n_samples,
            "coverage_pct": stats_sub["coverage_pct"],
            "n_nan": n_nan_sub,
            "val_accuracy": srf_acc,
            "model": "SRF",
        })

        # SPoSE is fixed
        ablation_results.append({
            "fraction": frac,
            "n_triplets": n_samples,
            "coverage_pct": stats_sub["coverage_pct"],
            "n_nan": n_nan_sub,
            "val_accuracy": spose_val_acc,
            "model": "SPoSE",
        })

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    log.info(f"\nSaved ablation_results.csv")

    # Plot ablation curves
    plot_ablation_curve(ablation_df, "ablation_curves.pdf")

    # =========================================================================
    # PART 4: Compare 1.47M vs 4.7M
    # =========================================================================
    if args.dataset == "1.47mio":
        log.info("\n" + "="*60)
        log.info("PART 4: Comparison with 4.7M dataset")
        log.info("="*60)

        # Load 4.7M data
        train_47 = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
        val_47 = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)

        rsm_47, _, shown_47 = build_rsm_with_stats(n, train_47)
        stats_47, _ = analyze_coverage(n, shown_47)

        rsm_47_filled = rsm_47.copy()
        rsm_47_filled[np.isnan(rsm_47_filled)] = 0.5

        model_47 = SRF(rank=args.rank, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        srf_47 = model_47.fit_transform(rsm_47_filled)

        # Evaluate on both val sets
        log.info(f"\n1.47M Dataset:")
        log.info(f"  Train triplets: {len(train_triplets):,}")
        log.info(f"  Coverage: {stats['coverage_pct']:.1f}%")
        log.info(f"  SRF on 1.47M val: {compute_triplet_accuracy(srf_emb, val_triplets):.4f}")

        log.info(f"\n4.7M Dataset:")
        log.info(f"  Train triplets: {len(train_47):,}")
        log.info(f"  Coverage: {stats_47['coverage_pct']:.1f}%")
        log.info(f"  SRF on 4.7M val: {compute_triplet_accuracy(srf_47, val_47):.4f}")

        # Cross-evaluation
        log.info(f"\nCross-evaluation:")
        log.info(f"  SRF(1.47M) on 4.7M val: {compute_triplet_accuracy(srf_emb, val_47):.4f}")
        log.info(f"  SRF(4.7M) on 1.47M val: {compute_triplet_accuracy(srf_47, val_triplets):.4f}")

    log.info(f"\n{'='*60}")
    log.info(f"All outputs saved to {OUTPUT_DIR}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
