"""
THINGS-2k Monkey Data: Sanity Check and SRF Embedding Analysis.

Data structure (from raw .mat files):
- ALLMUA: (300, n_trials, 1024) - 300 time bins (-100 to 199ms), n_trials, 1024 channels
- ALLMAT: (7, n_trials) - metadata per trial:
  - Row 0: Trial ID
  - Row 1: Image index (1-2000)
  - Row 2: Session (1-15)
  - Row 3: Condition (1-4)
  - Row 4: Constant (1)
  - Row 5: Block (1-5/6)
  - Row 6: Day/repeat (1-5)
- tb: (300, 1) - time bin centers in ms

This script:
1. Loads data, averages over trials per image
2. Computes split-half reliability per channel
3. Shows top responding images
4. Computes Gaussian kernel similarity (median heuristic)
5. Applies SRF factorization
6. Visualizes top images per dimension
"""
from datetime import datetime
from pathlib import Path

import argparse
import h5py
import logging
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("/LOCAL/fmahner/similarity-factorization/data/things-monkey/THINGS-2k")
OUTPUT_DIR = get_output_dir()

# Time window for averaging (stimulus-evoked response)
TIME_WINDOW = (50, 150)  # ms after stimulus onset


def load_monkey_data(monkey: str, time_window: tuple[int, int] = TIME_WINDOW) -> dict:
    """
    Load and preprocess monkey data.

    Returns dict with:
    - data_per_image: (n_images, n_channels) averaged responses
    - data_per_trial: (n_trials, n_channels) trial-level responses
    - image_indices: (n_trials,) image index per trial
    - n_trials_per_image: trials per image
    - metadata: file info
    """
    if monkey == "F":
        filepath = DATA_ROOT / "monkeyF" / "THINGS2_MUA_trials.mat"
    elif monkey == "N_new":
        filepath = DATA_ROOT / "monkeyN_new" / "THINGS2_new_MUA_trials.mat"
    else:
        raise ValueError(f"Unknown monkey: {monkey}")

    log.info(f"\n{'='*60}")
    log.info(f"Loading Monkey {monkey}")
    log.info(f"File: {filepath}")
    log.info(f"Size: {filepath.stat().st_size / 1e9:.2f} GB")
    log.info(f"{'='*60}")

    with h5py.File(filepath, "r") as f:
        # Load metadata (small)
        allmat = f["ALLMAT"][:]
        tb = f["tb"][:].flatten()

        # Image indices (1-based in file, convert to 0-based)
        image_indices = allmat[1].astype(int) - 1
        n_images = len(np.unique(image_indices))
        n_trials = len(image_indices)
        n_channels = f["ALLMUA"].shape[2]

        log.info(f"\nData dimensions:")
        log.info(f"  Time bins: {len(tb)} ({tb[0]:.0f} to {tb[-1]:.0f} ms)")
        log.info(f"  N trials: {n_trials}")
        log.info(f"  N images: {n_images}")
        log.info(f"  N channels: {n_channels}")

        # Find time indices for averaging window
        t_start_idx = np.argmin(np.abs(tb - time_window[0]))
        t_end_idx = np.argmin(np.abs(tb - time_window[1]))
        log.info(f"\nTime window: {time_window[0]}-{time_window[1]} ms (bins {t_start_idx}-{t_end_idx})")

        # Load MUA data for the time window only (to save memory)
        log.info(f"\nLoading MUA data (time window only)...")
        mua_window = f["ALLMUA"][t_start_idx:t_end_idx, :, :]  # (n_time, n_trials, n_channels)

        # Average over time window -> (n_trials, n_channels)
        log.info(f"  MUA window shape: {mua_window.shape}")
        data_per_trial = np.nanmean(mua_window, axis=0)
        log.info(f"  Per-trial data shape: {data_per_trial.shape}")

    # Average per image
    log.info(f"\nAveraging across trials per image...")
    data_per_image = np.zeros((n_images, n_channels))
    n_trials_per_image = np.zeros(n_images)

    for img_idx in range(n_images):
        mask = image_indices == img_idx
        n_trials_per_image[img_idx] = mask.sum()
        if mask.sum() > 0:
            data_per_image[img_idx] = np.nanmean(data_per_trial[mask], axis=0)

    log.info(f"  Per-image data shape: {data_per_image.shape}")
    log.info(f"  Trials per image: min={n_trials_per_image.min():.0f}, max={n_trials_per_image.max():.0f}, mean={n_trials_per_image.mean():.1f}")

    # Data stats
    log.info(f"\nData statistics:")
    log.info(f"  Response range: [{np.nanmin(data_per_image):.4f}, {np.nanmax(data_per_image):.4f}]")
    log.info(f"  Response mean: {np.nanmean(data_per_image):.4f}")
    log.info(f"  NaN ratio: {np.isnan(data_per_image).mean():.4f}")

    return {
        "data_per_image": data_per_image,
        "data_per_trial": data_per_trial,
        "image_indices": image_indices,
        "n_trials_per_image": n_trials_per_image,
        "time_bins": tb,
        "allmat": allmat,
        "n_images": n_images,
        "n_channels": n_channels,
    }


def compute_reliability(data_per_trial: np.ndarray, image_indices: np.ndarray) -> np.ndarray:
    """
    Compute split-half reliability per channel.

    Split trials for each image into two halves, average within halves,
    then correlate across images.
    """
    n_images = len(np.unique(image_indices))
    n_channels = data_per_trial.shape[1]

    # Split trials per image
    half1_avg = np.zeros((n_images, n_channels))
    half2_avg = np.zeros((n_images, n_channels))

    for img_idx in range(n_images):
        mask = image_indices == img_idx
        trial_data = data_per_trial[mask]
        n = len(trial_data)

        if n >= 2:
            mid = n // 2
            half1_avg[img_idx] = np.nanmean(trial_data[:mid], axis=0)
            half2_avg[img_idx] = np.nanmean(trial_data[mid:], axis=0)
        else:
            half1_avg[img_idx] = trial_data[0] if n > 0 else np.nan
            half2_avg[img_idx] = trial_data[0] if n > 0 else np.nan

    # Correlation per channel
    reliability = np.zeros(n_channels)
    for ch in range(n_channels):
        h1, h2 = half1_avg[:, ch], half2_avg[:, ch]
        valid = ~(np.isnan(h1) | np.isnan(h2))
        if valid.sum() > 10:
            r, _ = stats.pearsonr(h1[valid], h2[valid])
            reliability[ch] = 2 * r / (1 + r)  # Spearman-Brown
        else:
            reliability[ch] = np.nan

    return reliability


def sanity_check_reliability(reliability: np.ndarray, monkey: str) -> pd.DataFrame:
    """Analyze reliability distribution."""
    log.info(f"\n{'='*60}")
    log.info(f"RELIABILITY ANALYSIS: Monkey {monkey}")
    log.info(f"{'='*60}")

    valid = ~np.isnan(reliability)
    rel_valid = reliability[valid]

    log.info(f"\nReliability statistics:")
    log.info(f"  N channels: {len(reliability)} (valid: {valid.sum()})")
    log.info(f"  Mean: {np.mean(rel_valid):.3f}")
    log.info(f"  Median: {np.median(rel_valid):.3f}")
    log.info(f"  Std: {np.std(rel_valid):.3f}")
    log.info(f"  Range: [{rel_valid.min():.3f}, {rel_valid.max():.3f}]")

    # Thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    log.info(f"\nChannels passing thresholds:")
    for t in thresholds:
        n_pass = (rel_valid >= t).sum()
        pct = 100 * n_pass / len(rel_valid)
        log.info(f"  r >= {t}: {n_pass} ({pct:.1f}%)")

    # Per-channel stats
    results = []
    for ch in range(len(reliability)):
        results.append({
            "monkey": monkey,
            "channel": ch,
            "reliability": reliability[ch],
        })

    return pd.DataFrame(results)


def sanity_check_responses(data: np.ndarray, monkey: str) -> pd.DataFrame:
    """Analyze response distributions per channel."""
    log.info(f"\n{'='*60}")
    log.info(f"RESPONSE STATISTICS: Monkey {monkey}")
    log.info(f"{'='*60}")

    n_images, n_channels = data.shape

    ch_mean = np.nanmean(data, axis=0)
    ch_std = np.nanstd(data, axis=0)
    ch_max = np.nanmax(data, axis=0)
    ch_min = np.nanmin(data, axis=0)
    ch_range = ch_max - ch_min

    log.info(f"\nPer-channel statistics:")
    log.info(f"  Mean response: {ch_mean.mean():.4f} ± {ch_mean.std():.4f}")
    log.info(f"  Std response: {ch_std.mean():.4f} ± {ch_std.std():.4f}")
    log.info(f"  Dynamic range: {ch_range.mean():.4f} ± {ch_range.std():.4f}")

    # Check for dead/saturated channels
    dead_mask = ch_std < 0.001
    log.info(f"\nDead channels (std < 0.001): {dead_mask.sum()}")

    # Per-image statistics
    img_mean = np.nanmean(data, axis=1)
    img_std = np.nanstd(data, axis=1)

    log.info(f"\nPer-image statistics:")
    log.info(f"  Mean response: {img_mean.mean():.4f} ± {img_mean.std():.4f}")
    log.info(f"  Std response: {img_std.mean():.4f} ± {img_std.std():.4f}")

    results = []
    for ch in range(n_channels):
        results.append({
            "monkey": monkey,
            "channel": ch,
            "mean": ch_mean[ch],
            "std": ch_std[ch],
            "min": ch_min[ch],
            "max": ch_max[ch],
            "range": ch_range[ch],
        })

    return pd.DataFrame(results)


def get_top_responding_images(data: np.ndarray, k: int = 20) -> pd.DataFrame:
    """Get top-k images with highest mean response."""
    img_response = np.nanmean(data, axis=1)
    top_idx = np.argsort(img_response)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            "rank": rank + 1,
            "image_idx": idx,
            "mean_response": img_response[idx],
            "std_response": np.nanstd(data[idx]),
        })

    return pd.DataFrame(results)


def gaussian_kernel_median_heuristic(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute Gaussian kernel with median heuristic for bandwidth.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    where sigma = median of pairwise distances.
    """
    from scipy.spatial.distance import pdist, squareform

    log.info("\nComputing Gaussian kernel with median heuristic...")

    distances = pdist(X, metric="euclidean")
    sigma = np.median(distances)

    log.info(f"  Median distance (sigma): {sigma:.4f}")
    log.info(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

    dist_matrix = squareform(distances)
    K = np.exp(-dist_matrix**2 / (2 * sigma**2))

    log.info(f"  Kernel range: [{K.min():.4f}, {K.max():.4f}]")
    log.info(f"  Kernel mean (off-diag): {K[np.triu_indices_from(K, k=1)].mean():.4f}")

    return K, sigma


def apply_srf(rsm: np.ndarray, rank: int = 10) -> tuple[np.ndarray, dict]:
    """Apply SRF factorization."""
    from pysrf import SRF

    log.info(f"\nApplying SRF (rank={rank})...")

    model = SRF(
        rank=rank,
        rho=1.0,
        max_outer=200,
        max_inner=30,
        tol=1e-4,
        verbose=1,
        random_state=42,
    )

    embedding = model.fit_transform(rsm)
    reconstruction = embedding @ embedding.T
    rmse = np.sqrt(np.mean((rsm - reconstruction)**2))

    log.info(f"  Converged in {model.n_iter_} iterations")
    log.info(f"  Reconstruction RMSE: {rmse:.4f}")

    return embedding, {"rmse": rmse, "n_iter": model.n_iter_}


def get_top_images_per_dimension(embedding: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Get top-k images with highest loading per dimension."""
    results = []
    n_dims = embedding.shape[1]

    for dim in range(n_dims):
        loadings = embedding[:, dim]
        top_idx = np.argsort(loadings)[::-1][:k]

        for rank, idx in enumerate(top_idx):
            results.append({
                "dimension": dim + 1,
                "rank": rank + 1,
                "image_idx": idx,
                "loading": loadings[idx],
            })

    return pd.DataFrame(results)


def plot_reliability_distribution(rel_f: np.ndarray, rel_n: np.ndarray, output_path: Path):
    """Plot reliability distributions for both monkeys."""
    import matplotlib
    matplotlib.use("Agg")
    from src.colors import TEAL, ROSE, GRAY_DARK
    from src.utils.figure_theme import create_figure, save_figure, despine

    fig, ax = create_figure("wide")

    bins = np.linspace(-0.2, 1.0, 30)
    ax.hist(rel_f[~np.isnan(rel_f)], bins=bins, alpha=0.6,
            label=f"Monkey F (n={(~np.isnan(rel_f)).sum()})", color=TEAL)
    ax.hist(rel_n[~np.isnan(rel_n)], bins=bins, alpha=0.6,
            label=f"Monkey N_new (n={(~np.isnan(rel_n)).sum()})", color=ROSE)

    ax.axvline(0.3, color=GRAY_DARK, linestyle="--", linewidth=1)
    ax.set_xlabel("Split-half reliability")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    despine(ax)

    save_figure(fig, output_path)
    log.info(f"Saved: {output_path}")


def plot_response_distribution(data_f: np.ndarray, data_n: np.ndarray, output_path: Path):
    """Plot mean response distributions."""
    import matplotlib
    matplotlib.use("Agg")
    from src.colors import TEAL, ROSE
    from src.utils.figure_theme import create_figure, save_figure, despine

    fig, ax = create_figure("wide")

    mean_f = np.nanmean(data_f, axis=1)
    mean_n = np.nanmean(data_n, axis=1)

    ax.hist(mean_f, bins=50, alpha=0.6, label="Monkey F", color=TEAL)
    ax.hist(mean_n, bins=50, alpha=0.6, label="Monkey N_new", color=ROSE)

    ax.set_xlabel("Mean response per image")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    despine(ax)

    save_figure(fig, output_path)
    log.info(f"Saved: {output_path}")


def plot_embedding_structure(embedding: np.ndarray, monkey: str, output_path: Path):
    """Plot embedding structure (sorted heatmap + variance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.colors import TEAL
    from src.utils.figure_theme import create_figure, save_figure, despine

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Heatmap (top 100 images sorted by max loading)
    ax = axes[0]
    max_loadings = embedding.max(axis=1)
    sorted_idx = np.argsort(max_loadings)[::-1][:100]
    im = ax.imshow(embedding[sorted_idx], aspect="auto", cmap="viridis")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Image (sorted)")
    ax.set_title(f"Monkey {monkey}")
    plt.colorbar(im, ax=ax)

    # Variance per dimension
    ax = axes[1]
    dim_var = np.var(embedding, axis=0)
    dim_contrib = dim_var / dim_var.sum()
    ax.bar(range(1, len(dim_contrib) + 1), dim_contrib, color=TEAL)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Variance fraction")
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def main(rank: int = 10, reliab_threshold: float = 0.3, time_window: tuple[int, int] = TIME_WINDOW):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("="*70)
    log.info("THINGS-2k Monkey Data: Sanity Check and SRF Analysis")
    log.info("="*70)
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"SRF rank: {rank}")
    log.info(f"Reliability threshold: {reliab_threshold}")
    log.info(f"Time window: {time_window} ms")

    results = {}
    all_stats = []

    for monkey in ["F", "N_new"]:
        log.info(f"\n\n{'#'*70}")
        log.info(f"# MONKEY {monkey}")
        log.info(f"{'#'*70}")

        # Load data
        data = load_monkey_data(monkey, time_window)
        results[monkey] = data

        # Compute reliability
        log.info("\n--- RELIABILITY ---")
        reliability = compute_reliability(data["data_per_trial"], data["image_indices"])
        data["reliability"] = reliability
        rel_stats = sanity_check_reliability(reliability, monkey)

        # Response statistics
        log.info("\n--- RESPONSE STATISTICS ---")
        resp_stats = sanity_check_responses(data["data_per_image"], monkey)
        all_stats.append(resp_stats)

        # Top responding images
        log.info(f"\n--- TOP RESPONDING IMAGES ---")
        top_images = get_top_responding_images(data["data_per_image"], k=20)
        log.info(f"\nTop 10 images by mean response:")
        log.info(top_images.head(10).to_string(index=False))
        top_images.to_csv(OUTPUT_DIR / f"top_responding_{monkey}.csv", index=False)

        # Save reliability
        np.save(OUTPUT_DIR / f"reliability_{monkey}.npy", reliability)
        rel_stats.to_csv(OUTPUT_DIR / f"reliability_stats_{monkey}.csv", index=False)

    # Filter channels and compute similarity
    log.info("\n\n" + "="*70)
    log.info("SIMILARITY COMPUTATION AND SRF")
    log.info("="*70)

    embeddings = {}
    for monkey in ["F", "N_new"]:
        data = results[monkey]
        reliability = data["reliability"]

        # Filter by reliability
        reliable_mask = reliability >= reliab_threshold
        n_reliable = reliable_mask.sum()
        log.info(f"\n[{monkey}] Channels with reliability >= {reliab_threshold}: {n_reliable}")

        if n_reliable < 10:
            log.warning(f"  Too few reliable channels, using all {len(reliability)}")
            data_filtered = data["data_per_image"]
        else:
            data_filtered = data["data_per_image"][:, reliable_mask]

        log.info(f"  Filtered data shape: {data_filtered.shape}")

        # Compute Gaussian kernel
        rsm, sigma = gaussian_kernel_median_heuristic(data_filtered)
        np.save(OUTPUT_DIR / f"rsm_{monkey}.npy", rsm)
        data["rsm"] = rsm
        data["sigma"] = sigma

        # Apply SRF
        embedding, srf_info = apply_srf(rsm, rank=rank)
        embeddings[monkey] = embedding
        np.save(OUTPUT_DIR / f"embedding_{monkey}.npy", embedding)

        # Top images per dimension
        top_per_dim = get_top_images_per_dimension(embedding, k=10)
        top_per_dim.to_csv(OUTPUT_DIR / f"top_per_dimension_{monkey}.csv", index=False)

        log.info(f"\n  Top images per dimension ({monkey}):")
        for dim in range(min(5, rank)):
            dim_data = top_per_dim[top_per_dim["dimension"] == dim + 1]
            top_imgs = dim_data["image_idx"].head(5).tolist()
            top_loads = dim_data["loading"].head(5).tolist()
            log.info(f"    Dim {dim+1}: images {top_imgs}, loadings {[f'{l:.3f}' for l in top_loads]}")

    # Plots
    log.info("\n\n" + "="*70)
    log.info("GENERATING PLOTS")
    log.info("="*70)

    rel_f = results["F"]["reliability"]
    rel_n = results["N_new"]["reliability"]
    plot_reliability_distribution(rel_f, rel_n, OUTPUT_DIR / "reliability_distribution.pdf")

    data_f = results["F"]["data_per_image"]
    data_n = results["N_new"]["data_per_image"]
    plot_response_distribution(data_f, data_n, OUTPUT_DIR / "response_distribution.pdf")

    for monkey, emb in embeddings.items():
        plot_embedding_structure(emb, monkey, OUTPUT_DIR / f"embedding_structure_{monkey}.pdf")

    # Save summary
    summary = {
        "monkey_F": {
            "n_images": results["F"]["n_images"],
            "n_channels": results["F"]["n_channels"],
            "n_trials": len(results["F"]["image_indices"]),
            "mean_reliability": float(np.nanmean(rel_f)),
            "n_reliable_channels": int((rel_f >= reliab_threshold).sum()),
            "kernel_sigma": float(results["F"]["sigma"]),
        },
        "monkey_N_new": {
            "n_images": results["N_new"]["n_images"],
            "n_channels": results["N_new"]["n_channels"],
            "n_trials": len(results["N_new"]["image_indices"]),
            "mean_reliability": float(np.nanmean(rel_n)),
            "n_reliable_channels": int((rel_n >= reliab_threshold).sum()),
            "kernel_sigma": float(results["N_new"]["sigma"]),
        },
    }

    import json
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*70}")
    log.info(f"DONE! Results saved to: {OUTPUT_DIR}")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=10, help="SRF rank")
    parser.add_argument("--reliab-threshold", type=float, default=0.3)
    parser.add_argument("--time-start", type=int, default=50, help="Time window start (ms)")
    parser.add_argument("--time-end", type=int, default=150, help="Time window end (ms)")
    args = parser.parse_args()

    main(
        rank=args.rank,
        reliab_threshold=args.reliab_threshold,
        time_window=(args.time_start, args.time_end),
    )
