"""SPOSE dimension recovery: RSA vs SRF comparison.

Compares two approaches for testing dimension recovery:
1. RSA: Mantel test (hypothesis RSM vs measured similarity)
2. SRF-LOO: Leave-one-out alignment test (prevents overfitting)

Usage:
    ./scripts/submit experiments/rsa_comparison/spose.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from omegaconf import DictConfig
from statsmodels.stats.multitest import multipletests

from pysrf import SRF

from src.tools.rsa import mantel_test, loo_alignment_test_multi
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure
from utils.io import load_spose_embedding


def run_single_condition(
    full_data: np.ndarray,
    n_objects: int,
    num_dims: int,
    snr: float,
    n_permutations: int,
    alpha: float,
    seed: int,
) -> list[dict]:
    """Run one SPOSE condition."""
    rng = np.random.default_rng(seed)
    dims = rng.choice(full_data.shape[1], size=num_dims, replace=False)
    selected_objects = rng.choice(full_data.shape[0], size=n_objects, replace=False)

    data = full_data[selected_objects][:, dims]
    rank = len(dims)

    # Track variance of each selected dimension
    dim_variances = np.var(data, axis=0)

    seed_base = 10000 + 97 * (seed + 1)
    noisy_data = add_positive_noise_with_snr(data, snr, rng=seed_base)
    measured_similarity = noisy_data @ noisy_data.T

    model = SRF(rank=rank, verbose=False, tol=0.0, random_state=seed)
    W = model.fit_transform(measured_similarity)

    # RSA: Mantel test per dimension
    rsa_raw_ps = []
    rsa_r_obs = []
    for i in range(rank):
        x_i = data[:, [i]]
        hypothesis_rsm = x_i @ x_i.T
        p, _, r = mantel_test(
            hypothesis_rsm,
            measured_similarity,
            permutations=n_permutations,
            random_state=seed_base + 100 * i,
            two_sided=True,
        )
        rsa_raw_ps.append(p)
        rsa_r_obs.append(r)

    rsa_reject, rsa_corrected_ps = multipletests(rsa_raw_ps, alpha, method="fdr_bh")[:2]

    # SRF-LOO: Leave-one-out alignment test
    srf_results = loo_alignment_test_multi(
        W, data, permutations=n_permutations, alpha=alpha, random_state=seed_base + 200
    )

    rows = []
    for i in range(rank):
        rows.append({
            "n_objects": n_objects,
            "snr": snr,
            "repeat": seed,
            "dimension": i + 1,
            "dim_variance": float(dim_variances[i]),
            "method": "RSA",
            "r_obs": float(rsa_r_obs[i]),
            "raw_p": float(rsa_raw_ps[i]),
            "corrected_p": float(rsa_corrected_ps[i]),
            "significant": bool(rsa_reject[i]),
        })
        rows.append({
            "n_objects": n_objects,
            "snr": snr,
            "repeat": seed,
            "dimension": i + 1,
            "dim_variance": float(dim_variances[i]),
            "method": "SRF-LOO",
            "r_obs": float(srf_results["r_obs"][i]),
            "raw_p": float(srf_results["raw_p"][i]),
            "corrected_p": float(srf_results["corrected_p"][i]),
            "significant": bool(srf_results["significant"][i]),
        })
    return rows


def run(cfg: DictConfig) -> None:
    """Run SPOSE experiment across all SNR levels and repeats."""
    output_dir = Path.cwd()

    print("=" * 60)
    print("SPOSE: RSA vs SRF-LOO Comparison")
    print("=" * 60)

    full_data = load_spose_embedding(num_dims=66)
    print(f"Loaded SPOSE data: {full_data.shape}")

    snr_levels = list(cfg.snrs)
    n_repeats = cfg.n_repeats
    n_objects = cfg.n_objects
    num_dims = cfg.num_dims
    n_permutations = cfg.n_permutations
    alpha = cfg.alpha
    n_jobs = cfg.get("n_jobs", -1)

    print(f"\nParameters:")
    print(f"  Objects: {n_objects}, Dims: {num_dims}")
    print(f"  Repeats: {n_repeats}, Permutations: {n_permutations}")
    print(f"  SNR levels: {snr_levels}")

    all_rows = []
    for snr in snr_levels:
        print(f"  SNR={snr:.2f}...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_condition)(
                full_data, n_objects, num_dims, snr, n_permutations, alpha, seed
            )
            for seed in range(n_repeats)
        )

        for rep_rows in results:
            all_rows.extend(rep_rows)

    # Save results
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "spose.csv", index=False)
    print(f"\nSaved {output_dir / 'spose.csv'}")

    # Compute power summary
    print("\n" + "=" * 60)
    print("RESULTS (calibration should be ≤5% at SNR=0):")
    print("=" * 60)
    print(f"{'SNR':<8} {'RSA':<12} {'SRF-LOO':<12}")
    print("-" * 32)

    power_data = {"snr": [], "RSA": [], "SRF-LOO": []}
    for snr in snr_levels:
        snr_df = df[df["snr"] == snr]
        for method in ["RSA", "SRF-LOO"]:
            method_df = snr_df[snr_df["method"] == method]
            power = method_df["significant"].mean() * 100
            power_data[method if method != "SRF-LOO" else "SRF-LOO"].append(power)
        power_data["snr"].append(snr)
        print(f"{snr:<8.2f} {power_data['RSA'][-1]:<12.1f} {power_data['SRF-LOO'][-1]:<12.1f}")

    # Plot 1: Main power comparison
    fig, axes = create_figure("wide", ncols=2)

    ax = axes[0]
    ax.plot(power_data["snr"], power_data["RSA"], "^-", color=CMAP[2], lw=2, ms=6, label="RSA")
    ax.plot(power_data["snr"], power_data["SRF-LOO"], "s-", color=CMAP[1], lw=2, ms=6, label="SRF-LOO")
    ax.axhline(5, color=GRAY["dark"], ls="--", lw=1, alpha=0.7, label="α = 5%")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=8)
    ax.set_title(f"SPOSE: Power ({n_objects} objects, {num_dims} dims)")
    despine(ax)

    # P-value distribution at SNR=0
    ax = axes[1]
    snr0_df = df[df["snr"] == 0.0]
    bins = np.linspace(0, 1, 21)
    rsa_ps = snr0_df[snr0_df["method"] == "RSA"]["raw_p"]
    loo_ps = snr0_df[snr0_df["method"] == "SRF-LOO"]["raw_p"]
    ax.hist(rsa_ps, bins=bins, alpha=0.5, color=CMAP[2], label="RSA", density=True)
    ax.hist(loo_ps, bins=bins, alpha=0.5, color=CMAP[1], label="SRF-LOO", density=True)
    ax.axhline(1, color=GRAY["dark"], ls="--", lw=1.5, label="Uniform")
    ax.axvline(0.05, color="red", ls=":", lw=1.5)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Calibration at SNR=0")
    ax.legend(fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "spose_power.pdf")
    print(f"Saved {output_dir / 'spose_power.pdf'}")

    # Plot 2: Diagnostic - Variance distribution and its effect
    fig, axes = create_figure("full_width", ncols=3)

    # 2a: Histogram of dimension variances
    ax = axes[0]
    all_variances = df["dim_variance"].values
    ax.hist(all_variances, bins=30, color=GRAY["medium"], edgecolor="white", alpha=0.8)
    ax.axvline(np.median(all_variances), color=CMAP[0], ls="--", lw=2, label=f"Median: {np.median(all_variances):.3f}")
    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Count")
    ax.set_title("SPOSE dimension variances")
    var_ratio = all_variances.max() / all_variances.min()
    ax.text(0.95, 0.95, f"Max/Min: {var_ratio:.1f}×", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.legend(fontsize=8)
    despine(ax)

    # 2b: Power vs variance at high SNR (binned)
    ax = axes[1]
    high_snr_df = df[df["snr"] == 1.0]
    variance_bins = pd.qcut(high_snr_df["dim_variance"], q=5, duplicates="drop")
    for method, color, marker in [("RSA", CMAP[2], "^"), ("SRF-LOO", CMAP[1], "s")]:
        method_df = high_snr_df[high_snr_df["method"] == method]
        grouped = method_df.groupby(variance_bins)["significant"].mean() * 100
        bin_centers = [interval.mid for interval in grouped.index]
        ax.plot(bin_centers, grouped.values, f"{marker}-", color=color, lw=2, ms=6, label=method)
    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Power (%)")
    ax.set_title("Power vs variance (SNR=1.0)")
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=8)
    despine(ax)

    # 2c: RSA r_obs vs variance at high SNR
    ax = axes[2]
    high_snr_rsa = high_snr_df[high_snr_df["method"] == "RSA"]
    high_snr_loo = high_snr_df[high_snr_df["method"] == "SRF-LOO"]
    ax.scatter(high_snr_rsa["dim_variance"], high_snr_rsa["r_obs"], alpha=0.1, s=10, color=CMAP[2], label="RSA")
    ax.scatter(high_snr_loo["dim_variance"], high_snr_loo["r_obs"], alpha=0.1, s=10, color=CMAP[1], label="SRF-LOO")
    # Add trend lines
    from scipy.stats import pearsonr
    r_rsa, _ = pearsonr(high_snr_rsa["dim_variance"], high_snr_rsa["r_obs"])
    r_loo, _ = pearsonr(high_snr_loo["dim_variance"], high_snr_loo["r_obs"])
    ax.text(0.05, 0.95, f"r(RSA)={r_rsa:.2f}\nr(LOO)={r_loo:.2f}", transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Observed correlation (r)")
    ax.set_title("Effect strength vs variance (SNR=1.0)")
    ax.legend(fontsize=8, markerscale=3)
    despine(ax)

    save_figure(fig, output_dir / "spose_variance_effects.pdf")
    print(f"Saved {output_dir / 'spose_variance_effects.pdf'}")

    # Plot 3: Power curves stratified by variance
    fig, axes = create_figure("wide", ncols=2)

    # Stratify by variance (low/high)
    median_var = df["dim_variance"].median()
    df["var_group"] = np.where(df["dim_variance"] <= median_var, "Low variance", "High variance")

    for ax_idx, (method, color) in enumerate([("RSA", CMAP[2]), ("SRF-LOO", CMAP[1])]):
        ax = axes[ax_idx]
        method_df = df[df["method"] == method]

        for var_group, ls in [("High variance", "-"), ("Low variance", "--")]:
            group_df = method_df[method_df["var_group"] == var_group]
            power_by_snr = group_df.groupby("snr")["significant"].mean() * 100
            ax.plot(power_by_snr.index, power_by_snr.values, ls, color=color, lw=2, label=var_group)

        ax.axhline(5, color=GRAY["dark"], ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Power (%)")
        ax.set_ylim([-5, 105])
        ax.set_title(f"{method}: Power by variance")
        ax.legend(fontsize=8)
        despine(ax)

    save_figure(fig, output_dir / "spose_power_by_variance.pdf")
    print(f"Saved {output_dir / 'spose_power_by_variance.pdf'}")
