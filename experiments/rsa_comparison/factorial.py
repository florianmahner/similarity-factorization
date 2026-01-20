"""Factorial design: RSA vs SRF-LOO comparison.

Tests factor recovery using a factorial design matrix where items are
combinations of factor levels.

Usage:
    ./scripts/submit experiments/rsa_comparison/factorial.py
"""

from __future__ import annotations

import itertools as it
import pandas as pd
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from pysrf import SRF

from src.tools.rsa import mantel_test, loo_alignment_test_multi
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure


def create_factorial_data(
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]], list[tuple[str, ...]]]:
    """Create factorial design matrix with column ranges per factor."""
    items = list(it.product(*levels.values()))

    col_ranges: dict[str, tuple[int, int]] = {}
    features: list[np.ndarray] = []
    col_start = 0

    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        features.append(Z)
        col_ranges[factor] = (col_start, col_start + len(lvls))
        col_start += len(lvls)

    X = np.concatenate(features, axis=1)
    return X, col_ranges, items


def run_single_condition(
    X: np.ndarray,
    col_ranges: dict[str, tuple[int, int]],
    snr: float,
    n_permutations: int,
    alpha: float,
    seed: int,
) -> list[dict]:
    """Run one factorial condition."""
    rank = X.shape[1]
    seed_base = 10000 + 97 * (seed + 1)

    # Add noise to features and compute similarity
    noisy_data = add_positive_noise_with_snr(X, snr, rng=seed_base)
    measured_similarity = noisy_data @ noisy_data.T

    # Fit SRF
    model = SRF(rank=rank, random_state=seed, verbose=False, tol=0.0)
    W = model.fit_transform(measured_similarity)

    # SRF-LOO: Test all columns at once
    srf_results = loo_alignment_test_multi(
        W, X, permutations=n_permutations, alpha=alpha, random_state=seed_base + 200
    )

    # Build results per column first
    rows = []
    for col_idx in range(X.shape[1]):
        # Find which factor this column belongs to
        factor_name = None
        col_in_factor = None
        for fname, (col_start, col_end) in col_ranges.items():
            if col_start <= col_idx < col_end:
                factor_name = fname
                col_in_factor = col_idx - col_start
                break

        # RSA: Mantel test for this column's hypothesis RSM
        x_col = X[:, [col_idx]]
        hypothesis_rsm = x_col @ x_col.T
        p_rsa, _, r_rsa = mantel_test(
            hypothesis_rsm,
            measured_similarity,
            permutations=n_permutations,
            random_state=seed_base + 100 * col_idx,
            two_sided=True,
        )

        rows.append({
            "snr": snr,
            "repeat": seed,
            "factor": factor_name,
            "column": col_idx,
            "col_in_factor": col_in_factor,
            "method": "RSA",
            "r_obs": float(r_rsa),
            "raw_p": float(p_rsa),
        })
        rows.append({
            "snr": snr,
            "repeat": seed,
            "factor": factor_name,
            "column": col_idx,
            "col_in_factor": col_in_factor,
            "method": "SRF-LOO",
            "r_obs": float(srf_results["r_obs"][col_idx]),
            "raw_p": float(srf_results["raw_p"][col_idx]),
        })

    # FDR correction within each method
    for method in ["RSA", "SRF-LOO"]:
        method_rows = [r for r in rows if r["method"] == method]
        raw_ps = [r["raw_p"] for r in method_rows]
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]
        for r, corrected_p, rej in zip(method_rows, corr_ps, reject):
            r["corrected_p"] = float(corrected_p)
            r["significant"] = bool(rej)

    return rows


def run(cfg: DictConfig) -> None:
    """Run factorial experiment across all SNR levels and repeats."""
    output_dir = Path.cwd()

    print("=" * 60)
    print("Factorial Design: RSA vs SRF-LOO Comparison")
    print("=" * 60)

    levels = OmegaConf.to_container(cfg.levels, resolve=True)
    X, col_ranges, _ = create_factorial_data(levels)

    snr_levels = list(cfg.snrs)
    n_repeats = cfg.n_repeats
    n_permutations = cfg.n_permutations
    alpha = cfg.alpha
    n_jobs = cfg.get("n_jobs", -1)

    print(f"\nFactors: {list(levels.keys())}")
    for factor, (start, end) in col_ranges.items():
        print(f"  {factor}: {end - start} levels (columns {start}-{end-1})")
    print(f"Design matrix: {X.shape[0]} items × {X.shape[1]} columns")
    print(f"Repeats: {n_repeats}, Permutations: {n_permutations}")
    print(f"SNR levels: {snr_levels}")

    all_rows = []
    for snr in snr_levels:
        print(f"  SNR={snr:.2f}...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_condition)(
                X, col_ranges, snr, n_permutations, alpha, seed
            )
            for seed in range(n_repeats)
        )

        for rep_rows in results:
            all_rows.extend(rep_rows)

    # Save results
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "factorial.csv", index=False)
    print(f"\nSaved {output_dir / 'factorial.csv'}")

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
            power_data[method].append(power)
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
    ax.set_title(f"Factorial: Power ({X.shape[0]} items, {len(levels)} factors)")
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

    save_figure(fig, output_dir / "factorial_power.pdf")
    print(f"Saved {output_dir / 'factorial_power.pdf'}")

    # Plot 2: Power by factor
    fig, ax = create_figure("single")

    factors = list(col_ranges.keys())
    x_pos = np.arange(len(factors))
    width = 0.35

    high_snr_df = df[df["snr"] == snr_levels[-1]]  # Use highest SNR

    rsa_power = []
    loo_power = []
    for factor in factors:
        factor_df = high_snr_df[high_snr_df["factor"] == factor]
        rsa_power.append(factor_df[factor_df["method"] == "RSA"]["significant"].mean() * 100)
        loo_power.append(factor_df[factor_df["method"] == "SRF-LOO"]["significant"].mean() * 100)

    ax.bar(x_pos - width/2, rsa_power, width, label="RSA", color=CMAP[2], alpha=0.8)
    ax.bar(x_pos + width/2, loo_power, width, label="SRF-LOO", color=CMAP[1], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(factors, rotation=45, ha="right")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([0, 105])
    ax.set_title(f"Power by factor (SNR={snr_levels[-1]})")
    ax.legend(fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "factorial_power_by_factor.pdf")
    print(f"Saved {output_dir / 'factorial_power_by_factor.pdf'}")
