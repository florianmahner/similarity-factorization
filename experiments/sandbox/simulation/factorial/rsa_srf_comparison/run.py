"""
RSA vs SRF Power Analysis - Comprehensive Comparison.

Tests multiple conditions to understand when/why RSA vs SRF differ:
1. Noise model: features vs RSM
2. Permutation: restricted (factorial) vs unrestricted (full)
3. Methods: RSA, SRF structural alignment

Power = proportion of repeats where p < 0.05
"""

from datetime import datetime
from pathlib import Path
import itertools as it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from pysrf import SRF
from src.utils.helpers import add_positive_noise_with_snr
from src.colors import ROSE, TEAL, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir


# --- Permutation Helpers ---


def get_stratified_indices(n: int, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutation indices that only shuffle within strata groups."""
    idx = np.arange(n)
    unique_strata = np.unique(strata, axis=0)
    for group in unique_strata:
        mask = (strata == group).all(axis=1)
        indices = np.where(mask)[0]
        idx[indices] = rng.permutation(indices)
    return idx


# --- Mantel Tests ---


def _mantel_restricted_worker(
    model_rsm: np.ndarray, data_flat: np.ndarray, strata: np.ndarray, seed: int
) -> float:
    """Restricted permutation: shuffle rows/cols within strata."""
    rng = np.random.default_rng(seed)
    n = model_rsm.shape[0]
    idx = get_stratified_indices(n, strata, rng)
    model_perm = model_rsm[idx, :][:, idx]
    model_flat = squareform(model_perm, checks=False)
    return np.corrcoef(model_flat, data_flat)[0, 1]


def _mantel_unrestricted_worker(
    model_rsm: np.ndarray, data_flat: np.ndarray, seed: int
) -> float:
    """Unrestricted permutation: shuffle all rows/cols freely."""
    rng = np.random.default_rng(seed)
    n = model_rsm.shape[0]
    idx = rng.permutation(n)
    model_perm = model_rsm[idx, :][:, idx]
    model_flat = squareform(model_perm, checks=False)
    return np.corrcoef(model_flat, data_flat)[0, 1]


def mantel_test(
    model_rsm: np.ndarray,
    data_rsm: np.ndarray,
    strata: np.ndarray | None,
    restricted: bool,
    permutations: int = 1000,
    n_jobs: int = 1,
    seed_base: int = 0,
) -> tuple[float, float]:
    """Mantel test with optional restricted permutation."""
    model_flat = squareform(model_rsm, checks=False)
    data_flat = squareform(data_rsm, checks=False)
    obs_stat = np.corrcoef(model_flat, data_flat)[0, 1]

    seeds = range(seed_base, seed_base + permutations)
    if restricted and strata is not None:
        null_stats = Parallel(n_jobs=n_jobs)(
            delayed(_mantel_restricted_worker)(model_rsm, data_flat, strata, s)
            for s in seeds
        )
    else:
        null_stats = Parallel(n_jobs=n_jobs)(
            delayed(_mantel_unrestricted_worker)(model_rsm, data_flat, s)
            for s in seeds
        )

    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_stat) + 1) / (permutations + 1)
    return p_value, obs_stat


# --- SRF Structural Tests ---


def fast_alignment_score(X_std: np.ndarray, W_std: np.ndarray) -> float:
    """Vectorized correlation + Hungarian matching."""
    n = X_std.shape[0]
    corr_matrix = (X_std.T @ W_std) / n
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr_matrix))
    return np.abs(corr_matrix[row_ind, col_ind]).sum()


def stratified_shuffle(
    X_sub: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute rows of X_sub within each stratum group."""
    X_perm = np.empty_like(X_sub)
    unique_strata = np.unique(strata, axis=0)
    for group in unique_strata:
        mask = (strata == group).all(axis=1)
        indices = np.where(mask)[0]
        perm_indices = rng.permutation(indices)
        X_perm[indices] = X_sub[perm_indices]
    return X_perm


def _srf_restricted_worker(
    X_std: np.ndarray, W_std: np.ndarray, strata: np.ndarray, seed: int
) -> float:
    """Restricted permutation of X rows within strata."""
    rng = np.random.default_rng(seed)
    X_perm = stratified_shuffle(X_std, strata, rng)
    return fast_alignment_score(X_perm, W_std)


def _srf_unrestricted_worker(
    X_std: np.ndarray, W_std: np.ndarray, seed: int
) -> float:
    """Unrestricted permutation of X rows."""
    rng = np.random.default_rng(seed)
    X_perm = X_std[rng.permutation(X_std.shape[0])]
    return fast_alignment_score(X_perm, W_std)


def srf_structural_test(
    X_sub: np.ndarray,
    W: np.ndarray,
    strata: np.ndarray | None,
    restricted: bool,
    permutations: int = 1000,
    n_jobs: int = 1,
    seed_base: int = 0,
) -> tuple[float, float]:
    """SRF structural test with optional restricted permutation."""
    X_std = (X_sub - X_sub.mean(0)) / (X_sub.std(0) + 1e-9)
    W_std = (W - W.mean(0)) / (W.std(0) + 1e-9)
    obs_stat = fast_alignment_score(X_std, W_std)

    seeds = range(seed_base, seed_base + permutations)
    if restricted and strata is not None:
        null_stats = Parallel(n_jobs=n_jobs)(
            delayed(_srf_restricted_worker)(X_std, W_std, strata, s) for s in seeds
        )
    else:
        null_stats = Parallel(n_jobs=n_jobs)(
            delayed(_srf_unrestricted_worker)(X_std, W_std, s) for s in seeds
        )

    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_stat) + 1) / (permutations + 1)
    return p_value, obs_stat


# --- Factorial Data ---


def create_factorial_data(
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Create factorial design matrix with column ranges per factor."""
    items = list(it.product(*levels.values()))
    col_ranges, features, col_start = {}, [], 0

    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        features.append(Z)
        col_ranges[factor] = (col_start, col_start + len(lvls))
        col_start += len(lvls)

    X = np.concatenate(features, axis=1)
    return X, col_ranges


# --- Noise Models ---


def add_noise_to_features(X: np.ndarray, snr: float, rng: int) -> np.ndarray:
    """Add noise to features X, then compute RSM."""
    X_noisy = add_positive_noise_with_snr(X, snr, rng=rng)
    return X_noisy @ X_noisy.T


def add_noise_to_rsm(rsm: np.ndarray, snr: float, rng: int) -> np.ndarray:
    """Add symmetric noise directly to RSM."""
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if snr >= 1:
        return rsm.copy()

    n = rsm.shape[0]
    base_noise = rng.standard_normal((n, n))
    base_noise = (base_noise + base_noise.T) / 2

    if snr <= 0:
        noisy = np.clip(base_noise * np.std(rsm), 0, None)
        np.fill_diagonal(noisy, noisy.max())
        return noisy

    snr_target = snr / (1.0 - snr)
    var_signal = np.var(rsm, dtype=float)

    def snr_after_clipping(noise_scale: float) -> float:
        noisy = np.clip(rsm + base_noise * noise_scale, 0, None)
        var_noise = np.var(noisy - rsm, dtype=float)
        return np.inf if var_noise == 0 else var_signal / var_noise

    scale_min, scale_max = 0.0, 1.0
    while snr_after_clipping(scale_max) > snr_target:
        scale_min, scale_max = scale_max, scale_max * 2.0

    for _ in range(100):
        scale_mid = 0.5 * (scale_min + scale_max)
        snr_current = snr_after_clipping(scale_mid)
        if abs(snr_current - snr_target) / snr_target < 1e-4:
            break
        if snr_current > snr_target:
            scale_min = scale_mid
        else:
            scale_max = scale_mid

    noisy_rsm = np.clip(rsm + base_noise * scale_mid, 0, None)
    np.fill_diagonal(noisy_rsm, noisy_rsm.max())
    return noisy_rsm


# --- Main Experiment ---


def run_one_condition(
    snr: float,
    repeat: int,
    noise_model: str,
    perm_type: str,
    X: np.ndarray,
    col_ranges: dict,
    n_permutations: int,
) -> list[dict]:
    """Run all tests for one condition."""
    rank = X.shape[1]
    seed_base = int(snr * 1000 + repeat + hash(noise_model) % 1000 + hash(perm_type) % 1000)

    # Generate RSM based on noise model
    clean_rsm = X @ X.T
    if noise_model == "features":
        measured_rsm = add_noise_to_features(X, snr, rng=seed_base)
    else:  # "rsm"
        measured_rsm = add_noise_to_rsm(clean_rsm, snr, rng=seed_base)

    # Fit SRF
    model = SRF(rank=rank, random_state=seed_base, verbose=False)
    W = model.fit_transform(measured_rsm)

    restricted = perm_type == "restricted"
    results = []

    for i, (factor_name, (col_start, col_end)) in enumerate(col_ranges.items()):
        X_sub = X[:, col_start:col_end]
        non_factor_cols = list(range(0, col_start)) + list(range(col_end, X.shape[1]))
        strata = X[:, non_factor_cols]

        # RSA
        hypothesis_rsm = X_sub @ X_sub.T
        p_rsa, score_rsa = mantel_test(
            hypothesis_rsm,
            measured_rsm,
            strata,
            restricted=restricted,
            permutations=n_permutations,
            n_jobs=1,
            seed_base=seed_base + 100 * i,
        )

        # SRF Structural
        p_srf, score_srf = srf_structural_test(
            X_sub,
            W,
            strata,
            restricted=restricted,
            permutations=n_permutations,
            n_jobs=1,
            seed_base=seed_base + 200 * i,
        )

        base = {
            "snr": snr,
            "repeat": repeat,
            "noise_model": noise_model,
            "perm_type": perm_type,
            "factor": factor_name,
        }
        results.extend([
            {**base, "method": "RSA", "p_value": p_rsa, "score": score_rsa},
            {**base, "method": "SRF", "p_value": p_srf, "score": score_srf},
        ])

    return results


def main():
    OUTPUT_DIR = get_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Factorial design
    levels = {
        "animacy": ["animate", "inanimate"],
        "size": ["small", "medium", "large"],
        "shape": ["round", "angular"],
        "color": ["red", "green", "blue", "yellow"],
    }
    X, col_ranges = create_factorial_data(levels)
    rank = X.shape[1]
    print(f"Factorial: {X.shape[0]} items, {rank} dims, factors: {list(col_ranges.keys())}")

    # Parameters
    snrs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    noise_models = ["features", "rsm"]
    perm_types = ["restricted", "unrestricted"]
    n_repeats = 50
    n_permutations = 1000

    # All conditions
    conditions = [
        (snr, rep, nm, pt)
        for snr in snrs
        for rep in range(n_repeats)
        for nm in noise_models
        for pt in perm_types
    ]
    print(
        f"Running {len(conditions)} conditions "
        f"({len(snrs)} SNRs × {n_repeats} repeats × {len(noise_models)} noise × {len(perm_types)} perm)..."
    )

    # Parallel execution
    all_results_nested = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_condition)(snr, rep, nm, pt, X, col_ranges, n_permutations)
        for snr, rep, nm, pt in conditions
    )

    # Flatten and save
    all_results = [r for batch in all_results_nested for r in batch]
    df = pd.DataFrame(all_results)
    df["significant"] = (df["p_value"] < 0.05).astype(int)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # Compute power
    power_df = (
        df.groupby(["snr", "noise_model", "perm_type", "method"])["significant"]
        .mean()
        .reset_index()
    )
    power_df["power"] = power_df["significant"] * 100

    # --- PLOT 1: 2x2 grid (noise_model × perm_type) ---
    fig, axes = create_figure("full_width", nrows=2, ncols=2, height=4.5)

    for row, noise_model in enumerate(noise_models):
        for col, perm_type in enumerate(perm_types):
            ax = axes[row, col]
            sub = power_df[
                (power_df["noise_model"] == noise_model)
                & (power_df["perm_type"] == perm_type)
            ]

            for method, color, marker in [
                ("RSA", ROSE, "o"),
                ("SRF", TEAL, "s"),
            ]:
                data = sub[sub["method"] == method].sort_values("snr")
                ax.plot(
                    data["snr"],
                    data["power"],
                    f"{marker}-",
                    color=color,
                    label=method,
                    markersize=6,
                    linewidth=1.5,
                )

            ax.axhline(5, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("SNR" if row == 1 else "")
            ax.set_ylabel("Power (%)" if col == 0 else "")
            ax.set_title(f"Noise: {noise_model}, Perm: {perm_type}")
            ax.set_ylim(-5, 105)
            ax.set_xlim(-0.02, 0.52)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right", frameon=False)
            despine(ax)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "power_grid.pdf")

    # --- PLOT 2: RSA vs SRF difference (SRF - RSA) ---
    pivot = power_df.pivot_table(
        values="power",
        index=["snr", "noise_model", "perm_type"],
        columns="method",
    ).reset_index()
    pivot["diff"] = pivot["SRF"] - pivot["RSA"]

    fig, axes = create_figure("wide", ncols=2)

    for col, noise_model in enumerate(noise_models):
        ax = axes[col]
        sub = pivot[pivot["noise_model"] == noise_model]

        for perm_type, color, marker in [
            ("restricted", ROSE, "o"),
            ("unrestricted", TEAL, "s"),
        ]:
            data = sub[sub["perm_type"] == perm_type].sort_values("snr")
            ax.plot(
                data["snr"],
                data["diff"],
                f"{marker}-",
                color=color,
                label=perm_type,
                markersize=6,
                linewidth=1.5,
            )

        ax.axhline(0, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Power diff (SRF - RSA)" if col == 0 else "")
        ax.set_title(f"Noise on {noise_model}")
        ax.set_ylim(-50, 50)
        ax.set_xlim(-0.02, 0.52)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right", frameon=False)
        despine(ax)

    save_figure(fig, OUTPUT_DIR / "power_diff.pdf")

    # --- PLOT 3: Type I error at SNR=0 ---
    snr0 = power_df[power_df["snr"] == 0.0].copy()

    fig, ax = create_figure("single")
    x_pos = np.arange(4)
    width = 0.35

    conditions_order = [
        ("features", "restricted"),
        ("features", "unrestricted"),
        ("rsm", "restricted"),
        ("rsm", "unrestricted"),
    ]
    labels = ["Feat+Restr", "Feat+Unrestr", "RSM+Restr", "RSM+Unrestr"]

    rsa_vals = []
    srf_vals = []
    for nm, pt in conditions_order:
        row = snr0[(snr0["noise_model"] == nm) & (snr0["perm_type"] == pt)]
        rsa_vals.append(row[row["method"] == "RSA"]["power"].values[0])
        srf_vals.append(row[row["method"] == "SRF"]["power"].values[0])

    ax.bar(x_pos - width / 2, rsa_vals, width, label="RSA", color=ROSE)
    ax.bar(x_pos + width / 2, srf_vals, width, label="SRF", color=TEAL)
    ax.axhline(5, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7, label="α=0.05")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Type I Error (%)")
    ax.set_ylim(0, 15)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "type1_error.pdf")

    # Print summary
    print("\n" + "=" * 80)
    print("POWER SUMMARY (% significant at alpha=0.05)")
    print("=" * 80)
    summary = power_df.pivot_table(
        values="power",
        index=["noise_model", "perm_type"],
        columns=["snr", "method"],
    )
    print(summary.round(1))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
