"""
SRF vs RSA: Factorial Design Comparison.

Three tests for each factor:
1. RSA: Mantel test (unrestricted permutation)
2. SRF-Denoise: corr(WW', S_model) with unrestricted permutation
3. SRF-LOO: leave-one-out prediction of factor labels from W

Uses original 4-factor design: animacy(2) x size(3) x shape(2) x color(4) = 48 items.
"""

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

from pysrf import SRF
from src.utils.helpers import add_positive_noise_with_snr
from src.utils import get_output_dir


def create_factorial_design(
    n_levels_per_factor: list[int],
) -> tuple[np.ndarray, list[slice], np.ndarray]:
    """Create one-hot encoded factorial design matrix (full factorial)."""
    n_factors = len(n_levels_per_factor)
    n_items = int(np.prod(n_levels_per_factor))
    n_cols = sum(n_levels_per_factor)

    X = np.zeros((n_items, n_cols))
    factor_labels = np.zeros((n_items, n_factors), dtype=int)

    factor_slices = []
    col_start = 0
    for n_levels in n_levels_per_factor:
        factor_slices.append(slice(col_start, col_start + n_levels))
        col_start += n_levels

    for item_idx in range(n_items):
        remainder = item_idx
        for f in range(n_factors):
            stride = int(np.prod(n_levels_per_factor[f + 1:])) if f < n_factors - 1 else 1
            level = (remainder // stride) % n_levels_per_factor[f]
            col_idx = factor_slices[f].start + level
            X[item_idx, col_idx] = 1.0
            factor_labels[item_idx, f] = level

    return X, factor_slices, factor_labels


def add_noise_column(X: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Replace signal columns with noise columns based on SNR.

    At SNR=1: all signal columns (X)
    At SNR=0: all noise columns (same shape as X)
    At SNR=0.5: half signal, half noise columns

    This creates rank-deficient noise that properly degrades the factorial structure.
    """
    n, d = X.shape

    if snr >= 1.0:
        return X
    if snr <= 0.0:
        noise = np.abs(rng.standard_normal((n, d)))
        return noise

    # Number of signal columns to keep
    n_signal = max(1, int(np.round(snr * d)))
    n_noise = d - n_signal

    # Keep first n_signal columns of X, replace rest with noise
    # (Shuffle to avoid bias toward early factors)
    col_order = rng.permutation(d)
    signal_cols = col_order[:n_signal]
    noise_cols = col_order[n_signal:]

    X_noisy = np.zeros_like(X)
    X_noisy[:, signal_cols] = X[:, signal_cols]

    # Add noise columns with similar scale
    if n_noise > 0:
        noise = np.abs(rng.standard_normal((n, n_noise)))
        noise = noise / np.std(noise) * np.mean(np.std(X, axis=0))
        X_noisy[:, noise_cols] = noise

    return X_noisy


def add_noise_elementwise(X: np.ndarray, snr: float, seed: int) -> np.ndarray:
    """Add element-wise noise to X to achieve target SNR."""
    return add_positive_noise_with_snr(X, snr, rng=seed)


def correlation_lower_triangle(S1: np.ndarray, S2: np.ndarray) -> float:
    """Pearson correlation between lower triangles."""
    v1 = squareform(S1, checks=False)
    v2 = squareform(S2, checks=False)

    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        return 0.0

    return float(np.corrcoef(v1, v2)[0, 1])


def fit_srf(S: np.ndarray, rank: int, seed: int = 42) -> np.ndarray:
    """Fit SRF and return embedding W."""
    S_nn = np.maximum(S, 0)
    return SRF(rank=rank, random_state=seed).fit_transform(S_nn)


def _rsa_permutation_worker(
    S_data_flat: np.ndarray,
    S_model: np.ndarray,
    seed: int,
) -> float:
    """Single RSA permutation: permute model RSM rows/cols (unrestricted)."""
    rng = np.random.default_rng(seed)
    n = S_model.shape[0]
    perm = rng.permutation(n)
    S_model_perm = S_model[perm][:, perm]
    S_model_perm_flat = squareform(S_model_perm, checks=False)
    return float(np.corrcoef(S_data_flat, S_model_perm_flat)[0, 1])


def rsa_test(
    S_data: np.ndarray,
    S_model: np.ndarray,
    n_perm: int,
    seed_base: int,
    n_jobs: int = -1,
) -> dict:
    """RSA Mantel test with unrestricted permutation (standard approach)."""
    r_obs = correlation_lower_triangle(S_data, S_model)
    S_data_flat = squareform(S_data, checks=False)

    seeds = range(seed_base, seed_base + n_perm)
    r_null = Parallel(n_jobs=n_jobs)(
        delayed(_rsa_permutation_worker)(S_data_flat, S_model, s)
        for s in seeds
    )
    r_null = np.array(r_null)

    p_value = (np.sum(r_null >= r_obs) + 1) / (n_perm + 1)
    return {"r_obs": r_obs, "p_value": p_value, "r_null": r_null}


def _denoise_permutation_worker(
    S_recon_flat: np.ndarray,
    S_model: np.ndarray,
    seed: int,
) -> float:
    """Single SRF-Denoise permutation: permute model RSM (unrestricted)."""
    rng = np.random.default_rng(seed)
    n = S_model.shape[0]
    perm = rng.permutation(n)
    S_model_perm = S_model[perm][:, perm]
    S_model_perm_flat = squareform(S_model_perm, checks=False)
    return float(np.corrcoef(S_recon_flat, S_model_perm_flat)[0, 1])


def srf_denoise_test(
    S_reconstructed: np.ndarray,
    S_model: np.ndarray,
    n_perm: int,
    seed_base: int,
    n_jobs: int = -1,
) -> dict:
    """SRF Denoise test: corr(WW', S_model) with unrestricted permutation."""
    r_obs = correlation_lower_triangle(S_reconstructed, S_model)
    S_recon_flat = squareform(S_reconstructed, checks=False)

    seeds = range(seed_base, seed_base + n_perm)
    r_null = Parallel(n_jobs=n_jobs)(
        delayed(_denoise_permutation_worker)(S_recon_flat, S_model, s)
        for s in seeds
    )
    r_null = np.array(r_null)

    p_value = (np.sum(r_null >= r_obs) + 1) / (n_perm + 1)
    return {"r_obs": r_obs, "p_value": p_value, "r_null": r_null}


def loo_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Leave-one-out prediction of X from W using linear regression."""
    n = W.shape[0]
    d = X.shape[1]
    X_pred = np.zeros((n, d))

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        B, _, _, _ = np.linalg.lstsq(W[mask], X[mask], rcond=None)
        X_pred[i] = W[i] @ B

    return X_pred


def _loo_permutation_worker(
    X_pred_factor: np.ndarray,
    X_true: np.ndarray,
    factor_slice: slice,
    seed: int,
) -> float:
    """Single SRF-LOO permutation: permute true labels (unrestricted)."""
    rng = np.random.default_rng(seed)
    n = X_true.shape[0]
    perm = rng.permutation(n)
    X_true_perm = X_true[perm][:, factor_slice]
    return float(np.corrcoef(X_pred_factor.ravel(), X_true_perm.ravel())[0, 1])


def srf_loo_test(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    factor_slice: slice,
    n_perm: int,
    seed_base: int,
    n_jobs: int = -1,
) -> dict:
    """SRF-LOO test: corr(X_pred, X_true) per factor with unrestricted permutation."""
    pred_factor = X_pred[:, factor_slice]
    true_factor = X_true[:, factor_slice]

    r_obs = float(np.corrcoef(pred_factor.ravel(), true_factor.ravel())[0, 1])

    seeds = range(seed_base, seed_base + n_perm)
    r_null = Parallel(n_jobs=n_jobs)(
        delayed(_loo_permutation_worker)(pred_factor, X_true, factor_slice, s)
        for s in seeds
    )
    r_null = np.array(r_null)

    p_value = (np.sum(r_null >= r_obs) + 1) / (n_perm + 1)
    return {"r_obs": r_obs, "p_value": p_value, "r_null": r_null}


def run_single_rep(
    X: np.ndarray,
    factor_slices: list[slice],
    factor_labels: np.ndarray,
    snr: float,
    srf_rank: int,
    n_perm: int,
    seed: int,
    noise_model: str = "column",
) -> dict:
    """Run all tests for a single repetition."""
    rng = np.random.default_rng(seed)

    if noise_model == "column":
        X_noisy = add_noise_column(X, snr, rng)
    else:
        X_noisy = add_noise_elementwise(X, snr, seed)

    S = X_noisy @ X_noisy.T

    W = fit_srf(S, srf_rank, seed)
    S_recon = W @ W.T
    X_pred = loo_predict(W, X)

    results = {"rsa": [], "denoise": [], "loo": []}

    for f, sl in enumerate(factor_slices):
        X_factor = X[:, sl]
        S_model = X_factor @ X_factor.T

        res_rsa = rsa_test(S, S_model, n_perm, seed + 1000 * f, n_jobs=1)
        res_denoise = srf_denoise_test(S_recon, S_model, n_perm, seed + 2000 * f, n_jobs=1)
        res_loo = srf_loo_test(X_pred, X, sl, n_perm, seed + 3000 * f, n_jobs=1)

        results["rsa"].append({"factor": f, **res_rsa})
        results["denoise"].append({"factor": f, **res_denoise})
        results["loo"].append({"factor": f, **res_loo})

    return results


def run_power_simulation(
    n_levels_per_factor: list[int],
    factor_weights: list[float],
    snr_levels: np.ndarray,
    srf_rank: int | None = None,
    n_reps: int = 100,
    n_perm: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    noise_model: str = "column",
    verbose: bool = True,
) -> dict:
    """Run power simulation comparing all three tests."""
    X, factor_slices, factor_labels = create_factorial_design(n_levels_per_factor)
    n, d = X.shape
    n_factors = len(factor_slices)

    if srf_rank is None:
        srf_rank = d

    X_weighted = np.zeros_like(X)
    for f, (sl, w) in enumerate(zip(factor_slices, factor_weights)):
        X_weighted[:, sl] = X[:, sl] * w

    if verbose:
        print(f"Design: {n} items, {d} columns, rank {srf_rank}")
        print(f"Weights: {factor_weights}, Noise model: {noise_model}")

    n_snr = len(snr_levels)
    results = {
        "power_rsa": np.zeros((n_snr, n_factors)),
        "power_denoise": np.zeros((n_snr, n_factors)),
        "power_loo": np.zeros((n_snr, n_factors)),
    }

    for si, snr in enumerate(snr_levels):
        if verbose:
            print(f"  SNR={snr:.2f} ({si + 1}/{n_snr})...")

        rep_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_single_rep)(
                X_weighted,
                factor_slices,
                factor_labels,
                snr,
                srf_rank,
                n_perm,
                seed + rep * 10000 + si * 100000,
                noise_model,
            )
            for rep in range(n_reps)
        )

        for method in ["rsa", "denoise", "loo"]:
            sig = np.zeros((n_reps, n_factors))
            for rep, res in enumerate(rep_results):
                for f_res in res[method]:
                    f = f_res["factor"]
                    sig[rep, f] = f_res["p_value"] < 0.05

            results[f"power_{method}"][si] = sig.mean(axis=0)

    results.update({
        "snr_levels": snr_levels,
        "n_levels_per_factor": n_levels_per_factor,
        "factor_weights": factor_weights,
        "srf_rank": srf_rank,
        "noise_model": noise_model,
    })

    return results


def plot_power_comparison(
    results_column: dict,
    results_elementwise: dict,
    factor_names: list[str] | None = None,
):
    """Plot power comparison between noise models."""
    from src.colors import CYCLE, GRAY_DARK
    from src.utils.figure_theme import create_figure, despine

    snr = results_column["snr_levels"]
    weights = results_column["factor_weights"]
    n_factors = results_column["power_rsa"].shape[1]

    if factor_names is None:
        factor_names = [f"F{i + 1} (w={weights[i]})" for i in range(n_factors)]

    # 2 rows (noise models) x 3 cols (methods)
    fig, axes = create_figure("full_width", nrows=2, ncols=3)

    method_keys = ["power_rsa", "power_denoise", "power_loo"]
    method_titles = ["RSA", "SRF-Denoise", "SRF-LOO"]

    for row, (results, noise_label) in enumerate([
        (results_column, "Column noise"),
        (results_elementwise, "Element-wise noise"),
    ]):
        for col, (key, title) in enumerate(zip(method_keys, method_titles)):
            ax = axes[row, col]
            for f in range(n_factors):
                ax.plot(
                    snr, results[key][:, f] * 100, "-o",
                    color=CYCLE[f % len(CYCLE)], lw=1.5, ms=3, label=factor_names[f],
                )
            ax.axhline(5, color=GRAY_DARK, ls="--", lw=1, alpha=0.7)
            ax.set_xlabel("SNR" if row == 1 else "")
            ax.set_ylabel("Power (%)" if col == 0 else "")
            ax.set_title(f"{title}" if row == 0 else "")
            ax.set_ylim([-5, 105])
            ax.set_xlim([-0.05, 1.05])
            if row == 0 and col == 2:
                ax.legend(fontsize=5, loc="lower right", frameon=False)
            if col == 0:
                ax.text(-0.35, 0.5, noise_label, transform=ax.transAxes,
                        fontsize=8, va="center", ha="center", rotation=90)
            despine(ax)

    return fig


def run_experiment(
    n_levels: list[int],
    weights: list[float],
    snr_levels: np.ndarray,
    output_dir: Path,
    label: str,
    n_reps: int = 30,
    n_perm: int = 200,
):
    """Run full experiment with both noise models."""
    from src.utils.figure_theme import save_figure
    import json

    print(f"\n{'=' * 70}")
    print(f"Experiment: {label}")
    print(f"Weights: {weights}")
    print(f"{'=' * 70}")

    results = {}

    for noise_model in ["column", "elementwise"]:
        print(f"\nRunning {noise_model} noise model...")
        results[noise_model] = run_power_simulation(
            n_levels_per_factor=n_levels,
            factor_weights=weights,
            snr_levels=snr_levels,
            n_reps=n_reps,
            n_perm=n_perm,
            seed=42,
            n_jobs=-1,
            noise_model=noise_model,
        )

    # Save raw results
    results_serializable = {}
    for nm, res in results.items():
        results_serializable[nm] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in res.items()
        }
    with open(output_dir / f"{label}_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Plot
    factor_names = [f"F{i+1} (w={weights[i]})" for i in range(len(weights))]
    fig = plot_power_comparison(results["column"], results["elementwise"], factor_names)
    save_figure(fig, output_dir / f"{label}_power.pdf")

    return results


if __name__ == "__main__":
    from datetime import datetime

    OUTPUT_DIR = get_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SRF vs RSA: Factorial Design Comparison")
    print("=" * 70)
    print("\nComparing:")
    print("  - Two noise models: column vs element-wise")
    print("  - Two weight schemes: differential vs equal")
    print()

    # Original 4-factor design: animacy(2) x size(3) x shape(2) x color(4) = 48 items
    n_levels = [2, 3, 2, 4]  # 48 items, 11 columns
    factor_names_base = ["animacy", "size", "shape", "color"]
    snr_levels = np.linspace(0, 1, 6)

    # Use 200 reps for stable Type I error (~1.5% SE at 5%)
    n_reps = 200
    n_perm = 500

    # Experiment 1: Equal weights
    weights_equal = [1.0, 1.0, 1.0, 1.0]
    run_experiment(
        n_levels, weights_equal, snr_levels, OUTPUT_DIR,
        label="equal_weights",
        n_reps=n_reps,
        n_perm=n_perm,
    )

    # Experiment 2: Differential weights (minimum w=0.6 for detectable effect)
    weights_differential = [1.0, 0.7, 0.6, 0.8]
    run_experiment(
        n_levels, weights_differential, snr_levels, OUTPUT_DIR,
        label="differential_weights",
        n_reps=n_reps,
        n_perm=n_perm,
    )

    print(f"\n{'=' * 70}")
    print(f"All results saved to {OUTPUT_DIR}")
    print("=" * 70)
