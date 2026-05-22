"""Theory and SRF performance plots for simulation experiments.

Generates:
- alpha_factors_{alpha}.pdf     (4 plots) -- factor matrices W at different alpha
- alpha_rsm_{alpha}.pdf         (4 plots) -- similarity matrices at different alpha
- dirichlet_properties.pdf      (1 plot)  -- entropy + sparsity vs alpha
- srf_performance.pdf           (1 plot)  -- reconstruction R² + factor recovery vs alpha
- stability.pdf                 (1 plot)  -- cophenetic correlation vs alpha

Usage:
    poetry run python experiments/analyses/simulation/interpretability/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.colors import ROSE, TEAL, INDIGO, GRAY, GRAY_LIGHT, GRAY_PALE, setup_style
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils.simulation import simulation_dirichlet

OUTPUT_DIR = Path(__file__).parent / "outputs"

N = 300
K = 30
ALPHAS = np.logspace(-1, 3, 150)


def _hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer's sparsity measure. 0 = uniform, 1 = maximally sparse."""
    n = len(x)
    l1 = np.sum(np.abs(x))
    l2 = np.sqrt(np.sum(x**2))
    if l2 == 0:
        return 0.0
    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)


def plot_alpha_factors(output_dir: Path) -> None:
    """Plot factor matrices W for different alpha values."""
    alphas = [0.1, 1.0, 10.0, 100.0]
    rng = np.random.default_rng(42)

    for alpha in alphas:
        factors = simulation_dirichlet(N, K, alpha, rng)
        fig, ax = create_figure("single")

        x = np.arange(K + 1)
        y = np.arange(N + 1)
        im = ax.pcolormesh(x, y, factors, cmap="RdYlBu_r", vmin=0, vmax=1, rasterized=False)
        ax.set_xlim(0, K)
        ax.set_ylim(N, 0)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)

        alpha_str = str(alpha).replace(".", "_")
        save_figure(fig, output_dir / f"alpha_factors_{alpha_str}.pdf")


def plot_alpha_rsm(output_dir: Path) -> None:
    """Plot RSMs for different alpha values."""
    alphas = [0.1, 1.0, 10.0, 100.0]
    rng = np.random.default_rng(42)

    for alpha in alphas:
        factors = simulation_dirichlet(N, K, alpha, rng)
        rsm = factors @ factors.T

        fig, ax = create_figure("single")
        vmin, vmax = rsm.min(), rsm.max()

        x = np.arange(N + 1)
        y = np.arange(N + 1)
        im = ax.pcolormesh(x, y, rsm, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, rasterized=False)
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        alpha_str = str(alpha).replace(".", "_")
        save_figure(fig, output_dir / f"alpha_rsm_{alpha_str}.pdf")


def plot_dirichlet_properties(output_dir: Path) -> None:
    """Plot entropy and Hoyer sparsity vs alpha (dual y-axis)."""
    n_samples = 50
    rng = np.random.default_rng(42)
    max_entropy = np.log(K)

    entropy_mean = np.zeros(len(ALPHAS))
    entropy_sem = np.zeros(len(ALPHAS))
    sparsity_mean = np.zeros(len(ALPHAS))
    sparsity_sem = np.zeros(len(ALPHAS))

    for i, alpha in enumerate(ALPHAS):
        w = rng.dirichlet([alpha] * K, size=n_samples * N)
        ent = -np.sum(w * np.log(w + 1e-10), axis=1) / max_entropy
        ent = ent.reshape(n_samples, N).mean(axis=1)
        entropy_mean[i] = ent.mean()
        entropy_sem[i] = ent.std() / np.sqrt(n_samples)

        l1 = np.sum(np.abs(w), axis=1)
        l2 = np.sqrt(np.sum(w**2, axis=1))
        hoyer = ((np.sqrt(K) - l1 / (l2 + 1e-10)) / (np.sqrt(K) - 1))
        hoyer = hoyer.reshape(n_samples, N).mean(axis=1)
        sparsity_mean[i] = hoyer.mean()
        sparsity_sem[i] = hoyer.std() / np.sqrt(n_samples)

    fig, ax1 = create_figure("single", pad_left=0.7, pad_right=0.7)
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]

    ax1.axhline(1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax1.text(0.12, 0.96, "uniform", fontsize=7, color=GRAY, ha="left")

    ax1.plot(ALPHAS, entropy_mean, "-", color=ROSE)
    ax1.fill_between(ALPHAS, entropy_mean - entropy_sem, entropy_mean + entropy_sem,
                     color=ROSE, alpha=0.2, linewidth=0)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Complexity ($\alpha$)")
    ax1.set_ylabel("Normalized entropy", color=ROSE)
    ax1.set_xlim(0.08, 1200)
    ax1.set_ylim(0, 1.08)
    ax1.set_yticks(tick_positions)
    ax1.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = ax1.twinx()

    ax2.plot(ALPHAS, sparsity_mean, "-", color=TEAL)
    ax2.fill_between(ALPHAS, sparsity_mean - sparsity_sem, sparsity_mean + sparsity_sem,
                     color=TEAL, alpha=0.2, linewidth=0)
    ax2.set_ylabel("Hoyer sparsity", color=TEAL)
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks(tick_positions)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)

    save_figure(fig, output_dir / "dirichlet_properties.pdf")


def _compute_srf_performance() -> pd.DataFrame:
    """Compute factor alignment vs complexity at multiple SNR levels.

    Returns aggregated DataFrame with columns: alpha, snr, mean, sem.
    Caches to CSV to avoid recomputation.
    """
    cache = OUTPUT_DIR / "srf_performance.csv"
    if cache.exists():
        return pd.read_csv(cache)

    from joblib import Parallel, delayed
    from pysrf import SRF

    from src.tools.rsa import loo_alignment, _batch_corr_columns
    from src.utils.helpers import add_positive_noise_with_snr

    n_reps = 200
    snrs = [1.0, 0.7, 0.5, 0.3]

    def run_single(alpha, snr, seed):
        rng = np.random.default_rng(seed)
        factors = simulation_dirichlet(N, K, alpha, rng)
        similarity = factors @ factors.T
        if snr < 1.0:
            similarity = add_positive_noise_with_snr(similarity, snr, rng)
            similarity = (similarity + similarity.T) * 0.5

        srf = SRF(rank=K, random_state=seed)
        srf.fit(similarity)

        w_aligned = loo_alignment(srf.w_, factors)
        r_obs = _batch_corr_columns(w_aligned, factors)
        return {"alpha": alpha, "snr": snr, "r_mean": float(np.nanmean(r_obs))}

    conditions = [(a, s, seed) for a in ALPHAS for s in snrs for seed in range(n_reps)]
    results = Parallel(n_jobs=-1)(delayed(run_single)(*c) for c in conditions)

    df = pd.DataFrame(results)
    agg = df.groupby(["alpha", "snr"]).agg(
        mean=("r_mean", "mean"), sem=("r_mean", "sem"),
    ).reset_index()
    agg.to_csv(cache, index=False)
    return agg


def panel_srf_performance(ax: plt.Axes, agg: pd.DataFrame | None = None) -> None:
    """Factor alignment vs complexity on a given axis."""
    if agg is None:
        agg = _compute_srf_performance()

    snrs = sorted(agg["snr"].unique(), reverse=True)
    colors = {1.0: ROSE, 0.7: TEAL, 0.5: INDIGO, 0.3: GRAY}

    for snr in snrs:
        sub = agg[agg["snr"] == snr].sort_values("alpha")
        label = "No noise" if snr == 1.0 else f"SNR = {snr}"
        ax.plot(sub["alpha"], sub["mean"], "-", color=colors[snr], label=label)
        ax.fill_between(sub["alpha"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], color=colors[snr], alpha=0.2, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Complexity ($\alpha$)")
    ax.set_ylabel(r"Factor alignment ($r$)")
    ax.set_xlim(0.08, 1200)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    despine(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=7)


def plot_srf_performance(output_dir: Path) -> None:
    """Standalone figure: factor alignment vs complexity."""
    agg = _compute_srf_performance()
    fig, ax = create_figure("single")
    panel_srf_performance(ax, agg)
    save_figure(fig, output_dir / "srf_performance.pdf")


def plot_stability(output_dir: Path) -> None:
    """Plot stability (cophenetic correlation) vs alpha."""
    from joblib import Parallel, delayed
    from pysrf import SRF
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import squareform

    n_stability_runs = 20

    def run_stability(alpha, seed):
        def normalize(w):
            return w / (w.sum(axis=1, keepdims=True) + 1e-10)

        rng = np.random.default_rng(seed)
        factors = simulation_dirichlet(N, K, alpha, rng)
        similarity = factors @ factors.T

        consensus_sum = np.zeros((N, N))
        for i in range(n_stability_runs):
            srf = SRF(rank=K, random_state=seed + i)
            w = srf.fit_transform(similarity)
            assignments = np.argmax(normalize(w), axis=1)
            connectivity = (assignments[:, None] == assignments[None, :]).astype(float)
            consensus_sum += connectivity

        consensus = consensus_sum / n_stability_runs
        np.fill_diagonal(consensus, 1.0)
        condensed = squareform(1.0 - consensus, checks=False)
        linkage_matrix = linkage(condensed, method="average")
        coph, _ = cophenet(linkage_matrix, condensed)
        return {"alpha": alpha, "cophenetic": float(coph)}

    results = Parallel(n_jobs=-1)(
        delayed(run_stability)(alpha, int(alpha * 1000) + rep)
        for alpha in ALPHAS
        for rep in range(20)
    )

    df = pd.DataFrame(results).groupby("alpha").agg(
        mean=("cophenetic", "mean"), sem=("cophenetic", "sem")
    ).reset_index()

    fig, ax = create_figure("single")
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]

    ax.axhline(1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.plot(df["alpha"], df["mean"], "-", color=TEAL)
    ax.fill_between(df["alpha"], df["mean"] - df["sem"], df["mean"] + df["sem"],
                    color=TEAL, alpha=0.2, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"Complexity ($\alpha$)")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_xlim(0.08, 120)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(tick_positions)
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    despine(ax)

    save_figure(fig, output_dir / "stability.pdf")


def _compute_imputation_recovery() -> pd.DataFrame:
    """Compute factor recovery under missingness.

    Returns aggregated DataFrame with columns: obs_per_dof, method, mean, sem.
    Caches to CSV to avoid recomputation.
    """
    cache = OUTPUT_DIR / "imputation_recovery.csv"
    if cache.exists():
        return pd.read_csv(cache)

    from joblib import Parallel, delayed
    from pysrf import SRF
    from sklearn.impute import KNNImputer, SimpleImputer

    from src.tools.rsa import loo_alignment, _batch_corr_columns
    from src.utils.helpers import add_positive_noise_with_snr

    k = 5
    dir_alpha = 0.5
    snr = 0.7
    n_reps = 200
    fractions = np.geomspace(0.034, 0.75, 100)

    def _mask_similarity(matrix, fraction_retained, rng):
        n = matrix.shape[0]
        triu_i, triu_j = np.triu_indices(n, k=1)
        total = triu_i.shape[0]
        n_missing = int((1 - fraction_retained) * total)
        observed = matrix.copy()
        if n_missing > 0:
            choice = rng.choice(total, size=n_missing, replace=False)
            observed[triu_i[choice], triu_j[choice]] = np.nan
            observed[triu_j[choice], triu_i[choice]] = np.nan
        return observed

    def _adaptive_rho(obs_per_dof):
        if obs_per_dof <= 1.0:
            return 0.01
        elif obs_per_dof <= 2.0:
            return 0.01 + (obs_per_dof - 1.0) * 0.04
        elif obs_per_dof <= 5.0:
            return 0.05 + (obs_per_dof - 2.0) / 3.0 * 0.45
        return min(3.0, 0.5 + (obs_per_dof - 5.0) / 10.0 * 2.5)

    def run_single(fraction, method, seed):
        rng = np.random.default_rng(seed)
        factors = simulation_dirichlet(N, k, dir_alpha, rng)
        similarity = factors @ factors.T
        similarity = add_positive_noise_with_snr(similarity, snr, rng)
        similarity = (similarity + similarity.T) * 0.5

        n_observed = int(fraction * N * (N - 1) / 2)
        obs_per_dof = n_observed / (N * k)

        if fraction < 1.0:
            observed = _mask_similarity(similarity, fraction, np.random.default_rng(seed + 5000))
        else:
            observed = similarity

        if method == "SRF":
            rho = _adaptive_rho(obs_per_dof)
            srf = SRF(rank=k, random_state=seed, missing_values=np.nan,
                       max_outer=200, max_inner=500, tol=1e-5, rho=rho)
            srf.fit(observed)
            w_hat = srf.w_
        elif method == "KNN":
            filled = KNNImputer(n_neighbors=5).fit_transform(observed)
            filled = (filled + filled.T) * 0.5
            srf = SRF(rank=k, random_state=seed)
            srf.fit(filled)
            w_hat = srf.w_
        else:
            filled = SimpleImputer(strategy="median").fit_transform(observed)
            filled = (filled + filled.T) * 0.5
            srf = SRF(rank=k, random_state=seed)
            srf.fit(filled)
            w_hat = srf.w_

        w_aligned = loo_alignment(w_hat, factors)
        r_obs = _batch_corr_columns(w_aligned, factors)
        return {"fraction": fraction, "obs_per_dof": obs_per_dof, "method": method,
                "r_mean": float(np.nanmean(r_obs))}

    methods = ["SRF", "KNN", "Median"]
    conditions = [(f, m, s) for f in fractions for m in methods for s in range(n_reps)]
    results = Parallel(n_jobs=-1)(delayed(run_single)(*c) for c in conditions)

    df = pd.DataFrame(results)
    agg = df.groupby(["obs_per_dof", "method"]).agg(
        mean=("r_mean", "mean"), sem=("r_mean", "sem"),
    ).reset_index()
    agg.to_csv(cache, index=False)
    return agg


def panel_imputation_recovery(ax: plt.Axes, agg: pd.DataFrame | None = None) -> None:
    """Factor recovery under missingness on a given axis."""
    from matplotlib.ticker import FixedLocator, NullLocator

    if agg is None:
        agg = _compute_imputation_recovery()

    colors = {"SRF": ROSE, "KNN": TEAL, "Median": INDIGO}

    ax.axvspan(0, 1.0, color=GRAY_PALE, zorder=0)
    ax.axvline(x=1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.85, 0.50, "Underdetermined", rotation=90, va="center", ha="center",
            fontsize=7, color=GRAY)

    for method in ["SRF", "KNN", "Median"]:
        sub = agg[agg["method"] == method].sort_values("obs_per_dof")
        ax.plot(sub["obs_per_dof"], sub["mean"], "-", color=colors[method],
                label=method, zorder=3)
        ax.fill_between(sub["obs_per_dof"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], color=colors[method], alpha=0.2, linewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per model parameter")
    ax.set_ylabel(r"Factor recovery ($r$)")
    ax.set_xlim(0.7, 25)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7)


def plot_imputation_recovery(output_dir: Path) -> None:
    """Standalone figure: factor recovery under missingness."""
    agg = _compute_imputation_recovery()
    fig, ax = create_figure("single")
    panel_imputation_recovery(ax, agg)
    save_figure(fig, output_dir / "imputation_recovery.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    plot_alpha_factors(OUTPUT_DIR)
    plot_alpha_rsm(OUTPUT_DIR)
    plot_dirichlet_properties(OUTPUT_DIR)
    print("Theory: 9 plots")

    print("Computing SRF validation plots...")
    plot_srf_performance(OUTPUT_DIR)
    plot_stability(OUTPUT_DIR)
    plot_imputation_recovery(OUTPUT_DIR)
    print("SRF validation: 3 plots")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
