"""Plotting for RSA comparison experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kstest

import matplotlib.pyplot as plt
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure, apply_theme

# Paper style colors
BLUE = "#3498db"
RED = "#e74c3c"


def _save_fig(fig: plt.Figure, path: Path, close: bool = True) -> None:
    """Save figure as both PDF and PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # PDF
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    # PNG
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")

    if close:
        plt.close(fig)


def load_results(csv_path: Path, alpha: float = 0.05) -> pd.DataFrame:
    """Load CSV, applying FDR if not pre-computed."""
    from statsmodels.stats.multitest import multipletests

    df = pd.read_csv(csv_path)

    if "significant" in df.columns:
        return df

    # Backward compatibility: apply FDR if not in CSV
    def add_fdr(g):
        g = g.copy()
        _, corrected, _, _ = multipletests(g["raw_p"], alpha=alpha, method="fdr_bh")
        g["significant"] = corrected < alpha
        return g

    return df.groupby(["snr", "repeat", "method"], group_keys=False).apply(add_fdr)


def plot_power(df: pd.DataFrame, output_path: Path, title: str = ""):
    """Power curve: RSA vs SRF methods."""
    power = df.groupby(["snr", "method"])["significant"].mean().reset_index()

    fig, ax = create_figure("single")
    method_styles = [
        ("RSA", CMAP[2], "^"),
        ("SRF-LOO", CMAP[0], "o"),
        ("SRF-Global", CMAP[1], "s"),
    ]
    for method, color, marker in method_styles:
        m = power[power["method"] == method]
        if m.empty:
            continue
        ax.plot(
            m["snr"],
            m["significant"] * 100,
            f"{marker}-",
            color=color,
            lw=2,
            ms=6,
            label=method,
        )

    ax.axhline(5, color=GRAY["dark"], ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    despine(ax)
    _save_fig(fig, output_path)


def plot_calibration(df: pd.DataFrame, output_path: Path):
    """P-value distribution at SNR=0 (side-by-side panels)."""
    snr0 = df[df["snr"] == 0.0]
    if snr0.empty:
        return

    methods_in_data = [
        m for m in ["RSA", "SRF-LOO", "SRF-Global"] if m in snr0["method"].values
    ]
    ncols = len(methods_in_data)
    if ncols == 0:
        return

    fig, axes = create_figure("full_width" if ncols == 3 else "wide", ncols=ncols)
    if ncols == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 21)
    colors = {"RSA": CMAP[2], "SRF-LOO": CMAP[0], "SRF-Global": CMAP[1]}

    for ax, method in zip(axes, methods_in_data):
        ps = snr0[snr0["method"] == method]["raw_p"]
        if ps.empty:
            continue
        ax.hist(
            ps,
            bins=bins,
            color=colors[method],
            edgecolor="white",
            alpha=0.8,
            density=True,
        )
        ax.axhline(1, color=GRAY["dark"], ls="--", lw=1.5, label="Uniform")
        ax.axvline(0.05, color="red", ls=":", lw=1.5, label="α = 0.05")
        ax.set_xlabel("p-value")
        ax.set_ylabel("Density")
        ax.set_title(f"{method} (SNR=0)")
        ax.legend(fontsize=7)
        ax.set_ylim([0, 2.5])
        despine(ax)

    _save_fig(fig, output_path)


def plot_power_by_variance(df: pd.DataFrame, output_path: Path):
    """Power stratified by dimension variance."""
    if "dim_var" not in df.columns:
        return

    median = df["dim_var"].median()
    df = df.copy()
    df["var_group"] = np.where(df["dim_var"] <= median, "Low", "High")

    fig, axes = create_figure("wide", ncols=2)
    for ax, (method, color) in zip(axes, [("RSA", CMAP[2]), ("SRF-LOO", CMAP[1])]):
        m = df[df["method"] == method]
        for group, ls in [("High", "-"), ("Low", "--")]:
            g = m[m["var_group"] == group]
            power = g.groupby("snr")["significant"].mean() * 100
            ax.plot(
                power.index, power.values, ls, color=color, lw=2, label=f"{group} var"
            )

        ax.axhline(5, color=GRAY["dark"], ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Power (%)")
        ax.set_ylim([-5, 105])
        ax.set_title(method)
        ax.legend(fontsize=8)
        despine(ax)

    _save_fig(fig, output_path)


def plot_variance_effects(df: pd.DataFrame, output_path: Path, snr: float = 1.0):
    """Variance distribution and power vs variance."""
    if "dim_var" not in df.columns:
        return

    high_snr = df[df["snr"] == snr]
    if high_snr.empty:
        return

    fig, axes = create_figure("wide", ncols=2)

    # Variance histogram
    ax = axes[0]
    ax.hist(df["dim_var"], bins=30, color=GRAY["medium"], edgecolor="white", alpha=0.8)
    ax.axvline(df["dim_var"].median(), color=CMAP[0], ls="--", lw=2)
    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Count")
    ratio = df["dim_var"].max() / df["dim_var"].min()
    ax.set_title(f"Variance range: {ratio:.0f}×")
    despine(ax)

    # Power vs variance
    ax = axes[1]
    bins = pd.qcut(high_snr["dim_var"], q=5, duplicates="drop")
    for method, color, marker in [("RSA", CMAP[2], "^"), ("SRF-LOO", CMAP[1], "s")]:
        m = high_snr[high_snr["method"] == method]
        grouped = m.groupby(bins, observed=True)["significant"].mean() * 100
        centers = [iv.mid for iv in grouped.index]
        ax.plot(
            centers, grouped.values, f"{marker}-", color=color, lw=2, ms=6, label=method
        )

    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.set_title(f"Power vs variance (SNR={snr})")
    ax.legend(fontsize=8)
    despine(ax)

    _save_fig(fig, output_path)


def plot_power_by_factor(df: pd.DataFrame, output_path: Path):
    """Power breakdown by factor (for factorial designs).

    This is the key plot showing why LOO fails for 2-column factors:
    - LOO alignment is unstable when there are only 2 columns per factor
    - Global alignment uses all data at once and is stable
    """
    if "factor" not in df.columns:
        return

    methods_in_data = [
        m for m in ["RSA", "SRF-LOO", "SRF-Global"] if m in df["method"].values
    ]
    ncols = len(methods_in_data)
    if ncols == 0:
        return

    fig, axes = create_figure("full_width" if ncols == 3 else "wide", ncols=ncols)
    if ncols == 1:
        axes = [axes]
    colors = {"RSA": CMAP[2], "SRF-LOO": CMAP[0], "SRF-Global": CMAP[1]}
    factors = sorted(df["factor"].unique())

    for ax, method in zip(axes, methods_in_data):
        method_df = df[df["method"] == method]

        for i, factor in enumerate(factors):
            factor_df = method_df[method_df["factor"] == factor]
            power = factor_df.groupby("snr")["significant"].mean() * 100
            n_cols = factor_df["column"].nunique()
            ls = "--" if n_cols == 2 else "-"
            ax.plot(power.index, power.values, ls, lw=2, label=f"{factor} ({n_cols})")

        ax.axhline(5, color=GRAY["dark"], ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Power (%)")
        ax.set_ylim([-5, 105])
        ax.set_title(method)
        ax.legend(fontsize=7, loc="lower right")
        despine(ax)

    _save_fig(fig, output_path)


def print_validation(df: pd.DataFrame) -> None:
    """Print validation summary for the experiment."""
    methods = [m for m in ["RSA", "SRF-LOO", "SRF-Global"] if m in df["method"].values]
    snr0 = df[df["snr"] == 0.0]
    snr1 = (
        df[df["snr"] == 1.0]
        if 1.0 in df["snr"].values
        else df[df["snr"] == df["snr"].max()]
    )

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Calibration
    print()
    print("1. CALIBRATION (SNR=0, should be ≤5%):")
    for method in methods:
        if method not in snr0["method"].values:
            continue
        fp = snr0[snr0["method"] == method]["significant"].mean() * 100
        status = "✓" if fp <= 5 else "✗"
        print(f"   {method}: {fp:.1f}% {status}")

    # Power at high SNR
    print()
    print(f"2. POWER AT SNR={snr1['snr'].iloc[0]:.1f} (should be ~100%):")
    for method in methods:
        if method not in snr1["method"].values:
            continue
        power = snr1[snr1["method"] == method]["significant"].mean() * 100
        status = "✓" if power >= 95 else "~" if power >= 80 else "✗"
        print(f"   {method}: {power:.1f}% {status}")

    # Power by factor (if factorial)
    if "factor" in df.columns:
        print()
        print(f"3. POWER BY FACTOR AT SNR={snr1['snr'].iloc[0]:.1f}:")
        by_factor = (
            snr1.groupby(["factor", "method"])["significant"].mean().unstack() * 100
        )
        # Reorder columns
        cols = [c for c in ["RSA", "SRF-LOO", "SRF-Global"] if c in by_factor.columns]
        by_factor = by_factor[cols]
        print(by_factor.round(1).to_string())

    # P-value uniformity
    print()
    print("4. P-VALUE UNIFORMITY (KS test, SNR=0):")
    for method in methods:
        if method not in snr0["method"].values:
            continue
        ps = snr0[snr0["method"] == method]["raw_p"]
        ks_stat, ks_p = kstest(ps, "uniform")
        status = "✓" if ks_p > 0.05 else "✗"
        print(f"   {method}: KS p-value = {ks_p:.3f} {status}")


def plot_all(csv_path: Path, output_dir: Path | None = None, name: str = ""):
    """Generate all plots for an experiment."""
    output_dir = output_dir or csv_path.parent
    prefix = f"{name}_" if name else ""

    df = load_results(csv_path)
    methods = [m for m in ["RSA", "SRF-LOO", "SRF-Global"] if m in df["method"].values]

    # Core plots
    plot_power(df, output_dir / f"{prefix}power.pdf")
    plot_calibration(df, output_dir / f"{prefix}calibration.pdf")

    # SPOSE-style plots (dimension variance)
    if "dim_var" in df.columns:
        plot_power_by_variance(df, output_dir / f"{prefix}power_by_variance.pdf")
        plot_variance_effects(df, output_dir / f"{prefix}variance_effects.pdf")

    # Factorial-style plots (factors)
    if "factor" in df.columns:
        plot_power_by_factor(df, output_dir / f"{prefix}power_by_factor.pdf")

    # Power summary table
    power = df.groupby(["snr", "method"])["significant"].mean()
    header = f"{'SNR':<6}" + "".join(f" {m:>12}" for m in methods)
    print(f"\n{header}")
    print("-" * len(header))
    for snr in sorted(df["snr"].unique()):
        vals = [power.get((snr, m), 0) * 100 for m in methods]
        row = f"{snr:<6.2f}" + "".join(f" {v:>12.1f}" for v in vals)
        print(row)

    # Validation summary
    print_validation(df)


def plot_power_paper(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str = "",
    show_legend: bool = True,
    srf_method: str = "SRF-LOO",
) -> None:
    """Paper-style power curve on given axes.

    Style: blue SRF, red RSA, circle markers, clean labels.
    """
    power = df.groupby(["snr", "method"])["significant"].mean().reset_index()

    # Significance level reference line (draw first, behind data)
    ax.axhline(5, color=GRAY["medium"], ls="--", lw=1, zorder=1)

    # RSA (red) and SRF (blue)
    styles = [
        ("RSA", RED, "o", "RSA"),
        (srf_method, BLUE, "o", "SRF"),
    ]

    for method, color, marker, label in styles:
        m = power[power["method"] == method]
        if m.empty:
            continue
        ax.plot(
            m["snr"],
            m["significant"] * 100,
            marker=marker,
            color=color,
            lw=1.5,
            ms=5,
            label=label,
            zorder=2,
        )

    ax.set_xlabel("Signal-to-noise ratio")
    ax.set_ylabel("Statistical power (%)")
    ax.set_ylim([-5, 105])
    ax.set_xlim([-0.05, 1.05])

    if show_legend:
        ax.legend(loc="upper left", frameon=False)

    if title:
        ax.set_title(title)

    despine(ax)


def plot_paper_combined(
    factorial_csv: Path,
    spose_csv: Path,
    output_path: Path,
) -> None:
    """Create combined figure with factorial and SPOSE panels (paper style)."""
    apply_theme()

    # Load data
    df_factorial = load_results(factorial_csv)
    df_spose = load_results(spose_csv)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.35)

    # Factorial (use SRF-Global since it's the main comparison)
    plot_power_paper(
        axes[0], df_factorial,
        title="Simulation",
        srf_method="SRF-Global",
    )

    # SPOSE
    plot_power_paper(
        axes[1], df_spose,
        title="SPoSE semantic embedding",
        srf_method="SRF-LOO",
    )

    _save_fig(fig, output_path)


def plot_paper_single(
    csv_path: Path,
    output_path: Path,
    title: str = "",
    srf_method: str = "SRF-LOO",
) -> None:
    """Create single panel paper-style power plot."""
    apply_theme()

    df = load_results(csv_path)
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.8))
    plt.subplots_adjust(left=0.17, right=0.96, bottom=0.18, top=0.88)

    plot_power_paper(ax, df, title=title, srf_method=srf_method)

    _save_fig(fig, output_path)


def plot_linear_combination_explanation(output_path: Path) -> None:
    """Explanatory plot: why RSA has lower power (linear combination problem).

    Shows how individual factor RSMs combine into full S, and why testing
    each factor marginally yields weaker signal than factorizing first.
    """
    import itertools as it

    apply_theme()

    # Create simple factorial: 2 factors × 3 levels each = 9 items
    levels = {"Factor A": ["a1", "a2", "a3"], "Factor B": ["b1", "b2", "b3"]}
    items = list(it.product(*levels.values()))
    n = len(items)

    # One-hot encode
    x_a = np.eye(3)[[["a1", "a2", "a3"].index(item[0]) for item in items]]
    x_b = np.eye(3)[[["b1", "b2", "b3"].index(item[1]) for item in items]]

    # Hypothesis RSMs
    h_a = x_a @ x_a.T
    h_b = x_b @ x_b.T
    s = h_a + h_b  # Combined similarity

    # Correlations
    idx = np.triu_indices(n, k=1)
    r_a = np.corrcoef(h_a[idx], s[idx])[0, 1]
    r_b = np.corrcoef(h_b[idx], s[idx])[0, 1]

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.2))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.3)

    titles = ["H_A (Factor A)", "H_B (Factor B)", "S = H_A + H_B", "RSA correlations"]
    matrices = [h_a, h_b, s, None]

    for ax, title, mat in zip(axes[:3], titles[:3], matrices[:3]):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=2)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Bar plot of correlations
    ax = axes[3]
    bars = ax.bar(["cor(H_A, S)", "cor(H_B, S)"], [r_a, r_b], color=[CMAP[0], CMAP[1]], width=0.6)
    ax.axhline(1.0, color=GRAY["medium"], ls="--", lw=1)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Correlation")
    ax.set_title("RSA signal dilution", fontsize=9)
    for bar, r in zip(bars, [r_a, r_b]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"{r:.2f}",
                ha="center", va="bottom", fontsize=8)
    despine(ax)

    _save_fig(fig, output_path)


def plot_variance_explanation(df: pd.DataFrame, output_path: Path) -> None:
    """Explanatory plot: why dimension variance affects detection power.

    Shows variance distribution and power breakdown by variance quartile.
    """
    if "dim_var" not in df.columns:
        return

    apply_theme()

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.88, wspace=0.35)

    # Left: variance distribution
    ax = axes[0]
    ax.hist(df["dim_var"], bins=30, color=GRAY["medium"], edgecolor="white", alpha=0.8)
    quartiles = df["dim_var"].quantile([0.25, 0.5, 0.75])
    for q, ls in zip(quartiles, [":", "--", ":"]):
        ax.axvline(q, color=CMAP[1], ls=ls, lw=1.5)
    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Count")
    ax.set_title("Variance distribution")
    despine(ax)

    # Right: power by variance quartile at SNR=1.0
    ax = axes[1]
    snr1 = df[df["snr"] == 1.0] if 1.0 in df["snr"].values else df[df["snr"] == df["snr"].max()]

    df_q = snr1.copy()
    df_q["var_q"] = pd.qcut(df_q["dim_var"], q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])

    x_pos = np.arange(4)
    width = 0.35

    for i, (method, color) in enumerate([("RSA", RED), ("SRF-LOO", BLUE)]):
        m = df_q[df_q["method"] == method]
        power = m.groupby("var_q", observed=True)["significant"].mean() * 100
        offset = -width/2 if i == 0 else width/2
        ax.bar(x_pos + offset, power.values, width, color=color, label=method)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    ax.set_xlabel("Variance quartile")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([0, 105])
    ax.set_title("Power by variance (SNR=1.0)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    despine(ax)

    _save_fig(fig, output_path)


def plot_variance_scatter(df: pd.DataFrame, output_path: Path, snr: float = 1.0) -> None:
    """Scatter plot: variance vs power for each dimension."""
    if "dim_var" not in df.columns:
        return

    apply_theme()

    snr_df = df[df["snr"] == snr] if snr in df["snr"].values else df[df["snr"] == df["snr"].max()]

    # Aggregate by dim_var (each unique variance value represents a dimension in a repeat)
    power_by_var = snr_df.groupby(["dim_var", "method"])["significant"].mean().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.88)

    for method, color in [("RSA", RED), ("SRF-LOO", BLUE)]:
        m = power_by_var[power_by_var["method"] == method]
        ax.scatter(m["dim_var"], m["significant"] * 100, color=color, alpha=0.5, s=20, label=method)

    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.set_title(f"Variance vs power (SNR={snr})")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    despine(ax)

    _save_fig(fig, output_path)


def _load_spose_dim_labels() -> list[str]:
    """Load SPOSE dimension labels."""
    from pathlib import Path

    # Find repo root
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        label_path = parent / "data" / "things" / "labels_spose_66d_short.txt"
        if label_path.exists():
            return [line.strip() for line in label_path.read_text().strip().split("\n") if line.strip()]
    return [f"dim_{i}" for i in range(66)]


def plot_dimension_variance_ranking(output_path: Path) -> None:
    """Show which SPOSE dimensions have high vs low variance.

    Loads actual SPOSE embedding and computes variance per dimension.
    """
    from pathlib import Path

    apply_theme()

    # Load SPOSE embedding
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        embed_path = parent / "data" / "things" / "spose_embedding_66d.txt"
        if embed_path.exists():
            x = np.maximum(np.loadtxt(embed_path), 0)
            break
    else:
        return

    labels = _load_spose_dim_labels()

    # Compute variance per dimension
    var_per_dim = np.var(x, axis=0)
    sorted_idx = np.argsort(var_per_dim)

    # Show top 10 and bottom 10
    n_show = 10
    low_idx = sorted_idx[:n_show]
    high_idx = sorted_idx[-n_show:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.88, wspace=0.6)

    # Low variance dimensions
    ax = axes[0]
    y_pos = np.arange(n_show)
    low_labels = [labels[i] if i < len(labels) else f"dim_{i}" for i in low_idx]
    low_vars = var_per_dim[low_idx]
    ax.barh(y_pos, low_vars, color=CMAP[0], height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(low_labels, fontsize=7)
    ax.set_xlabel("Variance")
    ax.set_title("Low variance dimensions", fontsize=10)
    ax.invert_yaxis()
    despine(ax)

    # High variance dimensions
    ax = axes[1]
    high_labels = [labels[i] if i < len(labels) else f"dim_{i}" for i in high_idx]
    high_vars = var_per_dim[high_idx]
    ax.barh(y_pos, high_vars, color=CMAP[1], height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(high_labels, fontsize=7)
    ax.set_xlabel("Variance")
    ax.set_title("High variance dimensions", fontsize=10)
    ax.invert_yaxis()
    despine(ax)

    _save_fig(fig, output_path)


def plot_variance_power_relationship(df: pd.DataFrame, output_path: Path) -> None:
    """Binned scatter plot showing variance-power relationship with trend line."""
    if "dim_var" not in df.columns:
        return

    apply_theme()

    snr1 = df[df["snr"] == 1.0] if 1.0 in df["snr"].values else df[df["snr"] == df["snr"].max()]

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.88)

    # Bin by variance and compute mean power
    n_bins = 10
    for method, color, marker in [("RSA", RED, "o"), ("SRF-LOO", BLUE, "s")]:
        m = snr1[snr1["method"] == method].copy()
        m["var_bin"] = pd.qcut(m["dim_var"], q=n_bins, duplicates="drop")
        grouped = m.groupby("var_bin", observed=True).agg(
            var_mean=("dim_var", "mean"),
            power=("significant", "mean"),
            power_se=("significant", "sem"),
        ).reset_index()

        ax.errorbar(
            grouped["var_mean"], grouped["power"] * 100,
            yerr=grouped["power_se"] * 100 * 1.96,
            marker=marker, color=color, lw=1.5, ms=6, capsize=3, label=method
        )

    ax.axhline(5, color=GRAY["medium"], ls="--", lw=1, zorder=0)
    ax.set_xlabel("Dimension variance")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.set_title("Variance-power relationship (SNR=1.0)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    despine(ax)

    _save_fig(fig, output_path)


def plot_all_with_explanations(csv_path: Path, output_dir: Path | None = None) -> None:
    """Generate all plots including explanatory figures."""
    output_dir = output_dir or csv_path.parent

    df = load_results(csv_path)

    # Standard plots
    plot_power(df, output_dir / "power")
    plot_calibration(df, output_dir / "calibration")

    # SPOSE-specific
    if "dim_var" in df.columns:
        plot_power_by_variance(df, output_dir / "power_by_variance")
        plot_variance_effects(df, output_dir / "variance_effects")
        plot_variance_explanation(df, output_dir / "variance_explanation")
        plot_variance_scatter(df, output_dir / "variance_scatter")
        plot_variance_power_relationship(df, output_dir / "variance_power_relationship")
        plot_dimension_variance_ranking(output_dir / "dimension_variance_ranking")

    # Factorial-specific
    if "factor" in df.columns:
        plot_power_by_factor(df, output_dir / "power_by_factor")
        plot_linear_combination_explanation(output_dir / "linear_combination")

    # Paper-style power plot
    srf_method = "SRF-Global" if "SRF-Global" in df["method"].values else "SRF-LOO"
    title = "Simulation" if "factor" in df.columns else "SPoSE semantic embedding"

    apply_theme()
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.8))
    plt.subplots_adjust(left=0.17, right=0.96, bottom=0.18, top=0.88)
    plot_power_paper(ax, df, title=title, srf_method=srf_method)
    _save_fig(fig, output_dir / "power_paper")
