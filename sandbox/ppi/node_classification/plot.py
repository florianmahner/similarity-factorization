#!/usr/bin/env python3
"""Plot node classification benchmark results with publication-quality figures."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


def load_cfg_plot_dir() -> Path | None:
    """Load plot input directory from config.yaml if it exists."""
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    if "plot_input_dir" in cfg and cfg.plot_input_dir:
        return Path(cfg.plot_input_dir)
    return None


def latest_results_dir(base: Path) -> Path | None:
    """Find the most recently modified directory containing benchmark_results.csv."""
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.rglob("benchmark_results.csv") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def resolve_results_dir(args_dir: Path | None) -> Path:
    """
    Resolve results directory from multiple sources in priority order:
    1. Explicit --dir argument
    2. Config file plot_input_dir
    3. Latest results in ./outputs
    """
    script_dir = Path(__file__).resolve().parent
    cfg_dir = load_cfg_plot_dir()
    script_outputs = script_dir / "outputs"

    if args_dir:
        candidates = [args_dir]
        if not args_dir.is_absolute():
            candidates.append(script_dir / args_dir)
            candidates.append(Path.cwd() / args_dir)

        for cand in candidates:
            if (cand / "benchmark_results.csv").exists():
                return cand
        return args_dir

    if cfg_dir:
        return cfg_dir

    latest = latest_results_dir(script_outputs)
    if latest:
        return latest

    raise FileNotFoundError(
        "No results directory found. Pass --dir or set plot_input_dir in config.yaml."
    )


def identify_metric_columns(df: pd.DataFrame) -> list[str]:
    """Identify metric columns (numeric columns that aren't parameters)."""
    param_cols = {"Nodes", "Rank", "Bin", "Method"}
    metric_cols = [
        col for col in df.columns
        if col not in param_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    return metric_cols


def melt_metrics(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Reshape dataframe to long format with all metrics."""
    possible_id_vars = ["Nodes", "Rank", "Bin", "Method"]
    id_vars = [col for col in possible_id_vars if col in df.columns]
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score"
    )
    return df_long


def setup_plot_style() -> None:
    """Configure seaborn plotting style."""
    sns.set_theme(style="ticks", context="talk")


def save_figure(outdir: Path, filename: str) -> None:
    """Save figure as PDF with tight layout."""
    plt.savefig(outdir / filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {outdir / filename}")


def plot_method_comparison_facet(
    df: pd.DataFrame,
    outdir: Path,
    default_rank: int,
) -> None:
    """FacetGrid comparing methods across bins, faceted by N and Metric."""
    metric_cols = identify_metric_columns(df)
    if not metric_cols:
        return

    df_long = melt_metrics(df[df["Rank"] == default_rank], metric_cols)
    if df_long.empty:
        return

    g = sns.FacetGrid(
        df_long,
        col="Nodes",
        row="Metric",
        height=4,
        aspect=1.2,
        sharex=True,
        sharey="row",
    )
    g.map_dataframe(sns.barplot, x="Bin", y="Score", hue="Method", palette="viridis")
    g.add_legend()
    g.set_axis_labels("Label Sparsity Bin", "Score")
    g.set_titles(col_template="N={col_name}", row_template="{row_name}")
    g.figure.suptitle(f"Method Comparison (Rank={default_rank})", y=1.02, fontsize=14)

    for ax in g.axes.flat:
        sns.despine(ax=ax)

    save_figure(outdir, "method_comparison_facet.pdf")


def aggregate_across_bins(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Average metrics across bins to reduce visual clutter in scaling plots."""
    groupby_cols = ["Nodes", "Rank", "Method"]
    agg_dict = {col: "mean" for col in metric_cols}
    df_agg = df.groupby(groupby_cols, as_index=False).agg(agg_dict)
    return df_agg


def plot_scaling_rank_facet(
    df: pd.DataFrame,
    outdir: Path,
) -> None:
    """FacetGrid showing rank scaling, faceted by N and Metric (averaged across bins)."""
    if df["Rank"].nunique() <= 1:
        return

    metric_cols = identify_metric_columns(df)
    if not metric_cols:
        return

    df_agg = aggregate_across_bins(df, metric_cols)
    df_long = melt_metrics(df_agg, metric_cols)
    if df_long.empty:
        return

    g = sns.FacetGrid(
        df_long,
        col="Nodes",
        row="Metric",
        height=3.5,
        aspect=1.3,
        sharex=True,
        sharey=False,
        margin_titles=True,
    )
    g.map_dataframe(
        sns.lineplot,
        x="Rank",
        y="Score",
        hue="Method",
        markers=True,
        markersize=8,
        linewidth=2.5,
        palette="Set2",
    )
    g.add_legend(title="Method")
    g.set_axis_labels("Embedding Rank", "Score")
    g.set_titles(col_template="N={col_name}", row_template="{row_name}")
    g.figure.suptitle("Rank Scaling (averaged across bins)", y=1.01, fontsize=14)

    for ax in g.axes.flat:
        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    save_figure(outdir, "scaling_rank_facet.pdf")


def plot_scaling_nodes_facet(
    df: pd.DataFrame,
    outdir: Path,
    default_rank: int,
) -> None:
    """FacetGrid showing node scaling, faceted by Metric (averaged across bins)."""
    if df["Nodes"].nunique() <= 1:
        return

    metric_cols = identify_metric_columns(df)
    if not metric_cols:
        return

    df_subset = df[df["Rank"] == default_rank]
    df_agg = aggregate_across_bins(df_subset, metric_cols)
    df_long = melt_metrics(df_agg, metric_cols)
    if df_long.empty:
        return

    n_metrics = len(metric_cols)
    ncols = min(3, n_metrics)

    g = sns.FacetGrid(
        df_long,
        col="Metric",
        col_wrap=ncols,
        height=4,
        aspect=1.3,
        sharex=True,
        sharey=False,
    )
    g.map_dataframe(
        sns.lineplot,
        x="Nodes",
        y="Score",
        hue="Method",
        markers=True,
        markersize=8,
        linewidth=2.5,
        palette="Set2",
    )
    g.add_legend(title="Method")
    g.set_axis_labels("Number of Nodes", "Score")
    g.set_titles("{col_name}")
    g.figure.suptitle(f"Node Scaling (Rank={default_rank}, averaged across bins)", y=1.01, fontsize=14)

    for ax in g.axes.flat:
        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    save_figure(outdir, "scaling_nodes_facet.pdf")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot node classification benchmarks.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory containing benchmark_results.csv.",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """Main plotting workflow."""
    args = parse_args()
    run_dir = resolve_results_dir(args.dir)

    results_path = run_dir / "benchmark_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"benchmark_results.csv not found at {results_path}")

    print(f"Loading results from: {run_dir}")
    df = pd.read_csv(results_path)
    if df.empty:
        print("Empty results file, skipping plots.")
        return

    setup_plot_style()
    default_rank = df["Rank"].max()

    metrics = identify_metric_columns(df)
    print(f"Found metrics: {', '.join(metrics)}")
    print(f"Generating faceted plots (default Rank={default_rank})...")

    plot_method_comparison_facet(df, run_dir, default_rank)
    plot_scaling_rank_facet(df, run_dir)
    plot_scaling_nodes_facet(df, run_dir, default_rank)
    print("Done!")


if __name__ == "__main__":
    main()
