#!/usr/bin/env python3
"""
Plot complex prediction analysis results.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


def load_cfg_plot_dir() -> Path | None:
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    if "plot_input_dir" in cfg and cfg.plot_input_dir:
        return Path(cfg.plot_input_dir)
    return None


def latest_results_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.rglob("srf_corum_validation.csv") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def resolve_results_dir(args_dir: Path | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    cfg_dir = load_cfg_plot_dir()
    script_outputs = script_dir / "outputs"

    if args_dir:
        if (args_dir / "srf_corum_validation.csv").exists():
            return args_dir
        for cand in [script_dir / args_dir, Path.cwd() / args_dir]:
            if (cand / "srf_corum_validation.csv").exists():
                return cand
        return args_dir

    if cfg_dir:
        return cfg_dir

    latest = latest_results_dir(script_outputs)
    if latest:
        return latest

    raise FileNotFoundError("No results directory found. Pass --dir or set plot_input_dir.")


def setup_style():
    sns.set_theme(style="ticks", context="talk")
    plt.rcParams["figure.dpi"] = 100


def save_fig(outdir: Path, name: str):
    plt.savefig(outdir / name, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {outdir / name}")


def plot_corum_f1_per_dimension(df: pd.DataFrame, outdir: Path):
    """Bar plot of CORUM F1 per dimension, annotated with complex names."""
    df_sorted = df.sort_values("f1", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(df_sorted)), df_sorted["f1"], color="steelblue", edgecolor="black")

    top_5 = df_sorted.head(5)
    for i, (_, row) in enumerate(top_5.iterrows()):
        label = row["best_complex"][:25] + "..." if len(row["best_complex"]) > 25 else row["best_complex"]
        ax.annotate(
            label,
            (i, row["f1"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            rotation=45,
        )

    ax.set_xlabel("Dimension (sorted by F1)")
    ax.set_ylabel("CORUM F1 Score")
    ax.set_title("SRF Dimension Quality: Best CORUM Complex Match per Dimension")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="F1=0.5 threshold")
    ax.legend()
    sns.despine()

    save_fig(outdir, "srf_corum_f1_per_dimension.png")


def plot_method_comparison(combined_df: pd.DataFrame, outdir: Path):
    """Compare SRF vs LINE+KMeans on CORUM recovery."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.boxplot(data=combined_df, x="method", y="f1", ax=ax, palette="Set2")
    ax.set_xlabel("Method")
    ax.set_ylabel("CORUM F1 Score")
    ax.set_title("Complex Recovery: SRF vs LINE+KMeans")
    sns.despine(ax=ax)

    ax = axes[1]
    summary = combined_df.groupby("method")["f1"].agg(["mean", "std", "max"]).reset_index()
    x = range(len(summary))
    bars = ax.bar(x, summary["mean"], yerr=summary["std"], capsize=5, color=["steelblue", "coral"])
    ax.scatter(x, summary["max"], marker="*", s=200, c="gold", edgecolors="black", zorder=5, label="Max F1")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"])
    ax.set_ylabel("CORUM F1 Score")
    ax.set_title("Mean (±std) and Max F1")
    ax.legend()
    sns.despine(ax=ax)

    plt.tight_layout()
    save_fig(outdir, "method_comparison_corum.png")


def plot_leave_one_out(loo_df: pd.DataFrame, outdir: Path):
    """Visualize leave-one-out ablation results."""
    if loo_df.empty:
        print("No leave-one-out results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    recovery_by_complex = loo_df.groupby("complex")["recovered_top_k"].mean().sort_values(ascending=False)
    bars = ax.barh(range(len(recovery_by_complex)), recovery_by_complex.values, color="steelblue")
    ax.set_yticks(range(len(recovery_by_complex)))
    ax.set_yticklabels([c[:30] for c in recovery_by_complex.index], fontsize=9)
    ax.set_xlabel("Recovery Rate")
    ax.set_title("Leave-One-Out Recovery by Complex")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5)
    sns.despine(ax=ax)

    ax = axes[1]
    ax.hist(loo_df["rank_percentile"], bins=20, color="steelblue", edgecolor="black")
    ax.axvline(loo_df["rank_percentile"].mean(), color="red", linestyle="--", label=f"Mean={loo_df['rank_percentile'].mean():.2f}")
    ax.set_xlabel("Rank Percentile of Left-Out Protein")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Left-Out Protein Rankings")
    ax.legend()
    sns.despine(ax=ax)

    plt.tight_layout()
    save_fig(outdir, "leave_one_out_ablation.png")


def plot_novel_complexes(novel_df: pd.DataFrame, outdir: Path):
    """Visualize novel complex candidates."""
    if novel_df.empty:
        print("No novel complex candidates to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["coral" if x else "steelblue" for x in novel_df["is_novel_candidate"]]
    ax.scatter(novel_df["corum_f1"], -np.log10(novel_df["go_pvalue_fdr"] + 1e-10),
               c=colors, s=100, alpha=0.7, edgecolors="black")

    ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="FDR=0.05")
    ax.axvline(0.3, color="gray", linestyle="--", alpha=0.5, label="CORUM F1=0.3")

    ax.set_xlabel("CORUM F1 Score")
    ax.set_ylabel("-log10(GO enrichment FDR)")
    ax.set_title("Novel Complex Candidates\n(Low CORUM match, High GO enrichment)")
    ax.legend()
    sns.despine()

    for _, row in novel_df[novel_df["is_novel_candidate"]].iterrows():
        ax.annotate(
            f"Dim {int(row['dimension'])}",
            (row["corum_f1"], -np.log10(row["go_pvalue_fdr"] + 1e-10)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    save_fig(outdir, "novel_complex_candidates.png")


def plot_dimension_heatmap(srf_df: pd.DataFrame, outdir: Path):
    """Heatmap showing which dimensions match which complexes."""
    top_dims = srf_df.nlargest(15, "f1")

    fig, ax = plt.subplots(figsize=(12, 8))

    data = top_dims[["dimension", "best_complex", "f1", "precision", "recall"]].copy()
    data["label"] = data.apply(
        lambda r: f"{r['best_complex'][:25]}..." if len(r['best_complex']) > 25 else r['best_complex'],
        axis=1
    )

    bars = ax.barh(range(len(data)), data["f1"], color="steelblue", label="F1")
    ax.barh(range(len(data)), data["precision"], color="coral", alpha=0.5, label="Precision")
    ax.barh(range(len(data)), data["recall"], color="green", alpha=0.3, label="Recall")

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f"Dim {int(d)}: {l}" for d, l in zip(data["dimension"], data["label"])], fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("Top 15 SRF Dimensions: CORUM Complex Matches")
    ax.legend(loc="lower right")
    sns.despine()

    save_fig(outdir, "top_dimensions_corum_matches.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot complex prediction results.")
    parser.add_argument("--dir", type=Path, default=None, help="Results directory.")
    args, _ = parser.parse_known_args()
    return args


def main():
    import numpy as np

    args = parse_args()
    run_dir = resolve_results_dir(args.dir)

    print(f"Loading results from: {run_dir}")
    setup_style()

    srf_path = run_dir / "srf_corum_validation.csv"
    if srf_path.exists():
        srf_df = pd.read_csv(srf_path)
        plot_corum_f1_per_dimension(srf_df, run_dir)
        plot_dimension_heatmap(srf_df, run_dir)

    combined_path = run_dir / "combined_corum_validation.csv"
    if combined_path.exists():
        combined_df = pd.read_csv(combined_path)
        if combined_df["method"].nunique() > 1:
            plot_method_comparison(combined_df, run_dir)

    loo_path = run_dir / "leave_one_out_results.csv"
    if loo_path.exists():
        loo_df = pd.read_csv(loo_path)
        plot_leave_one_out(loo_df, run_dir)

    novel_path = run_dir / "novel_complex_candidates.csv"
    if novel_path.exists():
        novel_df = pd.read_csv(novel_path)
        plot_novel_complexes(novel_df, run_dir)

    print("Done!")


if __name__ == "__main__":
    main()
