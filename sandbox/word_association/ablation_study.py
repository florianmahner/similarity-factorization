"""Ablation study for SWOW graph construction approaches."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
AXES_DIR = BASE_DIR / "outputs" / "semantic_axes"

from analyses.words.swow_graph import (
    load_swow_data,
    filter_by_frequency,
    build_count_graph,
    symmetrize_counts_then_ppmi,
    calculate_ppmi_then_symmetrize,
    fit_srf,
)
from experiments.word_association.semantic_axes import validate_axes_quick


def run_config(
    df: pd.DataFrame,
    ratings_dir: Path,
    axes_file: Path,
    vocab_size: int,
    use_r123: bool,
    ppmi_timing: str,
    sym_method: str,
    unidirectional: str,
    rank: int,
) -> dict:
    try:
        df_work = filter_by_frequency(df, vocab_size)
        g = build_count_graph(df_work)

        if ppmi_timing == "after":
            similarity, vocab, meta = symmetrize_counts_then_ppmi(
                g, sym_method, unidirectional
            )
        else:
            similarity, vocab, meta = calculate_ppmi_then_symmetrize(
                g, df_work, sym_method
            )

        w = fit_srf(similarity, rank)
        validation = validate_axes_quick(
            w, vocab, axes_file, ratings_dir, axis_subset=["animacy", "concreteness"]
        )

        observed = ~np.isnan(similarity)
        recon = w @ w.T
        corr = np.corrcoef(similarity[observed], recon[observed])[0, 1]

        return {
            "vocab_size": vocab_size,
            "use_r123": use_r123,
            "ppmi_timing": ppmi_timing,
            "sym_method": sym_method,
            "unidirectional_handling": unidirectional,
            "reconstruction_corr": corr,
            **meta,
            **validation,
            "success": True,
        }
    except Exception as e:
        return {
            "vocab_size": vocab_size,
            "use_r123": use_r123,
            "ppmi_timing": ppmi_timing,
            "error": str(e),
            "success": False,
        }


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    df_ok = df[df["success"]].copy()
    if len(df_ok) == 0:
        return

    sns.set_style("whitegrid")

    for metric in ["animacy_corr", "concreteness_corr"]:
        if metric not in df_ok.columns:
            continue

        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        data = df_ok.dropna(subset=[metric])

        if len(data) > 0 and "ppmi_timing" in data.columns:
            sns.boxplot(
                data=data, x="ppmi_timing", y=metric, ax=axes[0], palette="Set2"
            )
            axes[0].set_title(f"{metric.replace('_', ' ').title()} by PPMI Timing")
            axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

        if len(data) > 0 and "use_r123" in data.columns:
            sns.boxplot(
                data=data,
                x="use_r123",
                y=metric,
                ax=axes[1],
                palette="Set1",
                order=[False, True],
            )
            axes[1].set_xticklabels(["R1", "R123"])
            axes[1].set_title(f"{metric.replace('_', ' ').title()} by Response Set")
            axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{metric}_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def main(output_dir: Path | None, vocab_sizes: list[int], rank: int) -> None:
    swow_dir = Path("data/small-world-of-words")
    ratings_dir = Path("data")
    axes_file = AXES_DIR / "semantic_axes.json"

    if output_dir is None:
        output_dir = Path("experiments/word_association/outputs/ablation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for vocab_size in vocab_sizes:
        for use_r123 in [False, True]:
            for ppmi_timing in ["after", "before"]:
                configs.append(
                    (vocab_size, use_r123, ppmi_timing, "geometric_mean", "copy")
                )

    for unidirectional in ["discard", "weight_half"]:
        configs.append(
            (vocab_sizes[0], False, "after", "geometric_mean", unidirectional)
        )

    df_r1 = load_swow_data(swow_dir, False)
    df_r123 = load_swow_data(swow_dir, True)

    def wrapper(cfg):
        vocab_size, use_r123, ppmi_timing, sym_method, unidirectional = cfg
        df = df_r123 if use_r123 else df_r1
        return run_config(
            df,
            ratings_dir,
            axes_file,
            vocab_size,
            use_r123,
            ppmi_timing,
            sym_method,
            unidirectional,
            rank,
        )

    results = Parallel(n_jobs=-1)(delayed(wrapper)(cfg) for cfg in configs)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "ablation_results.csv", index=False)
    plot_results(df_results, output_dir)

    df_ok = df_results[df_results["success"]]
    if len(df_ok) > 0:
        print(f"\nSuccessful: {len(df_ok)}/{len(df_results)}")
        if "animacy_corr" in df_ok.columns and df_ok["animacy_corr"].notna().any():
            best = df_ok.loc[df_ok["animacy_corr"].idxmax()]
            print(
                f"\nBest animacy ({best['animacy_corr']:.3f}): PPMI {best['ppmi_timing']}, {'R123' if best['use_r123'] else 'R1'}, {best['vocab_size']} words"
            )

    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--rank", type=int, default=30)
    args = parser.parse_args()
    main(
        Path(args.output_dir) if args.output_dir else None, args.vocab_sizes, args.rank
    )
