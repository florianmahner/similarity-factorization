from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

# Mapping for clean method names in plots
METHOD_NAMES = {
    "srf": "SRF (Ours)",
    "seal": "SEAL",
    "skipgnn": "SkipGNN",
    "cn": "Common Neighbors",
    "aa": "Adamic-Adar",
    "node2vec": "Node2Vec",
    "deepwalk": "DeepWalk",
    "line": "LINE"
}

def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Scans `{results_dir}/**/*.json` and returns a DataFrame.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"No results found in {results_dir}")

    records = []
    for file_path in results_dir.glob("**/*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    return df

def plot_method_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Bar plot comparing methods (aggregating over folds).
    Uses the BEST rank for each method if multiple ranks exist.
    """
    # Select best rank for each method based on AUROC
    best_ranks = df.groupby(["method", "rank"])["auroc"].mean().reset_index()
    best_ranks = best_ranks.sort_values("auroc", ascending=False).drop_duplicates("method")
    
    # Filter original DF to keep only best ranks
    # This is a bit tricky with pandas, so we just filter by (method, rank) pairs
    df_best = df.merge(best_ranks[["method", "rank"]], on=["method", "rank"])
    
    # Map names
    df_best["method_name"] = df_best["method"].map(lambda x: METHOD_NAMES.get(x, x.upper()))

    plt.figure(figsize=(12, 6))
    sns.set_style("ticks")
    
    ax = sns.barplot(
        x="method_name", 
        y="auroc", 
        data=df_best, 
        palette="viridis", 
        capsize=0.1,
        errwidth=1.5
    )
    
    plt.title("Link Prediction Performance (Best Configuration)", fontsize=14)
    plt.xlabel("")
    plt.ylabel("AUROC", fontsize=12)
    plt.ylim(0.5, 1.0)
    sns.despine()
    plt.tight_layout()
    
    plot_path = output_dir / "method_comparison.pdf"
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")

def plot_rank_influence(df: pd.DataFrame, output_dir: Path):
    """
    Line plot showing performance vs rank for embedding methods.
    """
    # Filter for methods that have multiple ranks
    rank_counts = df.groupby("method")["rank"].nunique()
    methods_with_ranks = rank_counts[rank_counts > 1].index.tolist()
    
    if not methods_with_ranks:
        print("No methods with multiple ranks found. Skipping rank plot.")
        return

    df_ranks = df[df["method"].isin(methods_with_ranks)].copy()
    df_ranks["method_name"] = df_ranks["method"].map(lambda x: METHOD_NAMES.get(x, x.upper()))
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    sns.lineplot(
        x="rank", 
        y="auroc", 
        hue="method_name", 
        style="method_name",
        markers=True, 
        dashes=False,
        data=df_ranks,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title("Influence of Embedding Dimension (Rank)", fontsize=14)
    plt.xlabel("Rank / Dimension", fontsize=12)
    plt.ylabel("AUROC", fontsize=12)
    plt.legend(title="Method")
    plt.tight_layout()
    
    plot_path = output_dir / "rank_influence.pdf"
    plt.savefig(plot_path)
    print(f"Saved rank plot to {plot_path}")

def run(cfg: DictConfig) -> None:
    results_dir = Path(cfg.results_dir)

    print(f"Loading results from: {results_dir}...")
    df = load_results(results_dir)
    
    if df.empty:
        print("No results found!")
        return
        
    print(f"Loaded {len(df)} records.")
    
    # Output directory for plots (experiment outputs folder)
    output_dir = results_dir.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Stats
    summary = df.groupby(["method", "rank"])[["auroc", "auprc", "p500", "ndcg"]].agg(["mean", "std"])
    print("\nSummary Statistics:")
    print(summary)
    summary.to_csv(output_dir / "summary_stats.csv")
    
    # Generate Plots
    plot_method_comparison(df, output_dir)
    plot_rank_influence(df, output_dir)
