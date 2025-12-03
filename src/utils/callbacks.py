"""
Gather, Merge, Plot, and Cleanup Script.
Acts as the 'on_multirun_end' callback for Slurm jobs.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Set Plotting Style
sns.set_theme(style="ticks", context="talk")

def parse_args():
    parser = argparse.ArgumentParser(description="Gather Slurm results, plot, and cleanup.")
    parser.add_argument(
        "input_dir", 
        nargs="?", 
        type=Path, 
        default=None,
        help="Path to the Hydra multirun directory"
    )
    # Add a safety flag so we don't delete things by accident unless asked
    parser.add_argument("--cleanup", action="store_true", help="Delete .tmp/ folders and individual json files after merging")
    return parser.parse_args()

def find_latest_multirun():
    """Helper to find the most recent run if no path provided."""
    base = Path("multirun")
    if not base.exists(): return None
    dates = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    if not dates: return None
    times = sorted([t for t in dates[0].iterdir() if t.is_dir()], reverse=True)
    if not times: return None
    return times[0]

def load_and_merge_results(search_dir: Path) -> pd.DataFrame:
    """Mimics your Callback logic: finds JSONs, merges them, saves CSV."""
    
    # 1. Find files (Recursive search)
    files = list(search_dir.rglob("results.json"))
    log.info(f"Found {len(files)} result files in {search_dir}")

    if not files:
        return pd.DataFrame()

    # 2. Read and Concatenate
    dfs = []
    for f in files:
        try:
            # Using your callback's orient="records" just in case, 
            # though standard dict dump usually requires orient='columns' or default.
            # We try standard read first since our run.py dumps a dict, not a list of records.
            with open(f, 'r') as fh:
                data = json.load(fh)
            
            # Filter for success only
            if data.get("status") == "success":
                dfs.append(pd.DataFrame([data])) # Wrap dict in list to make it a row
                
        except Exception as e:
            log.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Save to CSV (Your Callback logic)
    output_path = search_dir / "results.csv"
    merged_df.to_csv(output_path, index=False)
    log.info(f"Successfully merged {len(dfs)} runs into {output_path}")
    
    return merged_df

def generate_plots(df: pd.DataFrame, output_dir: Path):
    """Generates the analysis plots."""
    if df.empty: return

    # Normalize column names for plotting
    # The run.py saves keys: method, rank, size, bin_range
    rename_map = {"method": "Method", "rank": "Rank", "size": "Nodes", "bin_range": "Bin"}
    plot_df = df.rename(columns=rename_map)
    plot_df["Method"] = plot_df["Method"].str.upper()

    # --- Plotting Logic (Same as before) ---
    min_score = max(0, plot_df["Micro-F1"].min() * 0.95)
    max_score = plot_df["Micro-F1"].max() * 1.05
    
    # Defaults
    sizes = sorted(plot_df["Nodes"].unique())
    ranks = sorted(plot_df["Rank"].unique())
    # Sort bins by first number
    bins = sorted(plot_df["Bin"].unique(), key=lambda x: int(x.split('-')[0]))

    default_nodes = sizes[-1]
    default_rank = ranks[-1]
    default_bin = bins[-1]

    # Plot 1: Bar Chart
    subset = plot_df[(plot_df["Nodes"] == default_nodes) & (plot_df["Rank"] == default_rank)]
    if not subset.empty:
        df_melt = subset.melt(id_vars=["Method", "Bin"], value_vars=["Micro-F1", "Macro-F1"], var_name="Metric", value_name="Score")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Bin", y="Score", hue="Method", palette="mako")
        plt.title(f"Method Comparison (N={default_nodes}, R={default_rank})")
        plt.ylim(min_score, max_score)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "summary_barplot.png", dpi=300)
        plt.close()

    # Plot 2: Scaling
    subset_scaling = plot_df[(plot_df["Bin"] == default_bin) & (plot_df["Rank"] == default_rank)]
    if not subset_scaling.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset_scaling, x="Nodes", y="Micro-F1", hue="Method", markers=True, style="Method")
        plt.title(f"Scaling vs Size (Bin={default_bin})")
        plt.ylim(min_score, max_score)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "summary_scaling.png", dpi=300)
        plt.close()

def cleanup_files(sweep_dir: Path):
    """
    Your Callback's cleanup logic.
    Removes the individual job folders generated by Hydra/Submitit
    """
    # Identify subdirectories that look like job folders (e.g., '0', '1', 'bin_select=small...')
    # Be careful not to delete the plots we just made!
    
    log.info("Starting cleanup...")
    
    # 1. Remove individual result.json files to save space?
    # Or 2. Remove the entire job directories?
    
    # Logic: Delete any subdirectory that contains a results.json that we have already merged
    for json_file in sweep_dir.rglob("results.json"):
        parent_dir = json_file.parent
        # Ensure we are inside the sweep dir and it's a subdirectory
        if parent_dir != sweep_dir and parent_dir.is_dir():
            try:
                shutil.rmtree(parent_dir)
                # log.info(f"Deleted job dir: {parent_dir.name}")
            except Exception as e:
                log.warning(f"Could not delete {parent_dir}: {e}")
                
    # Also look for .tmp if submitit made it
    tmp_dir = sweep_dir / ".tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        log.info("Deleted .tmp directory")

def main():
    args = parse_args()
    
    # Auto-detect directory
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = find_latest_multirun()
        if input_dir is None:
            log.error("No input directory found.")
            sys.exit(1)
            
    input_dir = input_dir.resolve()
    log.info(f"Processing: {input_dir}")

    # 1. Merge
    df = load_and_merge_results(input_dir)
    
    if df.empty:
        log.error("No valid results found. Exiting.")
        sys.exit(1)

    # 2. Plot
    generate_plots(df, input_dir)
    
    # 3. Cleanup (Optional flag, or default if you prefer)
    if args.cleanup:
        cleanup_files(input_dir)
    
    log.info("Workflow complete.")

if __name__ == "__main__":
    main()