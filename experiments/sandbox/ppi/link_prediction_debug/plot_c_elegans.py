"""Plot C. elegans link prediction comparison: SRF vs SkipGNN."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.colors import ROSE, PURPLE, GRAY_DARK, GRAY, GRAY_LIGHT, GRAY_PALE
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure

DATA_DIR = Path(__file__).parents[2] / "outputs/experiments/ppi/data/link_prediction/c_elegans"
OUTPUT_DIR = get_output_dir()

METHOD_COLORS = {
    "srf": ROSE,
    "skipgnn": PURPLE,
    "aa": GRAY_DARK,
    "cn": GRAY,
    "ra": GRAY_LIGHT,
    "jc": GRAY_PALE,
}

METHOD_LABELS = {
    "srf": "SRF",
    "skipgnn": "SkipGNN",
    "aa": "AA",
    "cn": "CN",
    "ra": "RA",
    "jc": "JC",
}

METHOD_ORDER = ["srf", "skipgnn", "aa", "cn", "ra", "jc"]


def load_results() -> pd.DataFrame:
    records = []
    for file_path in DATA_DIR.glob("**/*.json"):
        with open(file_path) as f:
            records.append(json.load(f))
    return pd.DataFrame(records)


def main():
    df = load_results()
    print(f"Loaded {len(df)} records")

    stats = df.groupby("method")["auroc"].agg(["mean", "std"]).reset_index()
    stats = stats.set_index("method")

    methods = [m for m in METHOD_ORDER if m in stats.index]

    fig, ax = create_figure("single", pad_bottom=0.6, pad_left=0.5, pad_right=0.2, pad_top=0.1)

    x = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, GRAY) for m in methods]
    means = [stats.loc[m, "mean"] for m in methods]
    stds = [stats.loc[m, "std"] for m in methods]

    ax.bar(x, means, yerr=stds, capsize=3, color=colors, width=0.7, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 0.9)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "c_elegans_link_prediction.pdf")
    print(f"Saved to {OUTPUT_DIR / 'c_elegans_link_prediction.pdf'}")

    print("\nSummary:")
    for m in methods:
        print(f"  {METHOD_LABELS.get(m, m):10s}: {stats.loc[m, 'mean']:.3f} ± {stats.loc[m, 'std']:.3f}")


if __name__ == "__main__":
    main()
