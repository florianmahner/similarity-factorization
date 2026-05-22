"""Plot the longer-training CV curve vs the old baseline curve."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import GRAY, INDIGO
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
NEW_CSV = OUTPUT_DIR / "cv_results_batched.csv"
OLD_JSON = ROOT / "experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json"


def main() -> None:
    new = pd.read_csv(NEW_CSV).sort_values("rank").reset_index(drop=True)
    old = json.loads(OLD_JSON.read_text())["validations"]["5fold"]
    old_x = np.array(old["ranks"])
    old_y = np.array(old["val_mse_mean"])
    old_sem = np.array(old["val_mse_sem"])
    new_x = new["rank"].to_numpy()
    new_y = new["val_mse_mean"].to_numpy()
    new_sem = new["val_mse_sem"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar(
        old_x, old_y, yerr=2 * old_sem,
        marker="o", markersize=4, linewidth=1.2, capsize=2,
        color=GRAY, alpha=0.85,
        label="old: max_outer=50, n_repeats=20",
    )
    ax.errorbar(
        new_x, new_y, yerr=2 * new_sem,
        marker="o", markersize=4, linewidth=1.6, capsize=2,
        color=INDIGO,
        label="new: max_outer=200, n_repeats=10",
    )

    new_argmin = int(new_x[np.argmin(new_y)])
    old_argmin = int(old_x[np.argmin(old_y)])
    ax.axvline(new_argmin, color=INDIGO, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.axvline(old_argmin, color=GRAY, linestyle=":", linewidth=0.9, alpha=0.5)

    ax.set_xlabel("rank")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"things_behavior CV (n=1854): argmin {old_argmin} → {new_argmin}")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUTPUT_DIR / "u_curve_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
