"""Masked parallel analysis on THINGS behavioral data across triplet percentages.

Validates the masked parallel analysis rank estimator by showing how k*
increases with data volume, comparing against kappa (B1) and activation.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coherence import masked_parallel_analysis
from src.colors import ROSE, TEAL, GRAY, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N_OBJECTS = 1854

PERCENTAGES = [5, 10, 20, 50, 100]
K_MAX = 120
P_LIST = np.linspace(0.05, 0.95, 25)
B = 50
J = 100
ALPHA = 0.05


def _load_similarity(pct: int, partition: int = 0) -> tuple[np.ndarray, int]:
    """Load triplets at given percentage and build similarity matrix."""
    if pct == 100:
        triplets, _ = load_triplets(DATA_DIR)
    else:
        path = DATA_DIR / "partitions" / f"{pct}pct_part{partition}" / "train_90.txt"
        triplets = np.loadtxt(path).astype(int)
    s = compute_similarity_matrix_from_triplets(N_OBJECTS, triplets, alpha=0)
    return s, len(triplets)


def plot_kstar_vs_data(records: list[dict], output_dir: Path) -> None:
    """Line plot of k* vs triplet percentage."""
    fig, ax = create_figure("single")
    pcts = [r["pct"] for r in records]
    kstars = [r["k_star"] for r in records]
    ax.plot(pcts, kstars, marker="o", color=TEAL, linewidth=2, label="Parallel analysis")
    ax.set_xlabel("Triplet data (%)")
    ax.set_ylabel("Estimated rank (k*)")
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    despine(ax)
    fig.savefig(output_dir / "kstar_vs_data.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_eigenvalue_spectrum(result_100: dict, output_dir: Path) -> None:
    """Eigenvalue spectrum vs null threshold at 100% data."""
    fig, ax = create_figure("wide")
    k = np.arange(1, len(result_100["evals_ref"]) + 1)
    ax.plot(k, result_100["evals_observed"][:, -1], color=TEAL, label="Observed (p=0.95)")
    ax.plot(
        k,
        result_100["thresholds"][:, -1],
        color=GRAY,
        linestyle="--",
        label=f"Null ({int((1 - ALPHA) * 100)}th pctl)",
    )
    ax.axvline(
        result_100["k_star"],
        color=ROSE,
        linestyle=":",
        linewidth=1.5,
        label=f'k*={result_100["k_star"]}',
    )
    ax.set_xlabel("Component index (k)")
    ax.set_ylabel("Eigenvalue")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(
        output_dir / "eigenvalue_vs_null_100pct.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_pvalues(result_100: dict, output_dir: Path) -> None:
    """Bar plot of p-values for each component index."""
    fig, ax = create_figure("wide")
    k = np.arange(1, len(result_100["pvalues"]) + 1)
    colors = [TEAL if p < ALPHA else GRAY_LIGHT for p in result_100["pvalues"]]
    ax.bar(k, result_100["pvalues"], color=colors, width=1.0, edgecolor="white", linewidth=0.3)
    ax.axhline(ALPHA, color=ROSE, linestyle="--", linewidth=1, label=f"alpha={ALPHA}")
    ax.set_xlabel("Component index (k)")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(output_dir / "pvalues_100pct.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    records = []
    result_100 = None

    for pct in PERCENTAGES:
        log.info("Processing %d%% data...", pct)
        s, n_triplets = _load_similarity(pct)
        log.info("Loaded similarity matrix: shape=%s, n_triplets=%d", s.shape, n_triplets)

        t0 = time.time()
        result = masked_parallel_analysis(
            s,
            k_max=K_MAX,
            p_list=P_LIST,
            B=B,
            J=J,
            alpha=ALPHA,
            random_state=42,
            show_progress=True,
        )
        runtime = time.time() - t0

        k_star = result["k_star"]
        log.info("pct=%d%% -> k*=%d (%.1fs)", pct, k_star, runtime)

        record = dict(pct=pct, k_star=k_star, n_triplets=n_triplets, runtime_sec=round(runtime, 1))
        records.append(record)

        json_path = OUTPUT_DIR / f"things_{pct}pct.json"
        json_record = dict(
            pct=pct,
            k_star=k_star,
            n_triplets=n_triplets,
            pvalues=result["pvalues"].tolist(),
            runtime_sec=round(runtime, 1),
        )
        with open(json_path, "w") as f:
            json.dump(json_record, f, indent=2)
        log.info("Saved JSON: %s", json_path)

        npz_path = OUTPUT_DIR / f"things_{pct}pct.npz"
        np.savez(
            npz_path,
            evals_observed=result["evals_observed"],
            evals_null=result["evals_null"],
            thresholds=result["thresholds"],
            evals_ref=result["evals_ref"],
            pvalues=result["pvalues"],
        )
        log.info("Saved NPZ: %s", npz_path)

        if pct == 100:
            result_100 = result

    log.info("Generating plots...")
    plot_kstar_vs_data(records, OUTPUT_DIR)
    if result_100 is not None:
        plot_eigenvalue_spectrum(result_100, OUTPUT_DIR)
        plot_pvalues(result_100, OUTPUT_DIR)

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)

    log.info("Summary:\n%s", df.to_string(index=False))
    log.info("All outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
