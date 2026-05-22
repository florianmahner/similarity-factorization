"""Cross-validate SRF on THINGS behavioral using p*=0.875 from coherence kappa liftoff."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import logging

from src.similarity import build_similarity
from src.colors import ROSE, TEAL, CYAN
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine
from omegaconf import OmegaConf
from pysrf import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def main():
    cfg = OmegaConf.create({
        "name": "things_behavior",
        "type": "triplet",
        "path": "data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    s = build_similarity(cfg)
    log.info(f"THINGS behavioral: {s.shape}, NaN={np.sum(np.isnan(s))}")

    ranks = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    p_star = 0.875

    log.info(f"Running pysrf CV: ranks={ranks}, p*={p_star}, n_repeats=5")
    result = cross_val_score(
        s,
        param_grid={"rank": ranks},
        sampling_fraction=p_star,
        n_repeats=5,
        n_jobs=-1,
        verbose=1,
    )

    log.info(f"\nBest params: {result.best_params_}")
    log.info(f"Best score: {result.best_score_:.6f}")

    cv_df = result.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].agg(["mean", "std"]).reset_index()
    mean_scores = mean_scores.sort_values("rank")

    log.info("\n=== CV RESULTS ===")
    log.info(f"{'rank':>6} | {'mean_error':>12} | {'std':>10}")
    log.info("-" * 35)
    for _, row in mean_scores.iterrows():
        log.info(f"{int(row['rank']):>6} | {row['mean']:>12.6f} | {row['std']:>10.6f}")

    cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    # -- Plot CV curve --
    fig, ax = create_figure("wide")
    ax.errorbar(
        mean_scores["rank"], mean_scores["mean"], yerr=mean_scores["std"],
        marker="o", markersize=4, linewidth=1.5, capsize=3, color=TEAL,
    )
    best_k = int(result.best_params_["rank"])
    ax.axvline(best_k, linestyle="--", linewidth=1, color=ROSE,
               label=f"Best $k^*$ = {best_k}")

    # Mark coherence kappa k*=25
    ax.axvline(25, linestyle=":", linewidth=1, color=CYAN,
               label="Kappa $k^*$ = 25")

    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Validation error (Frobenius)")
    ax.set_title(f"SRF cross-validation (THINGS, $p$ = {p_star})")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "cv_curve.png", dpi=300, bbox_inches="tight", facecolor="white")

    log.info(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
