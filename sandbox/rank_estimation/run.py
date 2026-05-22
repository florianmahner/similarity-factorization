"""End-to-end demo of RankEstimator + cross_val_score on a 10-block synthetic.

Two-line user workflow:

    est = RankEstimator(recovery_tolerance=0.10).fit(s)
    curve = cross_val_score(s, ranks=[...], sampling_fraction=est.sampling_fraction_)

Outputs at ``sandbox/rank_estimation/outputs/``:
    summary.txt           Verdict, rank_, sampling_fraction_, per-rank V-MSE table
    01_eigenvalues.png    Reference spectrum with rank_ marker
    02_leakage.png        Per-dimension leakage with rank_ marker
    03_recovery.png       Recovery curve with sampling_fraction_ marker
    04_cv_curve.png       Individual folds + mean, with elbow marker
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.colors import GRAY, GRAY_LIGHT, INDIGO, ROSE, SAND, TEAL, setup_style
from src.utils import get_output_dir
from src.utils.figure_theme import despine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coherence import RankEstimator  # noqa: E402
from cross_validation import cross_val_score  # noqa: E402


log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()

FIG_SIZE = (7.0, 4.5)
PNG_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")


def _block_latents(n: int, true_rank: int, seed: int) -> np.ndarray:
    """Sparse block-structured latent codes for the synthetic kernel."""
    rng = np.random.default_rng(seed)
    overlap = max(1, n // 40)
    d = np.zeros((n, true_rank), dtype=float)
    block = n // true_rank
    for k in range(true_rank):
        start = k * block
        end = start + block + overlap
        for j in range(n):
            if start <= j <= end:
                d[j, k] = rng.uniform(0.5, 1.0) * rng.binomial(1, 0.9)
    d = np.clip(d, 0.0, 1.0)
    return d + rng.random((n, true_rank)) * 0.5


def make_block_similarity(
    n: int = 400, true_rank: int = 10, bandwidth: float = 1.0, seed: int = 0
) -> np.ndarray:
    """RBF kernel on sparse block-structured latents."""
    d = _block_latents(n, true_rank, seed)
    d_squared = cdist(d, d, metric="sqeuclidean")
    return np.exp(-d_squared / (2.0 * bandwidth * bandwidth))


def _rank_grid(n: int, est_rank: int, true_rank: int) -> list[int]:
    """Sparse candidate-rank grid matching the update_pysrf notebook's setup.

    The notebook (``update_pysrf/tutorial/recipe_K_demo.ipynb``) uses
    ``ranks = [3, 6, 9, 11, 13, 16, 20]`` and reports ``argmin = 11`` across
    ``k_inner ∈ {3, 5, 10}`` (fold-invariance demo). We use the same grid so
    the demo is directly comparable to the notebook result.
    """
    return [3, 6, 9, 11, 13, 16, 20]


def _argmin_rank(ranks: list[int], mean_curve: np.ndarray) -> int:
    """Bare argmin: the rank with the smallest mean validation MSE."""
    return int(ranks[int(np.nanargmin(mean_curve))])


def _elbow_rank(ranks: list[int], mean_curve: np.ndarray) -> int:
    """Kneedle elbow on a monotone-decreasing curve.

    Normalizes both axes to [0, 1] and returns the rank whose point lies
    farthest below the straight line from (min_rank, max_score) to
    (max_rank, min_score). For a CV curve with a sharp drop and then a
    slow plateau, this is the rank at the bend — robust to whether the
    curve keeps slowly decreasing past the true rank (the regime where
    bare argmin overshoots).
    """
    r = np.asarray(ranks, dtype=float)
    s = np.asarray(mean_curve, dtype=float)
    r_norm = (r - r.min()) / max(r.max() - r.min(), 1e-12)
    s_norm = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    distance_below_diagonal = (1.0 - r_norm) - s_norm
    return int(r[np.argmax(distance_below_diagonal)])


def _argmin_1se_rank(
    ranks: list[int], mean_curve: np.ndarray, sem_curve: np.ndarray
) -> int:
    """1-SE rule: smallest rank within one standard error of the minimum.

    Standard parsimonious rank-selection rule. For curves that genuinely
    U-turn (the notebook's ADMM fitter), this is sharp. For curves that
    only slowly plateau (pysrf.SRF), the threshold is tight and 1-SE
    drifts toward the argmin — use ``_elbow_rank`` instead.
    """
    finite = np.where(np.isfinite(mean_curve) & np.isfinite(sem_curve))[0]
    if finite.size == 0:
        return _argmin_rank(ranks, mean_curve)
    argmin_idx = int(finite[np.argmin(mean_curve[finite])])
    threshold = float(mean_curve[argmin_idx] + sem_curve[argmin_idx])
    for i in finite:
        if mean_curve[i] <= threshold:
            return int(ranks[i])
    return int(ranks[argmin_idx])


# ---------------------------------------------------------------------------
# Plot styling helpers
# ---------------------------------------------------------------------------


def _new_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title, fontsize=12, pad=10, weight="semibold")
    return fig, ax


def _finalize(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    legend_anchor: tuple[float, float] = (0.98, 0.98),
    legend_align: tuple[str, str] = ("right", "top"),
    legend_outside: bool = False,
) -> None:
    """Render legend at an explicit anchor point.

    `loc="best"` is a heuristic that frequently overlaps data. Anchored
    placement is deterministic. When ``legend_outside=True``, the legend
    is placed to the right of the axes, useful for plots with many
    entries where no inside quadrant is empty.
    """
    if legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            bbox_transform=ax.transAxes,
            frameon=False,
            fontsize=9.5,
        )
    else:
        vertical = {"top": "upper", "bottom": "lower"}[legend_align[1]]
        ax.legend(
            loc=f"{vertical} {legend_align[0]}",
            bbox_to_anchor=legend_anchor,
            bbox_transform=ax.transAxes,
            frameon=False,
            fontsize=9.5,
        )
    despine(ax)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    fig.savefig(path, **PNG_KW)
    plt.close(fig)
    log.info("saved %s", path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_eigenvalues(
    est: RankEstimator, true_rank: int, path: Path, n_show: int = 30
) -> None:
    evals = np.asarray(est.eigenvalues_, dtype=float)
    n_show = min(n_show, len(evals))
    ranks = np.arange(1, n_show + 1)

    fig, ax = _new_axes("Reference Eigenvalues")
    ax.plot(ranks, evals[:n_show], "o-", color=INDIGO, ms=6, lw=1.6)
    ax.axvline(true_rank, color=TEAL, ls=":", lw=1.8, label=f"True rank = {true_rank}")
    ax.axvline(est.rank_, color=ROSE, ls="--", lw=1.6, label=f"rank_ = {est.rank_}")
    ax.set_xlabel("Dimension Index", fontsize=11)
    ax.set_ylabel("Eigenvalue", fontsize=11)
    ax.set_yscale("log")
    _finalize(fig, ax, path, legend_anchor=(0.98, 0.98), legend_align=("right", "top"))


def plot_leakage(est: RankEstimator, true_rank: int, path: Path) -> None:
    leakage = np.asarray(est.leakage_, dtype=float)
    ranks = np.arange(1, len(leakage) + 1)

    fig, ax = _new_axes("Per-Dimension Leakage Profile")
    ax.plot(ranks, leakage, "o-", color=INDIGO, ms=5, lw=1.4)
    ax.axvline(true_rank, color=TEAL, ls=":", lw=1.8, label=f"True rank = {true_rank}")
    ax.axvline(est.rank_, color=ROSE, ls="--", lw=1.6, label=f"rank_ = {est.rank_}")
    if np.all(leakage > 0):
        ax.set_yscale("log")
    ax.set_xlabel("Dimension Index", fontsize=11)
    ax.set_ylabel("Leakage (Lower = Signal)", fontsize=11)
    _finalize(fig, ax, path, legend_anchor=(0.98, 0.02), legend_align=("right", "bottom"))


def plot_recovery(est: RankEstimator, path: Path) -> None:
    p_grid = np.asarray(est.sampling_grid_, dtype=float)
    raw = np.asarray(est.recovery_raw_, dtype=float)
    monotone = np.asarray(est.recovery_monotone_, dtype=float)

    fig, ax = _new_axes("Spectral Recovery vs Sampling Fraction")
    ax.plot(p_grid, raw, "s", color=GRAY, ms=6, alpha=0.55, label="Raw")
    ax.plot(p_grid, monotone, "o-", color=INDIGO, ms=5, lw=1.6, label="Monotone")
    ax.axhline(
        est.recovery_tolerance, color=SAND, ls=":", lw=1.5,
        label=f"Tolerance = {est.recovery_tolerance:.2f}",
    )
    ax.axvline(
        est.sampling_fraction_, color=TEAL, ls="--", lw=2.0,
        label=f"sampling_fraction_ = {est.sampling_fraction_:.3f}",
    )
    if est.detectability_floor_ > p_grid.min():
        ax.axvline(
            est.detectability_floor_, color=ROSE, ls="-.", lw=1.4,
            label=f"Detectability floor = {est.detectability_floor_:.3f}",
        )
    ax.set_xlabel("Sampling Fraction $p$", fontsize=11)
    ax.set_ylabel("Recovery Deficit (Lower = Better)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=-0.02)
    _finalize(fig, ax, path, legend_outside=True)


def plot_cv_curve(
    curve: pd.DataFrame,
    est_rank: int,
    true_rank: int,
    argmin_rank: int,
    one_se_rank: int,
    sampling_fraction: float,
    path: Path,
) -> None:
    """Validation MSE vs rank: individual folds + bold mean + selection markers."""
    mean_curve = curve.groupby("rank")["score"].mean()
    sem_curve = (
        curve.groupby("rank")["score"].std()
        / np.sqrt(curve.groupby("rank")["fold"].count())
    )
    ranks = mean_curve.index.to_numpy()

    fig, ax = _new_axes(
        f"5-Fold Cross-Validation at $p = {sampling_fraction:.3f}$"
    )
    for _, fold_df in curve.groupby("fold"):
        fold_df = fold_df.sort_values("rank")
        ax.plot(
            fold_df["rank"], fold_df["score"], "-",
            color=GRAY_LIGHT, lw=0.9, alpha=0.7,
        )
    ax.fill_between(
        ranks,
        mean_curve.to_numpy() - sem_curve.to_numpy(),
        mean_curve.to_numpy() + sem_curve.to_numpy(),
        color=INDIGO, alpha=0.15,
    )
    ax.plot(
        ranks, mean_curve.to_numpy(), "o-",
        color=INDIGO, ms=6, lw=2.0, label="Mean ± SEM across folds",
    )
    ax.axvline(true_rank, color=TEAL, ls=":", lw=1.8, label=f"True rank = {true_rank}")
    ax.axvline(est_rank, color=ROSE, ls="--", lw=1.5, label=f"rank_ = {est_rank}")
    ax.axvline(one_se_rank, color=SAND, ls="-", lw=1.8, label=f"1-SE rank = {one_se_rank}")
    if argmin_rank != one_se_rank:
        ax.axvline(
            argmin_rank, color=GRAY, ls="-.", lw=1.2,
            label=f"Bare argmin = {argmin_rank}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Rank", fontsize=11)
    ax.set_ylabel("Validation MSE (log)", fontsize=11)
    _finalize(fig, ax, path, legend_outside=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _verdict_line(
    est_rank: int, true_rank: int, one_se_rank: int, argmin_rank: int
) -> str:
    truth_match = (
        "matches truth"
        if est_rank == true_rank
        else f"off truth by {abs(est_rank - true_rank)}"
    )
    cv_match = (
        "agrees with rank_"
        if one_se_rank == est_rank
        else f"differs by {abs(one_se_rank - est_rank)}"
    )
    return (
        f"# VERDICT: rank_={est_rank} ({truth_match}); "
        f"1-SE rank={one_se_rank} ({cv_match}); bare argmin={argmin_rank}"
    )


def _format_summary(
    est: RankEstimator,
    true_rank: int,
    curve: pd.DataFrame,
    one_se_rank: int,
    argmin_rank: int,
) -> str:
    mean_curve = curve.groupby("rank")["score"].mean()
    sem_curve = (
        curve.groupby("rank")["score"].std()
        / np.sqrt(curve.groupby("rank")["fold"].count())
    )

    lines = [_verdict_line(est.rank_, true_rank, one_se_rank, argmin_rank), ""]
    lines.append("# RankEstimator")
    lines.append(f"  n_features_in_           = {est.n_features_in_}")
    lines.append(f"  true rank (ground truth) = {true_rank}")
    lines.append(f"  rank_                    = {est.rank_}")
    lines.append(f"  sampling_fraction_       = {est.sampling_fraction_:.4f}")
    lines.append(f"  detectability_floor_     = {est.detectability_floor_:.4f}")
    lines.append(f"  recovery_tolerance       = {est.recovery_tolerance:.4f}")
    lines.append("")
    lines.append("# 5-fold CV at sampling_fraction_")
    lines.append(f"  Selected rank (1-SE)     = {one_se_rank}")
    lines.append(f"  Bare argmin              = {argmin_rank}")
    lines.append(f"  Ranks evaluated          = {list(mean_curve.index)}")
    lines.append("")
    lines.append(f"  {'rank':>4}  {'mean V-MSE':>14}  {'SEM V-MSE':>14}")
    for rank in mean_curve.index:
        lines.append(
            f"  {rank:4d}  {mean_curve[rank]:14.4e}  {sem_curve[rank]:14.4e}"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    setup_style()

    n, true_rank = 200, 10
    s = make_block_similarity(n=n, true_rank=true_rank)
    log.info("synthetic: shape=%s, true rank=%d", s.shape, true_rank)

    log.info("fitting RankEstimator ...")
    est = RankEstimator(recovery_tolerance=0.10, n_bootstrap=20, random_state=0).fit(s)
    log.info(
        "rank_=%d  sampling_fraction_=%.4f  detectability_floor_=%.4f",
        est.rank_, est.sampling_fraction_, est.detectability_floor_,
    )

    ranks = _rank_grid(n, est.rank_, true_rank)
    log.info(
        "cross_val_score: %d ranks (1..%d) at sampling_fraction_=%.4f",
        len(ranks), ranks[-1], est.sampling_fraction_,
    )
    curve = cross_val_score(
        s, ranks=ranks,
        sampling_fraction=est.sampling_fraction_,
        n_folds=5, random_state=0,
    )

    mean_curve = curve.groupby("rank")["score"].mean()
    sem_curve = (
        curve.groupby("rank")["score"].std()
        / np.sqrt(curve.groupby("rank")["fold"].count())
    )
    argmin_rank = _argmin_rank(ranks, mean_curve.to_numpy())
    one_se_rank = _argmin_1se_rank(ranks, mean_curve.to_numpy(), sem_curve.to_numpy())
    log.info("1-SE rank=%d  bare argmin=%d", one_se_rank, argmin_rank)

    plot_eigenvalues(est, true_rank, OUTPUT_DIR / "01_eigenvalues.png")
    plot_leakage(est, true_rank, OUTPUT_DIR / "02_leakage.png")
    plot_recovery(est, OUTPUT_DIR / "03_recovery.png")
    plot_cv_curve(
        curve, est.rank_, true_rank, argmin_rank, one_se_rank,
        est.sampling_fraction_, OUTPUT_DIR / "04_cv_curve.png",
    )

    summary = _format_summary(est, true_rank, curve, one_se_rank, argmin_rank)
    (OUTPUT_DIR / "summary.txt").write_text(summary + "\n")
    log.info("saved %s", OUTPUT_DIR / "summary.txt")
    log.info("\n%s", summary)


if __name__ == "__main__":
    main()
