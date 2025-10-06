"""
Script to test how rho actually affects the rank recovery and the estimated error.
Needs more work.
"""

import numpy as np
from typing import Sequence, Iterable
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tools.rsa import compute_similarity
from models.admm import ADMM
from experiments.cross_validation import train_val_split
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass(slots=True)
class AdmmConfig:
    rho: float = 1.0
    max_outer: int = 8
    w_inner: int = 25
    tol: float = 1e-4


def _evaluate_single_trial(
    n: int,
    true_rank: int,
    candidate_ranks: Sequence[int],
    mask_frac: float,
    rho: float,
    noise: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    # generate synthetic ground truth
    w_true = rng.random((n, true_rank))
    s_full = compute_similarity(w_true, w_true, "linear")
    if noise > 0.0:
        s_full += noise * rng.standard_normal(s_full.shape)
        s_full[s_full < 0.0] = 0.0  # keep non-negative

    train_mask, val_mask = train_val_split(n, mask_frac, rng)

    cfg = AdmmConfig(rho=rho)
    best_rank, best_rmse = None, np.inf

    for rank in candidate_ranks:
        admm = ADMM(
            rank=rank, max_outer=cfg.max_outer, w_inner=cfg.w_inner, tol=cfg.tol
        )
        w_est = admm.fit_transform(s_full, train_mask)
        s_pred = compute_similarity(w_est, w_est, "linear")
        rmse = np.linalg.norm(val_mask * (s_full - s_pred), "fro") / np.sqrt(
            val_mask.sum()
        )
        if rmse < best_rmse:
            best_rmse, best_rank = rmse, rank

    return {
        "mask_frac": mask_frac,
        "rho": rho,
        "noise": noise,
        "success": int(best_rank == true_rank),
    }


# ---------------------------------------------------------------------
# 5.  Public grid-sweep function
# ---------------------------------------------------------------------
def run_rank_recovery_sweep(
    n: int,
    true_rank: int,
    candidate_ranks: Iterable[int],
    mask_fractions: Iterable[float],
    rho_values: Iterable[float],
    noise_levels: Iterable[float],
    repetitions: int = 10,
    n_jobs: int | None = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a full parameter sweep in parallel.

    Returns
    -------
    df : pandas.DataFrame
        Columns = [mask_frac, rho, noise, success_rate].
    """
    tasks = []
    rng_master = np.random.default_rng(12345)

    for noise in noise_levels:
        for mask_frac in mask_fractions:
            for rho in rho_values:
                for _ in range(repetitions):
                    seed = int(rng_master.integers(1e9))
                    tasks.append((mask_frac, rho, noise, seed))

    def _run(mask_frac, rho, noise, seed):
        return _evaluate_single_trial(
            n, true_rank, list(candidate_ranks), mask_frac, rho, noise, seed
        )

    iterator = tasks
    if verbose:
        iterator = tqdm(tasks, desc="sweep")

    results = Parallel(n_jobs=n_jobs, mmap_mode="c", verbose=10)(
        delayed(_run)(m, r, z, s) for m, r, z, s in iterator
    )

    df = pd.DataFrame(results)
    # aggregate into success rate
    df = df.groupby(["mask_frac", "rho", "noise"], as_index=False).agg(
        success_rate=("success", "mean")
    )
    return df


def plot_success_heatmaps(
    df: pd.DataFrame,
    row: str = "mask_frac",
    col: str = "rho",
    hue: str = "success_rate",
    noise_levels: Sequence[float] | None = None,
    cmap: str = "viridis",
) -> None:
    import seaborn as sns

    if noise_levels is None:
        noise_levels = sorted(df["noise"].unique())

    for noise in noise_levels:
        sub = (
            df[df["noise"] == noise]
            .pivot(index=row, columns=col, values=hue)
            .sort_index(ascending=True)
            .sort_index(axis=1, ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            sub,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "success rate"},
            xticklabels=2,
            yticklabels=2,
        )

        # Format x tick labels
        ax.set_xticklabels(
            [f"{float(label.get_text()):.2f}" for label in ax.get_xticklabels()]
        )
        # Format y tick labels
        ax.set_yticklabels(
            [f"{float(label.get_text()):.2f}" for label in ax.get_yticklabels()]
        )
        plt.title(f"Rank-recovery success\nnoise = {noise:.2f}")
        plt.tight_layout()
        plt.show()
