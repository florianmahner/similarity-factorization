"""Generate final publication-quality imputation plot."""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from pysrf import SRF
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import r2_score

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

OUTPUT_DIR = get_output_dir()


def generate_dataset(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=1.0)
    return compute_similarity(w, w, "gaussian_kernel")


def mask_similarity(
    matrix: np.ndarray, fraction_retained: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    total = triu_i.shape[0]
    n_missing = int((1 - fraction_retained) * total)
    mask = np.zeros_like(matrix, dtype=bool)
    if n_missing > 0:
        choice = rng.choice(total, size=n_missing, replace=False)
        mask[triu_i[choice], triu_j[choice]] = True
        mask[triu_j[choice], triu_i[choice]] = True
    observed = matrix.copy()
    observed[mask] = np.nan
    np.fill_diagonal(observed, matrix.diagonal())
    return observed, mask


def impute_symmetric(imputer, observed: np.ndarray, original: np.ndarray) -> np.ndarray:
    n = observed.shape[0]
    if isinstance(imputer, SRF):
        imputer.fit(observed)
        imputed = imputer.reconstruct()
    else:
        imputed = imputer.fit_transform(observed)
    triu_i, triu_j = np.triu_indices(n, k=1)
    result = imputed.copy()
    result[triu_i, triu_j] = (imputed[triu_i, triu_j] + imputed[triu_j, triu_i]) / 2
    result[triu_j, triu_i] = result[triu_i, triu_j]
    np.fill_diagonal(result, np.diag(original))
    return result


def get_adaptive_rho(obs_per_dof: float) -> float:
    if obs_per_dof <= 1.0:
        return 0.01
    elif obs_per_dof <= 2.0:
        return 0.01 + (obs_per_dof - 1.0) * 0.04
    elif obs_per_dof <= 5.0:
        return 0.05 + (obs_per_dof - 2.0) / 3.0 * 0.45
    else:
        return min(3.0, 0.5 + (obs_per_dof - 5.0) / 10.0 * 2.5)


def evaluate(
    similarity: np.ndarray,
    n: int,
    k: int,
    fraction_retained: float,
    replicate: int,
    mask_seed: int,
) -> list[dict]:
    mask_rng = np.random.default_rng(mask_seed + replicate)
    observed, mask = mask_similarity(similarity, fraction_retained, mask_rng)
    if not mask.any():
        return []

    n_upper = n * (n - 1) // 2
    n_observed = int(fraction_retained * n_upper)
    dof = n * k
    obs_per_dof = n_observed / dof

    records = []
    base = {"obs_per_dof": obs_per_dof}

    # Median
    median_imp = SimpleImputer(strategy="median")
    median_filled = impute_symmetric(median_imp, observed, similarity)
    records.append({
        **base,
        "method": "Median",
        "r2": float(r2_score(similarity[mask], median_filled[mask])),
    })

    # KNN
    knn_imp = KNNImputer()
    knn_filled = impute_symmetric(knn_imp, observed, similarity)
    records.append({
        **base,
        "method": "KNN",
        "r2": float(r2_score(similarity[mask], knn_filled[mask])),
    })

    # SRF
    adaptive_rho = get_adaptive_rho(obs_per_dof)
    max_outer = 300 if obs_per_dof <= 2.0 else 150
    srf = SRF(
        rank=k,
        random_state=mask_seed + replicate,
        max_outer=max_outer,
        max_inner=500,
        tol=1e-5,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
        rho=adaptive_rho,
    )
    srf_filled = impute_symmetric(srf, observed, similarity)
    records.append({
        **base,
        "method": "SRF",
        "r2": float(r2_score(similarity[mask], srf_filled[mask])),
    })

    return records


def create_plot(agg: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
    })

    colors = {"Median": "#E69F00", "KNN": "#56B4E9", "SRF": "#009E73"}

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    # Shaded underdetermined region (ratio < 1)
    ax.axvspan(0.5, 1.0, color="#f0f0f0", alpha=1.0, zorder=0)
    ax.axvline(x=1.0, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)

    for method in ["Median", "KNN", "SRF"]:
        data = agg[agg["method"] == method].sort_values("obs_per_dof")
        c = colors[method]
        lw = 1.8 if method == "SRF" else 1.0
        zorder = 10 if method == "SRF" else 5

        ax.plot(
            data["obs_per_dof"], data["mean"] * 100,
            label=method, color=c, lw=lw, zorder=zorder
        )
        ax.fill_between(
            data["obs_per_dof"],
            (data["mean"] - data["sem"]) * 100,
            (data["mean"] + data["sem"]) * 100,
            color=c, alpha=0.2, linewidth=0
        )

    ax.set_xscale("log")
    ax.set_xlabel("Observations / degrees of freedom", fontsize=10)
    ax.set_ylabel("Held-out variance explained (%)", fontsize=10)

    ax.set_xlim(0.6, 25)
    ax.set_ylim(0, 105)
    ax.set_xticks([1, 1.5, 2, 3, 5, 10, 20])
    ax.set_xticklabels(["1", "1.5", "2", "3", "5", "10", "20"])
    ax.set_yticks([0, 25, 50, 75, 100])

    ax.legend(loc="lower right", frameon=False, handlelength=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "imputation_r2.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    k = 5
    n = 300
    # obs/dof = fraction * (n-1) / (2*k) = fraction * 29.9

    # Dense log-spaced points: more near 1-2, fewer at higher ratios
    # 30 points from 1.01-2 (critical region) + 25 points from 2-20
    ratios_near_1 = np.geomspace(1.01, 2.0, 30).tolist()
    ratios_higher = np.geomspace(2.0, 20.0, 25)[1:].tolist()  # skip 2.0 (already included)
    ratios_target = ratios_near_1 + ratios_higher

    fractions = [r / 29.9 for r in ratios_target]

    n_datasets = 5
    n_replicates = 3

    print(f"Config: n={n}, k={k}")
    print(f"Points: {len(ratios_target)} log-spaced from {min(ratios_target):.2f} to {max(ratios_target):.2f}")
    print(f"Samples per point: {n_datasets} datasets × {n_replicates} replicates = {n_datasets * n_replicates}")

    rng = np.random.default_rng(42)

    print("Generating datasets...")
    datasets = Parallel(n_jobs=-1)(
        delayed(generate_dataset)(n, k, int(s))
        for s in rng.integers(0, 1_000_000, size=n_datasets)
    )

    tasks = []
    mask_seed = int(rng.integers(0, 1_000_000))
    for didx, sim in enumerate(datasets):
        for frac in fractions:
            for rep in range(n_replicates):
                tasks.append((sim, n, k, frac, rep, mask_seed + didx * 1000))

    print(f"Running {len(tasks)} evaluations...")
    results = Parallel(n_jobs=-1)(delayed(evaluate)(*t) for t in tasks)
    all_results = [r for res in results for r in res]

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "imputation_results.csv", index=False)

    agg = df.groupby(["obs_per_dof", "method"])["r2"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])

    print(f"Creating plot...")
    create_plot(agg, OUTPUT_DIR)

    print(f"Saved to {OUTPUT_DIR}/imputation_r2.pdf")


if __name__ == "__main__":
    main()
