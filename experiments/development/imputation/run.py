#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from pysrf import SRF
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import r2_score

from utils.simulation import simulation_dirichlet
from tools.rsa import compute_similarity


def _build_single_dataset(
    args: argparse.Namespace, seed: int, idx: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(args.n, args.k, rng=rng, alpha=1.0)
    similarity = compute_similarity(w, w, "gaussian_kernel")
    return {
        "name": f"simulation_{idx}",
        "matrix": similarity,
        "rank": args.k,
        "snr": 1.0,
        "seed": seed,
    }


def build_similarity_datasets(
    args: argparse.Namespace, rng: np.random.Generator
) -> list[dict[str, object]]:
    seeds = rng.integers(0, 1_000_000, size=args.nseeds)
    return Parallel(n_jobs=args.n_jobs)(
        delayed(_build_single_dataset)(args, int(seed), idx)
        for idx, seed in enumerate(seeds)
    )


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


def evaluate_similarity(
    original: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    name: str,
    fraction_retained: float,
    replicate: int,
    seed: int,
    snr: float,
    rank: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    mean_imputer = SimpleImputer(strategy="mean")
    mean_filled = impute_symmetric(mean_imputer, observed, original)
    mse_mean = float(np.mean((mean_filled[mask] - original[mask]) ** 2))
    r2_mean = float(r2_score(original[mask], mean_filled[mask]))
    rows.append(
        {
            "dataset": name,
            "method": "Mean",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_mean,
            "r2": r2_mean,
            "snr": snr,
        }
    )

    median_imputer = SimpleImputer(strategy="median")
    median_filled = impute_symmetric(median_imputer, observed, original)
    mse_median = float(np.mean((median_filled[mask] - original[mask]) ** 2))
    r2_median = float(r2_score(original[mask], median_filled[mask]))
    rows.append(
        {
            "dataset": name,
            "method": "Median",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_median,
            "r2": r2_median,
            "snr": snr,
        }
    )

    knn_filled = KNNImputer()
    knn_filled = impute_symmetric(knn_filled, observed, original)
    mse_knn = float(np.mean((knn_filled[mask] - original[mask]) ** 2))
    r2_knn = float(r2_score(original[mask], knn_filled[mask]))
    rows.append(
        {
            "dataset": name,
            "method": "KNN",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_knn,
            "r2": r2_knn,
            "snr": snr,
        }
    )

    model = SRF(
        rank=rank,
        random_state=seed + 5,
        max_outer=100,
        max_inner=500,
        tol=0.0,
        init="random_sqrt",
        verbose=0,
        rho=0.001,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
    )
    srf_reconstruction = impute_symmetric(model, observed, original)
    mse_srf = float(np.mean((srf_reconstruction[mask] - original[mask]) ** 2))
    r2_srf = float(r2_score(original[mask], srf_reconstruction[mask]))
    rows.append(
        {
            "dataset": name,
            "method": "SRF",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_srf,
            "r2": r2_srf,
            "snr": snr,
        }
    )
    return rows


def process_task(
    dataset: dict[str, object],
    fraction_retained: float,
    replicate: int,
    mask_seed: int,
) -> list[dict[str, object]]:
    mask_rng = np.random.default_rng(mask_seed)
    observed, mask = mask_similarity(dataset["matrix"], fraction_retained, mask_rng)
    return evaluate_similarity(
        dataset["matrix"],
        observed,
        mask,
        dataset["name"],
        fraction_retained,
        replicate,
        dataset["seed"],
        dataset["snr"],
        dataset["rank"],
    )


def run(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    if args.fraction_retained:
        fractions_retained = args.fraction_retained
    else:
        # fractions_retained = list(np.unique(np.round(np.geomspace(0.1, 0.9, 14), 3)))
        # fractions_retained = [0.01, 0.050.1, 0.2, 0.4, 0.8]
        fractions_retained = [0.01, 0.05]
    print(f"Building {args.nseeds} datasets")
    datasets = build_similarity_datasets(args, rng)
    tasks: list[tuple[dict[str, object], float, int, int]] = []
    mask_seed = int(rng.integers(0, 1_000_000))
    for dataset in datasets:
        for fraction_retained in fractions_retained:
            for replicate in range(args.repeats):
                tasks.append((dataset, fraction_retained, replicate, mask_seed))
    print(f"Running {len(tasks)} tasks")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_task)(
            dataset,
            fraction_retained,
            replicate,
            mask_seed,
        )
        for dataset, fraction_retained, replicate, mask_seed in tasks
    )
    rows = [row for result in results for row in result]
    df = pd.DataFrame(rows)
    return df


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        "Mean",
        "Median",
        "KNN",
        "SRF",
    ]
    colors = {
        "Mean": "#F1C40F",
        "Median": "#E74C3C",
        "KNN": "#3498DB",
        "SRF": "#2ECC71",
    }
    facet = sns.relplot(
        data=df[df["method"].isin(methods)],
        x="fraction_retained",
        y="r2",
        hue="method",
        col="snr",
        kind="line",
        errorbar="sd",
        palette=colors,
        height=4,
        aspect=1.0,
        facet_kws={"sharey": False},
    )
    facet.set_axis_labels("Fraction retained", "R² (explained variance)")
    facet.set_titles("SNR {col_name:.2f}")
    legend = facet._legend
    if legend is not None:
        legend.set_title("")
    facet.figure.tight_layout()
    facet.figure.savefig(output_dir / "snr_comparison.png", bbox_inches="tight")
    plt.close(facet.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear kernel imputation experiment")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fraction-retained", type=float, nargs="+", default=[])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nseeds", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    df = run(args)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_dir / "results.csv", index=False)
        plot_results(df, args.output_dir)
    else:
        print(df)


if __name__ == "__main__":
    main()
