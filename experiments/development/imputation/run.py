#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysrf import SRF
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--missing-rate", type=float, default=0.75)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--srf-rank", type=int, default=20)
    return parser.parse_args()


def add_missing_values(
    x: np.ndarray, missing_rate: float, rng: np.random.Generator
) -> np.ndarray:
    n_samples, n_features = x.shape
    n_missing_samples = int(n_samples * missing_rate)
    mask = np.zeros(n_samples, dtype=bool)
    mask[:n_missing_samples] = True
    rng.shuffle(mask)
    missing_features = rng.integers(0, n_features, size=n_missing_samples)
    x_missing = x.copy()
    x_missing[mask, missing_features] = np.nan
    return x_missing


def score_imputer(
    x: np.ndarray, y: np.ndarray, imputer, n_splits: int, seed: int
) -> tuple[float, float]:
    regressor = RandomForestRegressor(random_state=seed)
    if imputer is None:
        estimator = regressor
    else:
        estimator = make_pipeline(imputer, regressor)
    scores = cross_val_score(
        estimator, x, y, scoring="neg_mean_squared_error", cv=n_splits
    )
    mse = -scores.mean()
    std = scores.std()
    return mse, std


def evaluate_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    missing_rate: float,
    n_splits: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    x_missing = add_missing_values(x, missing_rate, rng)
    rows = []

    mse_full, std_full = score_imputer(x, y, None, n_splits, seed)
    rows.append(
        {
            "dataset": name,
            "method": "full_data",
            "mse": mse_full,
            "std": std_full,
            "modality": "features",
        }
    )

    imputer_zero = SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)
    mse_zero, std_zero = score_imputer(x_missing, y, imputer_zero, n_splits, seed)
    rows.append(
        {
            "dataset": name,
            "method": "zero_imputation",
            "mse": mse_zero,
            "std": std_zero,
            "modality": "features",
        }
    )

    imputer_mean = SimpleImputer(strategy="mean", add_indicator=True)
    mse_mean, std_mean = score_imputer(x_missing, y, imputer_mean, n_splits, seed)
    rows.append(
        {
            "dataset": name,
            "method": "mean_imputation",
            "mse": mse_mean,
            "std": std_mean,
            "modality": "features",
        }
    )

    imputer_knn = KNNImputer(add_indicator=True)
    mse_knn, std_knn = score_imputer(x_missing, y, imputer_knn, n_splits, seed)
    rows.append(
        {
            "dataset": name,
            "method": "knn_imputation",
            "mse": mse_knn,
            "std": std_knn,
            "modality": "features",
        }
    )

    imputer_iter = IterativeImputer(add_indicator=True, random_state=seed)
    mse_iter, std_iter = score_imputer(x_missing, y, imputer_iter, n_splits, seed)
    rows.append(
        {
            "dataset": name,
            "method": "iterative_imputation",
            "mse": mse_iter,
            "std": std_iter,
            "modality": "features",
        }
    )

    return rows


def load_dataset(name: str, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "diabetes":
        x, y = load_diabetes(return_X_y=True)
    elif name == "california":
        x, y = fetch_california_housing(return_X_y=True)
    else:
        raise ValueError(f"unknown dataset: {name}")
    if x.shape[0] > max_samples:
        x = x[:max_samples]
        y = y[:max_samples]
    return x, y


def run(args: argparse.Namespace) -> pd.DataFrame:
    datasets = {
        "diabetes": load_dataset("diabetes", args.max_samples),
        "california": load_dataset("california", args.max_samples),
    }
    rows: list[dict[str, object]] = []
    for name, (x, y) in datasets.items():
        if name == "california":
            scaler = RobustScaler()
            x = scaler.fit_transform(x)
        rows.extend(
            evaluate_dataset(name, x, y, args.missing_rate, args.n_splits, args.seed)
        )
    rng = np.random.default_rng(args.seed)
    similarity_datasets = build_similarity_datasets(rng)
    for sim_name, full_matrix in similarity_datasets.items():
        observed, mask = mask_similarity(full_matrix, args.missing_rate, rng)
        rows.extend(
            evaluate_similarity(
                full_matrix,
                observed,
                mask,
                args.srf_rank,
                sim_name,
            )
        )
    df = pd.DataFrame(rows)
    return df


def mask_similarity(
    matrix: np.ndarray, missing_rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    upper = rng.uniform(size=(n, n)) < missing_rate
    missing = np.triu(upper, k=1)
    missing = missing | missing.T
    np.fill_diagonal(missing, False)
    observed = matrix.copy()
    observed[missing] = np.nan
    np.fill_diagonal(observed, matrix.diagonal())
    return observed, missing


def symmetrize(imputed: np.ndarray, original: np.ndarray) -> np.ndarray:
    sym = 0.5 * (imputed + imputed.T)
    np.fill_diagonal(sym, np.diag(original))
    return sym


def evaluate_similarity(
    original: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    rank: int,
    name: str,
) -> list[dict[str, object]]:
    rows = []

    base_imputer = SimpleImputer(strategy="mean")
    base_filled = base_imputer.fit_transform(observed)
    base_filled = symmetrize(base_filled, original)
    mse = float(np.mean((base_filled[mask] - original[mask]) ** 2))
    rows.append(
        {
            "dataset": name,
            "method": "mean_imputation",
            "mse": mse,
            "std": 0.0,
            "modality": "similarity",
        }
    )

    knn_filled = KNNImputer().fit_transform(observed)
    knn_filled = symmetrize(knn_filled, original)
    mse_knn = float(np.mean((knn_filled[mask] - original[mask]) ** 2))
    rows.append(
        {
            "dataset": name,
            "method": "knn_imputation",
            "mse": mse_knn,
            "std": 0.0,
            "modality": "similarity",
        }
    )

    iter_filled = IterativeImputer(random_state=0).fit_transform(observed)
    iter_filled = symmetrize(iter_filled, original)
    mse_iter = float(np.mean((iter_filled[mask] - original[mask]) ** 2))
    rows.append(
        {
            "dataset": name,
            "method": "iterative_imputation",
            "mse": mse_iter,
            "std": 0.0,
            "modality": "similarity",
        }
    )

    model = SRF(rank=rank, rho=3.0, max_outer=100, max_inner=50, tol=1e-3)
    model.fit(observed)
    srf_recon = model.reconstruct()
    mse_srf = float(np.mean((srf_recon[mask] - original[mask]) ** 2))
    rows.append(
        {
            "dataset": name,
            "method": "srf",
            "mse": mse_srf,
            "std": 0.0,
            "modality": "similarity",
        }
    )
    return rows


def build_similarity_datasets(rng: np.random.Generator) -> dict[str, np.ndarray]:
    datasets = {}
    random_matrix = rng.normal(size=(200, 200))
    covariance = random_matrix @ random_matrix.T
    covariance /= covariance.max()
    datasets["synthetic_covariance"] = covariance

    random_block = rng.normal(size=(150, 30))
    latent = random_block @ random_block.T
    latent /= latent.max()
    datasets["latent_similarity"] = latent
    return datasets


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    feature_methods = [
        "full_data",
        "zero_imputation",
        "mean_imputation",
        "knn_imputation",
        "iterative_imputation",
        "srf",
    ]
    sim_methods = [
        "mean_imputation",
        "knn_imputation",
        "iterative_imputation",
        "srf",
    ]
    labels = {
        "full_data": "Full data",
        "zero_imputation": "Zero imputation",
        "mean_imputation": "Mean imputation",
        "knn_imputation": "KNN imputation",
        "iterative_imputation": "Iterative imputation",
        "srf": "SRF",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    feature_df = df[df["modality"] == "features"]
    similarity_df = df[df["modality"] == "similarity"]
    feature_datasets = feature_df["dataset"].unique()
    if feature_datasets.size > 0:
        fig_feat, axes_feat = plt.subplots(
            1,
            feature_datasets.size,
            figsize=(5 * feature_datasets.size, 4),
            squeeze=False,
        )
        axes = axes_feat.flatten()
        for idx, dataset in enumerate(feature_datasets):
            subset = feature_df[feature_df["dataset"] == dataset]
            ax = axes[idx]
            y_pos = list(range(len(feature_methods)))
            mses = []
            stds = []
            for method in feature_methods:
                row = subset[subset["method"] == method]
                if row.empty:
                    mses.append(np.nan)
                    stds.append(np.nan)
                else:
                    mses.append(row["mse"].iloc[0])
                    stds.append(row["std"].iloc[0])
            ax.barh(
                y_pos, mses, xerr=stds, color=colors[: len(feature_methods)], alpha=0.7
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels([labels[m] for m in feature_methods])
            ax.invert_yaxis()
            ax.set_title(f"{dataset.capitalize()} features")
            ax.set_xlabel("Mean squared error")
        fig_feat.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_path = output_dir / "comparison_features.png"
        fig_feat.savefig(feature_path, bbox_inches="tight")
        plt.close(fig_feat)
    sim_datasets = similarity_df["dataset"].unique()
    if sim_datasets.size > 0:
        fig_sim, axes_sim = plt.subplots(
            1, sim_datasets.size, figsize=(5 * sim_datasets.size, 4), squeeze=False
        )
        axes = axes_sim.flatten()
        for idx, dataset in enumerate(sim_datasets):
            subset = similarity_df[similarity_df["dataset"] == dataset]
            ax = axes[idx]
            y_pos = list(range(len(sim_methods)))
            mses = []
            for method in sim_methods:
                row = subset[subset["method"] == method]
                mses.append(np.nan if row.empty else row["mse"].iloc[0])
            ax.barh(y_pos, mses, color=colors[2 : 2 + len(sim_methods)], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([labels[m] for m in sim_methods])
            ax.invert_yaxis()
            ax.set_title(f"{dataset} similarity")
            ax.set_xlabel("Mean squared error")
        fig_sim.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        sim_path = output_dir / "comparison_similarity.png"
        fig_sim.savefig(sim_path, bbox_inches="tight")
        plt.close(fig_sim)


def main() -> None:
    args = parse_args()
    df = run(args)
    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output_dir) / "results.csv"
        df.to_csv(out_path, index=False)
        plot_results(df, Path(args.output_dir))
    else:
        print(df)


if __name__ == "__main__":
    main()
