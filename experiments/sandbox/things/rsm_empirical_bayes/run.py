from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.similarity.triplet_rsm import (
    build_bias_aware_triplet_matrix,
    fit_triplet_bias_prior,
)
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "things" / "triplets_47")
    parser.add_argument("--n-objects", type=int, default=1854)
    parser.add_argument("--train-file", type=str, default="train_90.txt")
    parser.add_argument("--eval-file", type=str, default="validationset.txt")
    parser.add_argument(
        "--spose-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt",
    )
    parser.add_argument(
        "--vice-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior-strengths", nargs="+", type=float, default=[0.0, 5.0, 10.0, 20.0, 30.0])
    parser.add_argument("--ranks", nargs="+", type=int, default=[66])
    parser.add_argument("--rhos", nargs="+", type=float, default=[3.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--init", choices=["random", "random_sqrt"], default="random_sqrt")
    parser.add_argument("--missing-strategies", nargs="+", choices=["prior", "nan"], default=["prior", "nan"])
    parser.add_argument(
        "--diagonal-mode",
        choices=["one", "nan", "rowmean2", "rowmax1.1"],
        default="one",
    )
    parser.add_argument("--weight-mode", choices=["fisher", "shown"], default="fisher")
    parser.add_argument("--max-outer", type=int, default=100)
    parser.add_argument("--max-inner", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(int)


def triplet_matrix_accuracy(similarity: np.ndarray, triplets: np.ndarray) -> float:
    sij = similarity[triplets[:, 0], triplets[:, 1]]
    sik = similarity[triplets[:, 0], triplets[:, 2]]
    sjk = similarity[triplets[:, 1], triplets[:, 2]]
    return float(np.mean((sij > sik) & (sij > sjk)))


def triplet_embedding_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    ei = embedding[triplets[:, 0]]
    ej = embedding[triplets[:, 1]]
    ek = embedding[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_srf_embedding(
    similarity: np.ndarray,
    rank: int,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> np.ndarray:
    model = SRF(
        rank=rank,
        rho=rho,
        init=init,
        random_state=seed,
        max_outer=max_outer,
        max_inner=max_inner,
        tol=tol,
        verbose=0,
    )
    return model.fit_transform(similarity)


def build_similarity_variants(
    wins: np.ndarray,
    trials: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    prior_strengths: list[float],
    missing_strategies: list[str],
    diagonal_mode: str,
) -> dict[tuple[float, str], np.ndarray]:
    variants: dict[tuple[float, str], np.ndarray] = {}
    for prior_strength in prior_strengths:
        for missing_strategy in missing_strategies:
            similarity = build_bias_aware_triplet_matrix(
                wins=wins,
                trials=trials,
                prior=prior,
                alpha=alpha,
                prior_strength=prior_strength,
                fill_missing_with_prior=missing_strategy == "prior",
            )
            variants[(prior_strength, missing_strategy)] = apply_diagonal_mode(
                similarity=similarity,
                diagonal_mode=diagonal_mode,
            )
    return variants


def apply_diagonal_mode(similarity: np.ndarray, diagonal_mode: str) -> np.ndarray:
    adjusted = similarity.copy()
    if diagonal_mode == "one":
        np.fill_diagonal(adjusted, 1.0)
    elif diagonal_mode == "nan":
        np.fill_diagonal(adjusted, np.nan)
    elif diagonal_mode == "rowmean2":
        row_mean = np.nanmean(adjusted, axis=1)
        np.fill_diagonal(adjusted, row_mean * 2.0)
    elif diagonal_mode == "rowmax1.1":
        row_max = np.nanmax(adjusted, axis=1)
        np.fill_diagonal(adjusted, row_max * 1.1)
    else:
        raise ValueError(f"Unknown diagonal_mode: {diagonal_mode}")
    return adjusted


def main() -> None:
    args = parse_args()

    train_triplets = load_triplets(args.data_dir / args.train_file)
    validation_triplets = load_triplets(args.data_dir / args.eval_file)
    log.info("Train triplets: %d", len(train_triplets))
    log.info("Evaluation triplets: %d", len(validation_triplets))
    log.info("Protocol: dir=%s train=%s eval=%s", args.data_dir, args.train_file, args.eval_file)

    wins, trials, prior = fit_triplet_bias_prior(
        n_objects=args.n_objects,
        triplets=train_triplets,
        alpha=args.alpha,
        weight_mode=args.weight_mode,
    )

    similarities = build_similarity_variants(
        wins=wins,
        trials=trials,
        prior=prior,
        alpha=args.alpha,
        prior_strengths=args.prior_strengths,
        missing_strategies=args.missing_strategies,
        diagonal_mode=args.diagonal_mode,
    )

    matrix_records: list[dict[str, float | str]] = []
    for (prior_strength, missing_strategy), similarity in similarities.items():
        matrix_records.append(
            {
                "model": "matrix",
                "prior_strength": prior_strength,
                "missing_strategy": missing_strategy,
                "rank": np.nan,
                "rho": np.nan,
                "seed": np.nan,
                "val_acc": triplet_matrix_accuracy(
                    np.nan_to_num(similarity, nan=0.5),
                    validation_triplets,
                ),
                "nan_fraction": float(np.isnan(similarity).mean()),
                "diagonal_mode": args.diagonal_mode,
                "train_file": args.train_file,
                "eval_file": args.eval_file,
                "init": "matrix",
            }
        )

    matrix_df = pd.DataFrame(matrix_records)
    matrix_df.to_csv(OUTPUT_DIR / "matrix_results.csv", index=False)

    spose_embedding = np.maximum(np.loadtxt(args.spose_file), 0.0)

    embedding_records: list[dict[str, float | str]] = [
        {
            "model": "SPoSE",
            "prior_strength": np.nan,
            "missing_strategy": "baseline",
            "rank": spose_embedding.shape[1],
            "rho": np.nan,
            "seed": 0,
            "val_acc": triplet_embedding_accuracy(spose_embedding, validation_triplets),
            "nan_fraction": 0.0,
            "diagonal_mode": "embedding",
            "train_file": args.train_file,
            "eval_file": args.eval_file,
            "init": "embedding",
        },
    ]
    if args.vice_file.exists():
        vice_embedding = np.maximum(np.loadtxt(args.vice_file), 0.0)
        embedding_records.append(
            {
                "model": "VICE",
                "prior_strength": np.nan,
                "missing_strategy": "baseline",
                "rank": vice_embedding.shape[1],
                "rho": np.nan,
                "seed": 0,
                "val_acc": triplet_embedding_accuracy(vice_embedding, validation_triplets),
                "nan_fraction": 0.0,
                "diagonal_mode": "embedding",
                "train_file": args.train_file,
                "eval_file": args.eval_file,
                "init": "embedding",
            }
        )

    tasks = [
        (prior_strength, missing_strategy, rank, rho, seed, similarities[(prior_strength, missing_strategy)])
        for prior_strength in args.prior_strengths
        for missing_strategy in args.missing_strategies
        for rank in args.ranks
        for rho in args.rhos
        for seed in args.seeds
    ]
    log.info("SRF fits: %d", len(tasks))

    def run_task(
        prior_strength: float,
        missing_strategy: str,
        rank: int,
        rho: float,
        seed: int,
        similarity: np.ndarray,
    ) -> dict[str, float | str]:
        embedding = fit_srf_embedding(
            similarity=similarity,
            rank=rank,
            rho=rho,
            init=args.init,
            seed=seed,
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            tol=args.tol,
        )
        return {
            "model": "SRF",
            "prior_strength": prior_strength,
            "missing_strategy": missing_strategy,
            "rank": rank,
            "rho": rho,
            "seed": seed,
            "val_acc": triplet_embedding_accuracy(embedding, validation_triplets),
            "nan_fraction": float(np.isnan(similarity).mean()),
            "diagonal_mode": args.diagonal_mode,
            "train_file": args.train_file,
            "eval_file": args.eval_file,
            "init": args.init,
        }

    srf_records = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_task)(prior_strength, missing_strategy, rank, rho, seed, similarity)
        for prior_strength, missing_strategy, rank, rho, seed, similarity in tasks
    )
    embedding_records.extend(srf_records)

    embedding_df = pd.DataFrame(embedding_records)
    embedding_df.to_csv(OUTPUT_DIR / "embedding_results.csv", index=False)

    matrix_summary = matrix_df.sort_values("val_acc", ascending=False)
    log.info("\nMatrix accuracies:")
    for row in matrix_summary.itertuples(index=False):
        log.info(
            "  prior=%5s missing=%5s val=%.4f nan=%.4f",
            f"{row.prior_strength:g}",
            row.missing_strategy,
            row.val_acc,
            row.nan_fraction,
        )

    log.info("\nEmbedding accuracies:")
    for row in embedding_df.sort_values("val_acc", ascending=False).itertuples(index=False):
        log.info(
            "  %-5s prior=%5s missing=%8s diag=%10s init=%11s rank=%3s rho=%5s seed=%2s val=%.4f",
            row.model,
            "-" if pd.isna(row.prior_strength) else f"{row.prior_strength:g}",
            row.missing_strategy,
            row.diagonal_mode,
            row.init,
            "-" if pd.isna(row.rank) else int(row.rank),
            "-" if pd.isna(row.rho) else f"{row.rho:g}",
            "-" if pd.isna(row.seed) else int(row.seed),
            row.val_acc,
        )


if __name__ == "__main__":
    main()
