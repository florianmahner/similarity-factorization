from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "things" / "triplets_147",
    )
    parser.add_argument("--n-objects", type=int, default=1854)
    parser.add_argument("--train-file", type=str, default="trainset.txt")
    parser.add_argument("--eval-file", type=str, default="validationset.txt")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ranks", nargs="+", type=int, default=[24, 40, 49])
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--init", choices=["random", "random_sqrt"], default="random_sqrt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-outer", type=int, default=30)
    parser.add_argument("--max-inner", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--output-prefix", type=str, required=True)
    return parser.parse_args()


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(np.int32)


def canonical_triplet_counts(triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_triplets = np.sort(triplets, axis=1)
    chosen = np.sort(triplets[:, :2], axis=1)

    choice_idx = np.full(len(triplets), -1, dtype=np.int8)
    ab = (chosen[:, 0] == sorted_triplets[:, 0]) & (chosen[:, 1] == sorted_triplets[:, 1])
    ac = (chosen[:, 0] == sorted_triplets[:, 0]) & (chosen[:, 1] == sorted_triplets[:, 2])
    choice_idx[ab] = 0
    choice_idx[ac] = 1
    choice_idx[choice_idx < 0] = 2

    triple_view = np.ascontiguousarray(sorted_triplets).view(
        np.dtype([("a", np.int32), ("b", np.int32), ("c", np.int32)])
    ).reshape(-1)
    unique_triples, inverse = np.unique(triple_view, return_inverse=True)
    triples = np.column_stack([unique_triples["a"], unique_triples["b"], unique_triples["c"]]).astype(np.int32)

    counts = np.bincount(inverse * 3 + choice_idx, minlength=len(triples) * 3).reshape(len(triples), 3)
    return triples, counts.astype(np.float64)


def build_context_lift_matrix(
    n_objects: int,
    triplets: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    triples, counts = canonical_triplet_counts(triplets)
    local = np.log(counts + alpha)
    local -= local.mean(axis=1, keepdims=True)
    lift = np.exp(local)

    sums = np.zeros((n_objects, n_objects), dtype=np.float64)
    num = np.zeros((n_objects, n_objects), dtype=np.float64)

    a = triples[:, 0]
    b = triples[:, 1]
    c = triples[:, 2]
    pairs = (
        (a, b, lift[:, 0]),
        (a, c, lift[:, 1]),
        (b, c, lift[:, 2]),
    )
    for i, j, values in pairs:
        np.add.at(sums, (i, j), values)
        np.add.at(sums, (j, i), values)
        np.add.at(num, (i, j), 1.0)
        np.add.at(num, (j, i), 1.0)

    matrix = np.divide(
        sums,
        num,
        out=np.full_like(sums, np.nan),
        where=num > 0,
    )
    np.fill_diagonal(matrix, np.nan)
    return matrix, num


def triplet_matrix_accuracy(matrix: np.ndarray, triplets: np.ndarray) -> float:
    sij = matrix[triplets[:, 0], triplets[:, 1]]
    sik = matrix[triplets[:, 0], triplets[:, 2]]
    sjk = matrix[triplets[:, 1], triplets[:, 2]]
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_task(
    matrix: np.ndarray,
    triplets: np.ndarray,
    rank: int,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> dict[str, float | int]:
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
    embedding = model.fit_transform(matrix)
    dot = embedding @ embedding.T
    return {
        "rank": rank,
        "val_acc": triplet_matrix_accuracy(dot, triplets),
        "n_iter": int(getattr(model, "n_iter_", -1)),
        "mean_norm": float(np.linalg.norm(embedding, axis=1).mean()),
    }


def main() -> None:
    args = parse_args()
    output = OUTPUT_DIR / f"{args.output_prefix}.csv"

    train_triplets = load_triplets(args.data_dir / args.train_file)
    eval_triplets = load_triplets(args.data_dir / args.eval_file)

    matrix, coverage = build_context_lift_matrix(
        n_objects=args.n_objects,
        triplets=train_triplets,
        alpha=args.alpha,
    )
    matrix_baseline = triplet_matrix_accuracy(np.nan_to_num(matrix, nan=1.0), eval_triplets)

    log.info(
        "Context lift probe: dataset=%s ranks=%s matrix_baseline=%0.6f observed_fraction=%0.6f",
        args.data_dir.name,
        args.ranks,
        matrix_baseline,
        float(np.isfinite(matrix).mean()),
    )

    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_task)(
            matrix=matrix,
            triplets=eval_triplets,
            rank=rank,
            rho=args.rho,
            init=args.init,
            seed=args.seed,
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            tol=args.tol,
        )
        for rank in args.ranks
    )
    df = pd.DataFrame(rows).sort_values("rank")
    df["matrix_baseline"] = matrix_baseline
    df["observed_fraction"] = float(np.isfinite(matrix).mean())
    df["mean_contexts"] = float(np.nanmean(np.where(coverage > 0, coverage, np.nan)))
    df.to_csv(output, index=False)

    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
