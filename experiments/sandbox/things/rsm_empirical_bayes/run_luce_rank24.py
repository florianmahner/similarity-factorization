from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from scipy.optimize import brentq, minimize
from scipy.special import expit, logsumexp

from src.similarity.triplet_rsm import build_bias_aware_triplet_matrix
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
        default=PROJECT_ROOT / "data" / "things" / "triplets_47",
    )
    parser.add_argument("--n-objects", type=int, default=1854)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-tune-file", type=str, default="train_90.txt")
    parser.add_argument("--eval-tune-file", type=str, default="test_10.txt")
    parser.add_argument("--train-final-file", type=str, default="trainset.txt")
    parser.add_argument("--eval-final-file", type=str, default="validationset.txt")
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
    parser.add_argument("--luce-regs", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--prior-strengths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--inits", nargs="+", choices=["random", "random_sqrt"], default=["random", "random_sqrt"])
    parser.add_argument("--tune-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--max-inner", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=72)
    parser.add_argument("--tune-output", type=Path, default=OUTPUT_DIR / "train47_luce_rank24_tune_test10.csv")
    parser.add_argument("--final-output", type=Path, default=OUTPUT_DIR / "train47_luce_rank24_validation.csv")
    return parser.parse_args()


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(np.int32)


def count_pair_wins_trials(n_objects: int, triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    nn = n_objects * n_objects

    wins = np.zeros(nn, dtype=np.float64)
    wins += np.bincount(i * n_objects + j, minlength=nn)
    wins += np.bincount(j * n_objects + i, minlength=nn)
    wins = wins.reshape(n_objects, n_objects)

    trials = np.zeros(nn, dtype=np.float64)
    for a, b in ((i, j), (i, k), (j, k)):
        trials += np.bincount(a * n_objects + b, minlength=nn)
        trials += np.bincount(b * n_objects + a, minlength=nn)
    trials = trials.reshape(n_objects, n_objects)
    return wins, trials


def triplet_embedding_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    ei = embedding[triplets[:, 0]]
    ej = embedding[triplets[:, 1]]
    ek = embedding[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def triplet_matrix_accuracy(matrix: np.ndarray, triplets: np.ndarray) -> float:
    sij = matrix[triplets[:, 0], triplets[:, 1]]
    sik = matrix[triplets[:, 0], triplets[:, 2]]
    sjk = matrix[triplets[:, 1], triplets[:, 2]]
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_luce_worth(
    triplets: np.ndarray,
    n_items: int,
    reg: float,
    maxiter: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    i = triplets[:, 0]
    j = triplets[:, 1]
    k = triplets[:, 2]

    init_counts = np.bincount(np.concatenate([i, j]), minlength=n_items).astype(np.float64)
    init_exposure = np.bincount(np.concatenate([i, j, k]), minlength=n_items).astype(np.float64)
    init_rate = (init_counts + 1.0) / (init_exposure + 2.0)
    init_beta = np.log(init_rate)
    init_beta -= init_beta.mean()
    x0 = init_beta[:-1]

    def unpack(theta: np.ndarray) -> np.ndarray:
        beta = np.empty(n_items, dtype=np.float64)
        beta[:-1] = theta
        beta[-1] = -theta.sum()
        return beta

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = unpack(theta)
        u_ij = beta[i] + beta[j]
        u_ik = beta[i] + beta[k]
        u_jk = beta[j] + beta[k]
        logz = logsumexp(np.stack([u_ij, u_ik, u_jk], axis=0), axis=0)
        ll = np.sum(u_ij - logz)
        p_ij = np.exp(u_ij - logz)
        p_ik = np.exp(u_ik - logz)
        p_jk = np.exp(u_jk - logz)
        grad = np.zeros(n_items, dtype=np.float64)
        grad += np.bincount(i, weights=p_jk, minlength=n_items)
        grad += np.bincount(j, weights=p_ik, minlength=n_items)
        grad += np.bincount(k, weights=p_ij - 1.0, minlength=n_items)
        ll -= 0.5 * reg * np.dot(beta, beta)
        grad -= reg * beta
        return -ll, -(grad[:-1] - grad[-1])

    result = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    beta = unpack(result.x)
    beta -= beta.mean()
    meta = {
        "luce_reg": reg,
        "opt_success": float(result.success),
        "opt_nit": float(result.nit),
    }
    return beta, meta


def fit_luce_prior(
    triplets: np.ndarray,
    wins: np.ndarray,
    trials: np.ndarray,
    n_objects: int,
    reg: float,
) -> tuple[np.ndarray, dict[str, float]]:
    beta, meta = fit_luce_worth(triplets=triplets, n_items=n_objects, reg=reg, maxiter=20)
    obs = (trials > 0) & ~np.eye(n_objects, dtype=bool)
    observed_prob = np.divide(
        wins + 1.0,
        trials + 2.0,
        out=np.full_like(wins, np.nan),
        where=trials > 0,
    )
    luce_score = beta[:, None] + beta[None, :]
    z = luce_score[obs]
    y = observed_prob[obs]
    w = trials[obs]
    intercept = brentq(lambda c: float(np.sum(w * (expit(c + z) - y))), -20.0, 20.0)
    prior = expit(intercept + luce_score)
    np.fill_diagonal(prior, 1.0)
    meta["intercept"] = float(intercept)
    return prior, meta


def fit_srf_task(
    wins: np.ndarray,
    trials: np.ndarray,
    prior: np.ndarray,
    eval_triplets: np.ndarray,
    rank: int,
    alpha: float,
    luce_reg: float,
    prior_strength: float,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> dict[str, float | str | int]:
    similarity = build_bias_aware_triplet_matrix(
        wins=wins,
        trials=trials,
        prior=prior,
        alpha=alpha,
        prior_strength=prior_strength,
        fill_missing_with_prior=True,
    )
    np.fill_diagonal(similarity, np.nan)
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
    embedding = model.fit_transform(similarity)
    return {
        "luce_reg": luce_reg,
        "prior_strength": prior_strength,
        "rho": rho,
        "init": init,
        "seed": seed,
        "val_acc": triplet_embedding_accuracy(embedding, eval_triplets),
        "n_iter": getattr(model, "n_iter_", np.nan),
    }


def main() -> None:
    args = parse_args()
    args.tune_output.parent.mkdir(parents=True, exist_ok=True)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)

    train_tune = load_triplets(args.data_dir / args.train_tune_file)
    eval_tune = load_triplets(args.data_dir / args.eval_tune_file)
    train_final = load_triplets(args.data_dir / args.train_final_file)
    eval_final = load_triplets(args.data_dir / args.eval_final_file)

    log.info("Tune protocol: %s -> %s", args.train_tune_file, args.eval_tune_file)
    wins_tune, trials_tune = count_pair_wins_trials(args.n_objects, train_tune)
    priors_tune: dict[float, np.ndarray] = {}
    prior_meta_tune: dict[float, dict[str, float]] = {}
    for reg in args.luce_regs:
        prior, meta = fit_luce_prior(
            triplets=train_tune,
            wins=wins_tune,
            trials=trials_tune,
            n_objects=args.n_objects,
            reg=reg,
        )
        priors_tune[reg] = prior
        prior_meta_tune[reg] = meta
        log.info("Luce prior fit: reg=%g nit=%d success=%s", reg, int(meta["opt_nit"]), bool(meta["opt_success"]))

    tune_jobs = [
        delayed(fit_srf_task)(
            wins_tune,
            trials_tune,
            priors_tune[reg],
            eval_tune,
            args.rank,
            args.alpha,
            reg,
            prior_strength,
            rho,
            init,
            seed,
            args.max_outer,
            args.max_inner,
            args.tol,
        )
        for reg in args.luce_regs
        for prior_strength in args.prior_strengths
        for rho in args.rhos
        for init in args.inits
        for seed in args.tune_seeds
    ]
    log.info("Tune tasks: %d", len(tune_jobs))
    tune_df = pd.DataFrame(Parallel(n_jobs=args.n_jobs, verbose=10)(tune_jobs))
    for reg, meta in prior_meta_tune.items():
        for key, value in meta.items():
            tune_df.loc[tune_df["luce_reg"] == reg, key] = value
    tune_df.to_csv(args.tune_output, index=False)

    tune_summary = (
        tune_df.groupby(["luce_reg", "prior_strength", "rho", "init"], as_index=False)
        .agg(mean_val_acc=("val_acc", "mean"), std_val_acc=("val_acc", "std"))
        .sort_values(["mean_val_acc", "std_val_acc"], ascending=[False, True])
    )
    best = tune_summary.iloc[0]
    log.info("Best tune config: %s", best.to_dict())

    best_reg = float(best["luce_reg"])
    best_prior_strength = float(best["prior_strength"])
    best_rho = float(best["rho"])
    best_init = str(best["init"])

    log.info("Final protocol: %s -> %s", args.train_final_file, args.eval_final_file)
    wins_final, trials_final = count_pair_wins_trials(args.n_objects, train_final)
    prior_final, prior_meta_final = fit_luce_prior(
        triplets=train_final,
        wins=wins_final,
        trials=trials_final,
        n_objects=args.n_objects,
        reg=best_reg,
    )
    final_jobs = [
        delayed(fit_srf_task)(
            wins_final,
            trials_final,
            prior_final,
            eval_final,
            args.rank,
            args.alpha,
            best_reg,
            best_prior_strength,
            best_rho,
            best_init,
            seed,
            args.max_outer,
            args.max_inner,
            args.tol,
        )
        for seed in args.final_seeds
    ]
    final_df = pd.DataFrame(
        Parallel(n_jobs=min(args.n_jobs, len(final_jobs)), verbose=10)(final_jobs)
    )
    for key, value in prior_meta_final.items():
        final_df[key] = value
    final_df["model"] = "SRF_LUCE"
    final_df["rank"] = args.rank

    spose = np.maximum(np.loadtxt(args.spose_file), 0.0)
    vice = np.maximum(np.loadtxt(args.vice_file), 0.0)
    baseline_df = pd.DataFrame(
        [
            {
                "model": "SPoSE",
                "rank": spose.shape[1],
                "seed": 0,
                "val_acc": triplet_embedding_accuracy(spose, eval_final),
            },
            {
                "model": "VICE",
                "rank": vice.shape[1],
                "seed": 0,
                "val_acc": triplet_embedding_accuracy(vice, eval_final),
            },
        ]
    )
    final_out = pd.concat([baseline_df, final_df], ignore_index=True, sort=False)
    final_out.to_csv(args.final_output, index=False)

    log.info("\nTune summary top 10:")
    log.info("%s", tune_summary.head(10).to_string(index=False))
    log.info("\nFinal results:")
    log.info("%s", final_out.to_string(index=False))


if __name__ == "__main__":
    main()
