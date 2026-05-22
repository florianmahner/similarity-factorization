from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp

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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "things" / "triplets_147",
    )
    parser.add_argument("--n-objects", type=int, default=1854)
    parser.add_argument("--train-file", type=str, default="trainset.txt")
    parser.add_argument("--eval-file", type=str, default="validationset.txt")
    parser.add_argument("--prior-family", choices=["luce", "bias"], default="luce")
    parser.add_argument("--weight-mode", choices=["fisher", "shown"], default="shown")
    parser.add_argument("--luce-reg", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior-strength", type=float, default=0.5)
    parser.add_argument("--ranks", nargs="+", type=int, default=[24, 40, 49])
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--init", choices=["random", "random_sqrt"], default="random_sqrt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-outer", type=int, default=30)
    parser.add_argument("--max-inner", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--clip-eps", type=float, default=1e-4)
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=["prob", "odds", "prob_lift", "odds_lift"],
        default=["prob", "odds", "prob_lift", "odds_lift"],
    )
    parser.add_argument("--output-prefix", type=str, required=True)
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


def fit_luce_worth(
    triplets: np.ndarray,
    n_items: int,
    reg: float,
    max_iter: int = 20,
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

    result = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
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
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    beta, meta = fit_luce_worth(triplets=triplets, n_items=n_objects, reg=reg, max_iter=20)
    observed = (trials > 0) & ~np.eye(n_objects, dtype=bool)
    observed_prob = np.divide(
        wins + alpha,
        trials + 2.0 * alpha,
        out=np.full_like(wins, np.nan),
        where=trials > 0,
    )
    score = beta[:, None] + beta[None, :]
    intercept = brentq(
        lambda c: float(np.sum(trials[observed] * (expit(c + score[observed]) - observed_prob[observed]))),
        -20.0,
        20.0,
    )
    prior = expit(intercept + score)
    np.fill_diagonal(prior, 1.0)
    meta["intercept"] = float(intercept)
    return prior, meta


def build_transform_target(
    transform: str,
    similarity: np.ndarray,
    prior: np.ndarray,
    clip_eps: float,
) -> np.ndarray:
    clipped_similarity = np.clip(similarity, clip_eps, 1.0 - clip_eps)
    clipped_prior = np.clip(prior, clip_eps, 1.0 - clip_eps)

    if transform == "prob":
        target = clipped_similarity.copy()
    elif transform == "odds":
        target = clipped_similarity / (1.0 - clipped_similarity)
    elif transform == "prob_lift":
        target = clipped_similarity / clipped_prior
    elif transform == "odds_lift":
        target = np.exp(logit(clipped_similarity) - logit(clipped_prior))
    else:
        raise ValueError(transform)

    np.fill_diagonal(target, np.nan)
    return target


def score_triplets(
    transform: str,
    embedding: np.ndarray,
    triplets: np.ndarray,
    prior: np.ndarray,
    clip_eps: float,
) -> np.ndarray:
    dot = embedding @ embedding.T
    dot = np.maximum(dot, clip_eps)

    i = triplets[:, 0]
    j = triplets[:, 1]
    k = triplets[:, 2]

    if transform == "prob":
        score = dot
    elif transform == "odds":
        score = np.log(dot)
    elif transform == "prob_lift":
        score = np.log(np.clip(prior, clip_eps, None)) + np.log(dot)
    elif transform == "odds_lift":
        score = logit(np.clip(prior, clip_eps, 1.0 - clip_eps)) + np.log(dot)
    else:
        raise ValueError(transform)

    sij = score[i, j]
    sik = score[i, k]
    sjk = score[j, k]
    return (sij > sik) & (sij > sjk)


def fit_task(
    transform: str,
    target: np.ndarray,
    prior: np.ndarray,
    triplets: np.ndarray,
    rank: int,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
    clip_eps: float,
) -> dict[str, float | int | str]:
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
    embedding = model.fit_transform(target)
    correct = score_triplets(transform, embedding, triplets, prior, clip_eps)
    return {
        "transform": transform,
        "rank": rank,
        "val_acc": float(np.mean(correct)),
        "n_iter": int(getattr(model, "n_iter_", -1)),
        "mean_norm": float(np.linalg.norm(embedding, axis=1).mean()),
    }


def main() -> None:
    args = parse_args()
    output = OUTPUT_DIR / f"{args.output_prefix}.csv"

    train_triplets = load_triplets(args.data_dir / args.train_file)
    eval_triplets = load_triplets(args.data_dir / args.eval_file)
    wins, trials = count_pair_wins_trials(args.n_objects, train_triplets)

    if args.prior_family == "luce":
        prior, prior_meta = fit_luce_prior(
            triplets=train_triplets,
            wins=wins,
            trials=trials,
            n_objects=args.n_objects,
            reg=args.luce_reg,
            alpha=args.alpha,
        )
    else:
        wins, trials, prior = fit_triplet_bias_prior(
            n_objects=args.n_objects,
            triplets=train_triplets,
            alpha=args.alpha,
            weight_mode=args.weight_mode,
        )
        prior_meta = {"weight_mode": args.weight_mode}

    similarity = build_bias_aware_triplet_matrix(
        wins=wins,
        trials=trials,
        prior=prior,
        alpha=args.alpha,
        prior_strength=args.prior_strength,
        fill_missing_with_prior=True,
    )
    np.fill_diagonal(similarity, np.nan)

    targets = {
        transform: build_transform_target(
            transform=transform,
            similarity=similarity,
            prior=prior,
            clip_eps=args.clip_eps,
        )
        for transform in args.transforms
    }

    jobs = [
        delayed(fit_task)(
            transform=transform,
            target=targets[transform],
            prior=prior,
            triplets=eval_triplets,
            rank=rank,
            rho=args.rho,
            init=args.init,
            seed=args.seed,
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            tol=args.tol,
            clip_eps=args.clip_eps,
        )
        for transform in args.transforms
        for rank in args.ranks
    ]

    log.info(
        "Transform probe: dataset=%s transforms=%s ranks=%s prior=%s strength=%g",
        args.data_dir.name,
        args.transforms,
        args.ranks,
        args.prior_family,
        args.prior_strength,
    )
    rows = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="threads")(jobs)
    df = pd.DataFrame(rows).sort_values(["transform", "rank"])
    for key, value in prior_meta.items():
        df[key] = value
    df["prior_family"] = args.prior_family
    df["prior_strength"] = args.prior_strength
    df["rho"] = args.rho
    df["init"] = args.init
    df["seed"] = args.seed
    df.to_csv(output, index=False)

    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
