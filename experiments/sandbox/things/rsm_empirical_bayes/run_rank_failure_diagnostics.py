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

from src.similarity.triplet_rsm import (
    build_bias_aware_triplet_matrix,
    fit_triplet_bias_prior,
)
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRIAL_BINS = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, np.inf])
TRIAL_LABELS = ["1", "2", "3", "4-5", "6-8", "9+"]


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
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--init", choices=["random", "random_sqrt"], default="random_sqrt")
    parser.add_argument("--ranks", nargs="+", type=int, default=[24, 40, 49])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-outer", type=int, default=30)
    parser.add_argument("--max-inner", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=3)
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


def triplet_embedding_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> tuple[float, np.ndarray]:
    ei = embedding[triplets[:, 0]]
    ej = embedding[triplets[:, 1]]
    ek = embedding[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    correct = (sij > sik) & (sij > sjk)
    return float(np.mean(correct)), correct


def fit_rank(
    similarity: np.ndarray,
    rank: int,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> tuple[int, np.ndarray, int]:
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
    return rank, embedding, int(getattr(model, "n_iter_", -1))


def summarize_fit(
    rank: int,
    embedding: np.ndarray,
    n_iter: int,
    similarity: np.ndarray,
    prior: np.ndarray,
    trials: np.ndarray,
    train_triplets: np.ndarray,
    eval_triplets: np.ndarray,
) -> dict[str, float | int]:
    mask = (trials > 0) & ~np.eye(trials.shape[0], dtype=bool)
    predicted = embedding @ embedding.T

    target_obs = similarity[mask]
    pred_obs = predicted[mask]
    prior_obs = prior[mask]
    weights = trials[mask]

    train_acc, _ = triplet_embedding_accuracy(embedding, train_triplets)
    val_acc, _ = triplet_embedding_accuracy(embedding, eval_triplets)

    residual_target = target_obs - prior_obs
    residual_pred = pred_obs - prior_obs

    return {
        "rank": rank,
        "n_iter": n_iter,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "mse_obs": float(np.mean((pred_obs - target_obs) ** 2)),
        "weighted_mse_obs": float(np.average((pred_obs - target_obs) ** 2, weights=weights)),
        "corr_obs": float(np.corrcoef(pred_obs, target_obs)[0, 1]),
        "residual_corr_obs": float(np.corrcoef(residual_pred, residual_target)[0, 1]),
        "mean_norm": float(np.linalg.norm(embedding, axis=1).mean()),
        "std_norm": float(np.linalg.norm(embedding, axis=1).std()),
        "max_norm": float(np.linalg.norm(embedding, axis=1).max()),
        "mean_pred": float(np.mean(pred_obs)),
        "std_pred": float(np.std(pred_obs)),
        "mean_abs_residual_pred": float(np.mean(np.abs(residual_pred))),
    }


def pair_bin_diagnostics(
    base_rank: int,
    new_rank: int,
    base_pred: np.ndarray,
    new_pred: np.ndarray,
    similarity: np.ndarray,
    trials: np.ndarray,
) -> pd.DataFrame:
    mask = (trials > 0) & ~np.eye(trials.shape[0], dtype=bool)
    trial_obs = trials[mask]
    target_obs = similarity[mask]
    base_err = (base_pred[mask] - target_obs) ** 2
    new_err = (new_pred[mask] - target_obs) ** 2
    trial_bin = pd.cut(trial_obs, bins=TRIAL_BINS, labels=TRIAL_LABELS)

    frame = pd.DataFrame(
        {
            "trial_bin": trial_bin,
            "trials": trial_obs,
            "delta_mse": base_err - new_err,
            "base_err": base_err,
            "new_err": new_err,
        }
    )
    summary = (
        frame.groupby("trial_bin", observed=False, as_index=False)
        .agg(
            n_pairs=("delta_mse", "size"),
            mean_trials=("trials", "mean"),
            mean_delta_mse=("delta_mse", "mean"),
            weighted_delta_mse=("delta_mse", lambda x: float(np.average(x, weights=frame.loc[x.index, "trials"]))),
            base_mse=("base_err", "mean"),
            new_mse=("new_err", "mean"),
        )
    )
    summary.insert(0, "comparison", f"{base_rank}_to_{new_rank}")
    return summary


def triplet_flip_summary(
    base_rank: int,
    new_rank: int,
    base_correct: np.ndarray,
    new_correct: np.ndarray,
    similarity: np.ndarray,
    prior: np.ndarray,
    trials: np.ndarray,
    triplets: np.ndarray,
) -> pd.DataFrame:
    i = triplets[:, 0]
    j = triplets[:, 1]
    k = triplets[:, 2]

    target_margin = similarity[i, j] - np.maximum(similarity[i, k], similarity[j, k])
    prior_margin = prior[i, j] - np.maximum(prior[i, k], prior[j, k])
    chosen_trials = trials[i, j]
    alt_max_trials = np.maximum(trials[i, k], trials[j, k])
    min_pair_trials = np.minimum.reduce([trials[i, j], trials[i, k], trials[j, k]])
    mean_pair_trials = (trials[i, j] + trials[i, k] + trials[j, k]) / 3.0

    groups = {
        "base_only": base_correct & ~new_correct,
        "new_only": new_correct & ~base_correct,
        "both_correct": base_correct & new_correct,
        "both_wrong": ~base_correct & ~new_correct,
    }

    rows: list[dict[str, float | int | str]] = []
    for group_name, mask in groups.items():
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "comparison": f"{base_rank}_to_{new_rank}",
                "group": group_name,
                "n_triplets": count,
                "fraction": float(count / len(triplets)),
                "mean_target_margin": float(target_margin[mask].mean()),
                "median_target_margin": float(np.median(target_margin[mask])),
                "mean_prior_margin": float(prior_margin[mask].mean()),
                "prior_support_fraction": float(np.mean(prior_margin[mask] > 0.0)),
                "mean_chosen_trials": float(chosen_trials[mask].mean()),
                "mean_alt_max_trials": float(alt_max_trials[mask].mean()),
                "mean_min_pair_trials": float(min_pair_trials[mask].mean()),
                "mean_pair_trials": float(mean_pair_trials[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    metrics_output = OUTPUT_DIR / f"{args.output_prefix}_metrics.csv"
    pair_output = OUTPUT_DIR / f"{args.output_prefix}_pair_bins.csv"
    flip_output = OUTPUT_DIR / f"{args.output_prefix}_triplet_flips.csv"

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

    log.info(
        "Running ranks=%s on %s with prior=%s strength=%g rho=%g init=%s",
        args.ranks,
        args.data_dir.name,
        args.prior_family,
        args.prior_strength,
        args.rho,
        args.init,
    )
    rank_results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_rank)(
            similarity=similarity,
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

    embeddings: dict[int, np.ndarray] = {}
    predicted: dict[int, np.ndarray] = {}
    correct_eval: dict[int, np.ndarray] = {}
    metrics_rows: list[dict[str, float | int | str]] = []

    for rank, embedding, n_iter in sorted(rank_results, key=lambda x: x[0]):
        embeddings[rank] = embedding
        predicted[rank] = embedding @ embedding.T
        _, eval_correct = triplet_embedding_accuracy(embedding, eval_triplets)
        correct_eval[rank] = eval_correct
        row = summarize_fit(
            rank=rank,
            embedding=embedding,
            n_iter=n_iter,
            similarity=similarity,
            prior=prior,
            trials=trials,
            train_triplets=train_triplets,
            eval_triplets=eval_triplets,
        )
        row["prior_family"] = args.prior_family
        row["prior_strength"] = args.prior_strength
        row["rho"] = args.rho
        row["init"] = args.init
        row["seed"] = args.seed
        row.update(prior_meta)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rank")
    metrics_df.to_csv(metrics_output, index=False)

    pair_frames: list[pd.DataFrame] = []
    flip_frames: list[pd.DataFrame] = []
    base_rank = min(args.ranks)
    for rank in sorted(args.ranks):
        if rank == base_rank:
            continue
        pair_frames.append(
            pair_bin_diagnostics(
                base_rank=base_rank,
                new_rank=rank,
                base_pred=predicted[base_rank],
                new_pred=predicted[rank],
                similarity=similarity,
                trials=trials,
            )
        )
        flip_frames.append(
            triplet_flip_summary(
                base_rank=base_rank,
                new_rank=rank,
                base_correct=correct_eval[base_rank],
                new_correct=correct_eval[rank],
                similarity=similarity,
                prior=prior,
                trials=trials,
                triplets=eval_triplets,
            )
        )

    pair_df = pd.concat(pair_frames, ignore_index=True)
    pair_df.to_csv(pair_output, index=False)

    flip_df = pd.concat(flip_frames, ignore_index=True)
    flip_df.to_csv(flip_output, index=False)

    log.info("\nMetrics:")
    log.info("%s", metrics_df.to_string(index=False))
    log.info("\nPair bins:")
    log.info("%s", pair_df.to_string(index=False))
    log.info("\nTriplet flips:")
    log.info("%s", flip_df.to_string(index=False))


if __name__ == "__main__":
    main()
