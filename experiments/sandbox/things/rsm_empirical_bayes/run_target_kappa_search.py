from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp

from src.coherence import (
    _estimate_kappa_hat,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from src.similarity.triplet_rsm import build_bias_aware_triplet_matrix
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-objects", type=int, default=1854)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--train-file", type=str, default="train_90.txt")
    parser.add_argument("--eval-file", type=str, default="test_10.txt")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior-strengths", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--inits", nargs="+", choices=["random", "random_sqrt"], default=["random", "random_sqrt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--luce-regs", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--use-bias-weight-modes", nargs="+", choices=["fisher", "shown"], default=["fisher", "shown"])
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--max-inner", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=72)
    parser.add_argument("--kappa-n-jobs", type=int, default=140)
    parser.add_argument("--kappa-k-max", type=int, default=60)
    parser.add_argument("--kappa-B", type=int, default=30)
    parser.add_argument("--kappa-B-null", type=int, default=20)
    parser.add_argument("--kappa-top-n", type=int, default=6)
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


def fit_bias_prior_from_counts(
    wins: np.ndarray,
    trials: np.ndarray,
    weight_mode: str,
    alpha: float,
    clip_eps: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    baseline = np.divide(
        wins + alpha,
        trials + 2.0 * alpha,
        out=np.full_like(wins, 0.5, dtype=np.float64),
        where=trials > 0,
    )
    np.fill_diagonal(baseline, 1.0)
    observed = (trials > 0) & ~np.eye(trials.shape[0], dtype=bool)
    clipped = np.clip(baseline, clip_eps, 1.0 - clip_eps)
    logits = logit(clipped)
    logits = np.where(observed, logits, 0.0)

    if weight_mode == "fisher":
        weights = trials * clipped * (1.0 - clipped)
    elif weight_mode == "shown":
        weights = trials.copy()
    else:
        raise ValueError(weight_mode)
    weights = np.where(observed, weights, 0.0)

    total_weight = float(weights.sum())
    intercept = float((weights * logits).sum() / total_weight)
    item_bias = np.zeros(trials.shape[0], dtype=np.float64)
    denom = weights.sum(axis=1)

    for _ in range(max_iter):
        updated_bias = np.divide(
            (weights * (logits - intercept - item_bias[None, :])).sum(axis=1),
            denom,
            out=np.zeros_like(item_bias),
            where=denom > 0,
        )
        updated_bias -= updated_bias.mean()
        intercept = float(
            (
                weights
                * (logits - updated_bias[:, None] - updated_bias[None, :])
            ).sum()
            / total_weight
        )
        if np.max(np.abs(updated_bias - item_bias)) < tol:
            item_bias = updated_bias
            break
        item_bias = updated_bias

    prior = expit(intercept + item_bias[:, None] + item_bias[None, :])
    np.fill_diagonal(prior, 1.0)
    return prior


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
    return beta, {"luce_reg": reg, "opt_success": float(result.success), "opt_nit": float(result.nit)}


def fit_luce_prior(
    triplets: np.ndarray,
    wins: np.ndarray,
    trials: np.ndarray,
    n_objects: int,
    reg: float,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    beta, meta = fit_luce_worth(triplets=triplets, n_items=n_objects, reg=reg, maxiter=20)
    observed = (trials > 0) & ~np.eye(n_objects, dtype=bool)
    observed_prob = np.divide(
        wins + alpha,
        trials + 2.0 * alpha,
        out=np.full_like(wins, np.nan),
        where=trials > 0,
    )
    score = beta[:, None] + beta[None, :]
    z = score[observed]
    y = observed_prob[observed]
    w = trials[observed]
    intercept = brentq(lambda c: float(np.sum(w * (expit(c + z) - y))), -20.0, 20.0)
    prior = expit(intercept + score)
    np.fill_diagonal(prior, 1.0)
    meta["intercept"] = float(intercept)
    return prior, meta


def triplet_embedding_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    ei = embedding[triplets[:, 0]]
    ej = embedding[triplets[:, 1]]
    ek = embedding[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_srf_task(
    similarity: np.ndarray,
    triplets: np.ndarray,
    rank: int,
    rho: float,
    init: str,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
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
    embedding = model.fit_transform(similarity)
    return {
        "rho": rho,
        "init": init,
        "seed": seed,
        "val_acc": triplet_embedding_accuracy(embedding, triplets),
        "n_iter": getattr(model, "n_iter_", np.nan),
    }


def compute_kappa_rank(
    similarity: np.ndarray,
    k_max: int,
    B: int,
    B_null: int,
    n_jobs: int,
) -> tuple[int, dict]:
    k_list = list(range(1, k_max + 1))
    p_list = np.linspace(0.05, 0.95, 25)
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        similarity,
        k_list=k_list,
        p_list=p_list,
        B=B,
        random_state=42,
        compute_null=True,
        B_null=B_null,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=n_jobs,
        show_progress=True,
        visualize=False,
    )
    kappa_hat, _ = _estimate_kappa_hat(result["diagnostics"]["x_median"], result["p"], hi_band_quantile=0.85)
    k_star, cp_info = kappa_changepoint(kappa_hat, result["k_list"])
    return int(k_star), {"kappa": np.asarray(kappa_hat).tolist(), "cp_info": cp_info}


def main() -> None:
    args = parse_args()
    tune_output = OUTPUT_DIR / f"{args.output_prefix}_tune.csv"
    matrix_output = OUTPUT_DIR / f"{args.output_prefix}_matrix_summary.csv"
    kappa_output = OUTPUT_DIR / f"{args.output_prefix}_kappa.csv"
    selection_output = OUTPUT_DIR / f"{args.output_prefix}_selection.json"

    train_triplets = load_triplets(args.data_dir / args.train_file)
    eval_triplets = load_triplets(args.data_dir / args.eval_file)
    wins, trials = count_pair_wins_trials(args.n_objects, train_triplets)

    matrix_specs: list[dict[str, object]] = [
        {
            "prior_family": "raw",
            "weight_mode": "",
            "luce_reg": np.nan,
            "prior_strength": 0.0,
        }
    ]
    for weight_mode in args.use_bias_weight_modes:
        for prior_strength in args.prior_strengths:
            matrix_specs.append(
                {
                    "prior_family": "bias",
                    "weight_mode": weight_mode,
                    "luce_reg": np.nan,
                    "prior_strength": float(prior_strength),
                }
            )
    for luce_reg in args.luce_regs:
        for prior_strength in args.prior_strengths:
            matrix_specs.append(
                {
                    "prior_family": "luce",
                    "weight_mode": "",
                    "luce_reg": float(luce_reg),
                    "prior_strength": float(prior_strength),
                }
            )

    bias_priors: dict[str, np.ndarray] = {}
    for weight_mode in args.use_bias_weight_modes:
        key = f"bias::{weight_mode}"
        bias_priors[key] = fit_bias_prior_from_counts(
            wins=wins,
            trials=trials,
            weight_mode=weight_mode,
            alpha=args.alpha,
        )

    luce_priors: dict[float, np.ndarray] = {}
    luce_meta: dict[float, dict[str, float]] = {}
    for luce_reg in args.luce_regs:
        prior, meta = fit_luce_prior(
            triplets=train_triplets,
            wins=wins,
            trials=trials,
            n_objects=args.n_objects,
            reg=luce_reg,
            alpha=args.alpha,
        )
        luce_priors[float(luce_reg)] = prior
        luce_meta[float(luce_reg)] = meta
        log.info("Luce prior fit: reg=%g nit=%d success=%s", luce_reg, int(meta["opt_nit"]), bool(meta["opt_success"]))

    similarity_variants: list[tuple[dict[str, object], np.ndarray]] = []
    for spec in matrix_specs:
        prior_family = str(spec["prior_family"])
        prior_strength = float(spec["prior_strength"])
        if prior_family == "raw":
            prior = np.full((args.n_objects, args.n_objects), 0.5, dtype=np.float64)
        elif prior_family == "bias":
            prior = bias_priors[f"bias::{spec['weight_mode']}"]
        elif prior_family == "luce":
            prior = luce_priors[float(spec["luce_reg"])]
        else:
            raise ValueError(prior_family)
        similarity = build_bias_aware_triplet_matrix(
            wins=wins,
            trials=trials,
            prior=prior,
            alpha=args.alpha,
            prior_strength=prior_strength,
            fill_missing_with_prior=True,
        )
        np.fill_diagonal(similarity, np.nan)
        similarity_variants.append((spec, similarity))

    tune_jobs = []
    for spec, similarity in similarity_variants:
        for rho in args.rhos:
            for init in args.inits:
                for seed in args.seeds:
                    tune_jobs.append(
                        delayed(fit_srf_task)(
                            similarity,
                            eval_triplets,
                            args.target_rank,
                            rho,
                            init,
                            seed,
                            args.max_outer,
                            args.max_inner,
                            args.tol,
                        )
                    )

    log.info("Tune tasks: %d", len(tune_jobs))
    raw_rows = Parallel(n_jobs=args.n_jobs, verbose=10)(tune_jobs)

    rows: list[dict[str, object]] = []
    idx = 0
    for spec, _similarity in similarity_variants:
        for rho in args.rhos:
            for init in args.inits:
                for seed in args.seeds:
                    row = dict(spec)
                    row.update(raw_rows[idx])
                    idx += 1
                    if row["prior_family"] == "luce":
                        row.update(luce_meta[float(row["luce_reg"])])
                    rows.append(row)

    tune_df = pd.DataFrame(rows)
    tune_df.to_csv(tune_output, index=False)

    solver_summary = (
        tune_df.groupby(["prior_family", "weight_mode", "luce_reg", "prior_strength", "rho", "init"], dropna=False, as_index=False)
        .agg(mean_val_acc=("val_acc", "mean"), std_val_acc=("val_acc", "std"))
        .sort_values(["mean_val_acc", "std_val_acc"], ascending=[False, True])
    )

    matrix_summary = (
        solver_summary.sort_values(["mean_val_acc", "std_val_acc"], ascending=[False, True])
        .groupby(["prior_family", "weight_mode", "luce_reg", "prior_strength"], dropna=False, as_index=False)
        .first()
        .sort_values(["mean_val_acc", "std_val_acc"], ascending=[False, True])
    )
    matrix_summary.to_csv(matrix_output, index=False)

    strength_summary = (
        matrix_summary.sort_values(["mean_val_acc", "std_val_acc"], ascending=[False, True])
        .groupby(["prior_family", "weight_mode", "prior_strength"], dropna=False, as_index=False)
        .first()
    )
    selected = strength_summary.head(args.kappa_top_n).copy()
    family_best = (
        strength_summary.sort_values(["mean_val_acc", "std_val_acc"], ascending=[False, True])
        .groupby(["prior_family", "weight_mode"], dropna=False, as_index=False)
        .first()
    )
    selected = (
        pd.concat([selected, family_best], ignore_index=True)
        .drop_duplicates(subset=["prior_family", "weight_mode", "prior_strength"])
        .reset_index(drop=True)
    )

    selection_payload = {
        "data_dir": str(args.data_dir),
        "target_rank": args.target_rank,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "selected_candidates": selected[["prior_family", "weight_mode", "luce_reg", "prior_strength", "rho", "init", "mean_val_acc"]].to_dict(orient="records"),
    }
    selection_output.write_text(json.dumps(selection_payload, indent=2))

    kappa_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        prior_family = row.prior_family
        weight_mode = row.weight_mode
        luce_reg = row.luce_reg
        prior_strength = float(row.prior_strength)

        if prior_family == "raw":
            prior = np.full((args.n_objects, args.n_objects), 0.5, dtype=np.float64)
        elif prior_family == "bias":
            prior = bias_priors[f"bias::{weight_mode}"]
        else:
            prior = luce_priors[float(luce_reg)]

        similarity = build_bias_aware_triplet_matrix(
            wins=wins,
            trials=trials,
            prior=prior,
            alpha=args.alpha,
            prior_strength=prior_strength,
            fill_missing_with_prior=True,
        )
        np.fill_diagonal(similarity, np.nan)

        log.info(
            "Kappa candidate: family=%s weight=%s luce_reg=%s prior_strength=%g",
            prior_family,
            weight_mode,
            "-" if pd.isna(luce_reg) else f"{float(luce_reg):g}",
            prior_strength,
        )
        k_star, meta = compute_kappa_rank(
            similarity=similarity,
            k_max=args.kappa_k_max,
            B=args.kappa_B,
            B_null=args.kappa_B_null,
            n_jobs=args.kappa_n_jobs,
        )
        kappa_rows.append(
            {
                "prior_family": prior_family,
                "weight_mode": weight_mode,
                "luce_reg": luce_reg,
                "prior_strength": prior_strength,
                "mean_val_acc": row.mean_val_acc,
                "best_rho": row.rho,
                "best_init": row.init,
                "k_star_kappa": k_star,
                "kappa_json": json.dumps({"kappa": meta["kappa"]}),
            }
        )
        pd.DataFrame(kappa_rows).to_csv(kappa_output, index=False)
        log.info("  -> kappa=%d", k_star)

    kappa_df = pd.DataFrame(kappa_rows).sort_values(["k_star_kappa", "mean_val_acc"], ascending=[False, False])
    kappa_df.to_csv(kappa_output, index=False)

    log.info("\nMatrix summary top 10:")
    log.info("%s", matrix_summary.head(10).to_string(index=False))
    log.info("\nKappa summary:")
    log.info("%s", kappa_df.to_string(index=False))


if __name__ == "__main__":
    main()
