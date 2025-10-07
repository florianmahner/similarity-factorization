#!/usr/bin/env python3

"""Development experiment for rank detection via synthetic simulations."""

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from pysrf.cross_validation import cross_val_score
from utils.simulation import SimulationParams, generate_simulation_data


@dataclass(frozen=True)
class SimulationConfig:
    true_rank: int
    snr: float
    sparsity: float | None
    membership_type: str
    primary_concentration: float | None
    base_concentration: float | None
    seed: int


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items]


def parse_optional_float_list(value: str) -> list[float | None]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    parsed: list[float | None] = []
    for item in items:
        if item == "none":
            parsed.append(None)
        else:
            parsed.append(float(item))
    return parsed


def parse_membership_types(value: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    return items


def build_configs(args: argparse.Namespace) -> list[SimulationConfig]:
    seeds = [args.seed_offset + idx for idx in range(args.n_seeds)]
    configs: list[SimulationConfig] = []
    for true_rank, snr, sparsity in product(
        args.true_ranks, args.snrs, args.sparsities
    ):
        for membership_type in args.membership_types:
            if membership_type == "dirichlet":
                for primary, base in product(
                    args.primary_concentrations, args.base_concentrations
                ):
                    for seed in seeds:
                        configs.append(
                            SimulationConfig(
                                true_rank=true_rank,
                                snr=snr,
                                sparsity=sparsity,
                                membership_type=membership_type,
                                primary_concentration=primary,
                                base_concentration=base,
                                seed=seed,
                            )
                        )
            else:
                for seed in seeds:
                    configs.append(
                        SimulationConfig(
                            true_rank=true_rank,
                            snr=snr,
                            sparsity=sparsity,
                            membership_type=membership_type,
                            primary_concentration=None,
                            base_concentration=None,
                            seed=seed,
                        )
                    )
    return configs


def evaluate_config(
    config: SimulationConfig, args: argparse.Namespace
) -> dict[str, object]:
    params = SimulationParams(
        n=args.n_observations,
        p=args.n_features,
        k=config.true_rank,
        snr=config.snr,
        rng_state=config.seed,
        sparsity=config.sparsity,
        membership_type=config.membership_type,
        primary_concentration=(
            config.primary_concentration
            if config.primary_concentration is not None
            else args.default_primary_concentration
        ),
        base_concentration=(
            config.base_concentration
            if config.base_concentration is not None
            else args.default_base_concentration
        ),
    )
    record: dict[str, object] = {
        "true_rank": config.true_rank,
        "snr": config.snr,
        "sparsity": np.nan if config.sparsity is None else config.sparsity,
        "membership_type": config.membership_type,
        "primary_concentration": (
            config.primary_concentration
            if config.primary_concentration is not None
            else np.nan
        ),
        "base_concentration": (
            config.base_concentration
            if config.base_concentration is not None
            else np.nan
        ),
        "seed": config.seed,
        "status": "ok",
        "error": "",
    }
    try:
        _, _, _, similarity = generate_simulation_data(params)
        estimator = SRF(
            rank=args.rank_grid[-1],
            random_state=config.seed,
            rho=args.rho,
            init=args.init,
            verbose=False,
        )
        grid = cross_val_score(
            similarity,
            estimator=estimator,
            param_grid={"rank": args.rank_grid},
            n_repeats=args.cv_repeats,
            estimate_sampling_fraction=True,
            sampling_selection=args.sampling_selection,
            random_state=config.seed,
            verbose=args.cv_verbose,
            n_jobs=args.cv_n_jobs,
            fit_final_estimator=False,
        )
        cv_df = grid.cv_results_
        mean_scores = cv_df.groupby("rank")["score"].mean().sort_values()
        std_scores = cv_df.groupby("rank")["score"].std()
        rank_order = [int(rank) for rank in mean_scores.index]
        best_rank = rank_order[0]
        best_score = float(mean_scores.iloc[0])
        best_std = (
            float(std_scores.loc[best_rank]) if best_rank in std_scores else np.nan
        )
        second_score = float(mean_scores.iloc[1]) if len(mean_scores) > 1 else np.nan
        score_gap = second_score - best_score if not np.isnan(second_score) else np.nan
        true_score = (
            float(mean_scores.loc[config.true_rank])
            if config.true_rank in mean_scores.index
            else np.nan
        )
        record.update(
            {
                "selected_rank": best_rank,
                "best_score": best_score,
                "best_score_std": best_std,
                "second_best_score": second_score,
                "score_gap": score_gap,
                "true_rank_score": true_score,
                "rank_order": "|".join(str(rank) for rank in rank_order),
                "abs_error": abs(best_rank - config.true_rank),
                "signed_error": best_rank - config.true_rank,
                "is_correct": best_rank == config.true_rank,
                "candidate_count": len(rank_order),
            }
        )
    except Exception as exc:
        record.update(
            {
                "selected_rank": np.nan,
                "best_score": np.nan,
                "best_score_std": np.nan,
                "second_best_score": np.nan,
                "score_gap": np.nan,
                "true_rank_score": np.nan,
                "rank_order": "",
                "abs_error": np.nan,
                "signed_error": np.nan,
                "is_correct": False,
                "candidate_count": 0,
                "status": "error",
                "error": str(exc),
            }
        )
    return record


def _evaluate_with_progress(
    payload: tuple[int, SimulationConfig, argparse.Namespace, int],
) -> dict[str, object]:
    idx, config, args, total = payload
    result = evaluate_config(config, args)
    print(f"Completed {idx}/{total}")
    return result


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    group_cols = [
        "true_rank",
        "snr",
        "sparsity",
        "membership_type",
        "primary_concentration",
        "base_concentration",
    ]
    summary = (
        ok.groupby(group_cols)
        .agg(
            trials=("seed", "count"),
            accuracy=("is_correct", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_signed_error=("signed_error", "mean"),
            selected_rank_std=("selected_rank", "std"),
            score_gap_mean=("score_gap", "mean"),
            score_gap_std=("score_gap", "std"),
            best_score_mean=("best_score", "mean"),
        )
        .reset_index()
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic rank detection sweeps using SRF cross-validation"
    )
    parser.add_argument("--n-observations", type=int, default=150)
    parser.add_argument("--n-features", type=int, default=150)
    parser.add_argument("--true-ranks", type=str, default="3,6")
    parser.add_argument("--snrs", type=str, default="0.2,0.5")
    parser.add_argument("--sparsities", type=str, default="none,0.6")
    parser.add_argument("--membership-types", type=str, default="dirichlet,hard")
    parser.add_argument("--primary-concentrations", type=str, default="4.0,7.0")
    parser.add_argument("--base-concentrations", type=str, default="0.05,0.2")
    parser.add_argument("--rank-grid", type=str, default="2,4,6,8,10")
    parser.add_argument("--cv-repeats", type=int, default=4)
    parser.add_argument("--sampling-selection", type=str, default="mean")
    parser.add_argument("--cv-n-jobs", type=int, default=-1)
    parser.add_argument("--cv-verbose", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--init", type=str, default="random_sqrt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preview"))
    parser.add_argument("--default-primary-concentration", type=float, default=5.0)
    parser.add_argument("--default-base-concentration", type=float, default=0.1)
    args = parser.parse_args()
    args.true_ranks = parse_int_list(args.true_ranks)
    args.snrs = parse_float_list(args.snrs)
    args.sparsities = parse_optional_float_list(args.sparsities)
    args.membership_types = parse_membership_types(args.membership_types)
    args.primary_concentrations = parse_float_list(args.primary_concentrations)
    args.base_concentrations = parse_float_list(args.base_concentrations)
    args.rank_grid = sorted(set(parse_int_list(args.rank_grid)))
    if not args.rank_grid:
        raise ValueError("rank grid cannot be empty")
    args.sampling_selection = args.sampling_selection.lower()
    valid_sampling = {"mean", "min", "max"}
    if args.sampling_selection not in valid_sampling:
        raise ValueError("sampling-selection must be one of mean, min, max")
    if args.n_jobs != -1:
        print("Overriding n-jobs to -1 for full parallelism")
        args.n_jobs = -1
    if args.cv_n_jobs != -1:
        print("Overriding cv-n-jobs to -1 for full parallelism")
        args.cv_n_jobs = -1
    return args


def main() -> None:
    args = parse_args()
    configs = build_configs(args)
    total_jobs = len(configs)
    print(f"Launching {total_jobs} configurations")
    indexed_payloads = [
        (idx, config, args, total_jobs) for idx, config in enumerate(configs, start=1)
    ]
    if args.n_jobs == 1:
        results = []
        for payload in indexed_payloads:
            idx = payload[0]
            print(f"Running config {idx}/{total_jobs}")
            results.append(_evaluate_with_progress(payload))
    else:
        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(_evaluate_with_progress)(payload) for payload in indexed_payloads
        )
    df = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "rank_detection_results.csv"
    df.to_csv(results_path, index=False)
    summary = aggregate_results(df)
    if not summary.empty:
        summary_path = args.output_dir / "rank_detection_summary.csv"
        summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
