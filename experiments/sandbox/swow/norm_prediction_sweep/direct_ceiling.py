"""Direct supervised ceiling for SWOW similarity matrices.

This evaluates whether candidate word-word similarities can predict Glasgow
norms directly, before fitting SRF. Outputs are intentionally stable and clean:

    outputs/direct_ceiling/current/results.csv
    outputs/direct_ceiling/current/summary.csv
    outputs/direct_ceiling/current/plots/*.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
for path in [str(SCRIPT_DIR), str(SRC_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from screen_matrices import (  # noqa: E402
    Candidate,
    PROPERTIES,
    aligned_ratings,
    candidate_matrix,
    load_ratings,
    swow_counts,
    weighted_predict,
)


OUT_DIR = SCRIPT_DIR / "outputs" / "direct_ceiling" / "current"
SRF_RESULTS = PROJECT_ROOT / "experiments" / "analyses" / "swow" / "outputs" / "results.csv"


FOCUSED_CANDIDATES = [
    Candidate("R123_profile_both_cosine", True, "profile_both_cosine"),
    Candidate("R123_profile_in_cosine", True, "profile_in_cosine"),
    Candidate("R123_npmi_geometric_mean", True, "npmi", "geometric_mean"),
    Candidate("R123_ppmi_geometric_mean", True, "ppmi", "geometric_mean"),
    Candidate("R123_log_count_geometric_mean", True, "log_count", "geometric_mean"),
    Candidate("R123_log_count_sum", True, "log_count", "sum"),
    Candidate("R1_profile_both_cosine", False, "profile_both_cosine"),
    Candidate("R1_log_count_sum", False, "log_count", "sum"),
]


def as_rated_matrix(cand: Candidate, ratings: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    counts, vocabulary, df = swow_counts(cand.use_all_responses)
    rated_indices, aligned = aligned_ratings(vocabulary, ratings)
    aligned["_matrix_index"] = rated_indices
    matrix = candidate_matrix(cand, counts, vocabulary, df, rated_indices)
    if matrix.shape[0] != len(aligned):
        matrix = matrix[np.ix_(rated_indices, rated_indices)]
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix, aligned


def topk_matrix(matrix: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= matrix.shape[1] - 1:
        return matrix.copy()
    x = matrix.copy()
    keep = np.argpartition(x, -top_k, axis=1)[:, -top_k:]
    mask = np.zeros_like(x, dtype=bool)
    rows = np.arange(x.shape[0])[:, None]
    mask[rows, keep] = True
    x = np.where(mask, x, 0.0)
    x = np.maximum(x, x.T)
    np.fill_diagonal(x, 0.0)
    return x.astype(np.float32, copy=False)


def row_rank_percentile(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    n = x.shape[1] - 1
    order = np.argsort(x, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[np.arange(x.shape[0])[:, None], order] = np.arange(x.shape[1], dtype=np.float32)
    out = ranks / max(n, 1)
    out[x <= 0] = 0.0
    np.fill_diagonal(out, 0.0)
    out = np.maximum(out, out.T)
    return out.astype(np.float32, copy=False)


def matrix_variants(matrix: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    base = np.maximum(matrix, 0.0).astype(np.float32, copy=False)
    variants: dict[str, np.ndarray] = {}
    for name in names:
        if name == "raw":
            x = base.copy()
        elif name == "sqrt":
            x = np.sqrt(base)
        elif name == "square":
            x = base * base
        elif name == "top100":
            x = topk_matrix(base, 100)
        elif name == "top300":
            x = topk_matrix(base, 300)
        elif name == "row_rank":
            x = row_rank_percentile(base)
        elif name == "profile_cosine":
            x = cosine_similarity(base).astype(np.float32)
        else:
            raise ValueError(f"Unknown variant: {name}")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        np.fill_diagonal(x, 0.0)
        variants[name] = x
    return variants


def fit_predict_ridge(
    k: np.ndarray,
    y: np.ndarray,
    outer_cv: KFold,
    inner_cv: KFold,
    alphas: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    pred = np.full(len(y), np.nan)
    chosen = []
    for train, test in outer_cv.split(k):
        scores = []
        for alpha in alphas:
            inner_scores = []
            for inner_train_rel, inner_val_rel in inner_cv.split(train):
                inner_train = train[inner_train_rel]
                inner_val = train[inner_val_rel]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(k[np.ix_(inner_train, inner_train)])
                x_val = scaler.transform(k[np.ix_(inner_val, inner_train)])
                model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-3, max_iter=1000)
                model.fit(x_train, y[inner_train])
                inner_pred = model.predict(x_val)
                inner_scores.append(spearmanr(inner_pred, y[inner_val]).statistic)
            scores.append(np.nanmean(inner_scores))
        best_alpha = float(alphas[int(np.nanargmax(scores))])
        chosen.append(best_alpha)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(k[np.ix_(train, train)])
        x_test = scaler.transform(k[np.ix_(test, train)])
        model = Ridge(alpha=best_alpha, solver="lsqr", tol=1e-3, max_iter=1000)
        model.fit(x_train, y[train])
        pred[test] = model.predict(x_test)
    return pred, chosen


def evaluate_task(
    candidate: str,
    variant: str,
    matrix: np.ndarray,
    aligned: pd.DataFrame,
    dimension: str,
    seed: int,
    n_folds: int,
    alphas: np.ndarray,
) -> list[dict]:
    y_all = aligned[dimension].to_numpy(float)
    valid = np.isfinite(y_all)
    y = y_all[valid]
    k = matrix[np.ix_(valid, valid)]
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=seed + 1)

    rows = []
    pred, chosen = fit_predict_ridge(k, y, outer_cv, inner_cv, alphas)
    stat = spearmanr(pred, y)
    rows.append(
        {
            "candidate": candidate,
            "variant": variant,
            "decoder": "ridge_similarity_features",
            "dimension": dimension,
            "spearman": float(stat.statistic),
            "pvalue": float(stat.pvalue),
            "n": int(len(y)),
            "selected_alpha_median": float(np.median(chosen)),
        }
    )

    for top_k in [25, 100, 300, None]:
        weighted = np.full(len(y), np.nan)
        for train, test in outer_cv.split(k):
            weighted[test] = weighted_predict(k[np.ix_(test, train)], y[train], top_k)
        stat = spearmanr(weighted, y)
        rows.append(
            {
                "candidate": candidate,
                "variant": variant,
                "decoder": f"weighted_top{top_k if top_k is not None else 'all'}",
                "dimension": dimension,
                "spearman": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "n": int(len(y)),
                "selected_alpha_median": np.nan,
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.pivot_table(
        index=["candidate", "variant", "decoder"],
        columns="dimension",
        values="spearman",
        aggfunc="mean",
    ).reset_index()
    summary["mean_spearman"] = summary[PROPERTIES].mean(axis=1)
    summary["max_spearman"] = summary[PROPERTIES].max(axis=1)
    summary = summary.sort_values(["max_spearman", "mean_spearman"], ascending=False)
    return summary


def load_srf_reference() -> pd.DataFrame:
    if not SRF_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(SRF_RESULTS)
    return df[df["method"].eq("Ridge_CV")][["dimension", "correlation"]].rename(
        columns={"correlation": "srf_ridge"}
    )


def make_plots(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    best_dim = (
        results.sort_values("spearman", ascending=False)
        .groupby("dimension", as_index=False)
        .first()
        .sort_values("spearman", ascending=False)
    )
    ref = load_srf_reference()
    if not ref.empty:
        best_dim = best_dim.merge(ref, on="dimension", how="left")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(best_dim))
    width = 0.38 if "srf_ridge" in best_dim else 0.7
    ax.bar(x - width / 2, best_dim["spearman"], width=width, label="Best direct matrix")
    if "srf_ridge" in best_dim:
        ax.bar(x + width / 2, best_dim["srf_ridge"], width=width, label="Existing SRF ridge")
    ax.axhline(0.9, color="black", linewidth=1.0, linestyle=":")
    ax.set_ylabel("Held-out Spearman r")
    ax.set_xticks(x)
    ax.set_xticklabels(best_dim["dimension"], rotation=35, ha="right")
    ax.set_ylim(0, max(0.95, float(best_dim["spearman"].max()) + 0.05))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_dimension.pdf")
    plt.close(fig)

    top = summary.head(20).copy()
    top["label"] = top["candidate"] + "\n" + top["variant"] + " / " + top["decoder"]
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.barh(np.arange(len(top)), top["max_spearman"])
    ax.axvline(0.9, color="black", linewidth=1.0, linestyle=":")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Best dimension Spearman r")
    fig.tight_layout()
    fig.savefig(plot_dir / "top_max_spearman.pdf")
    plt.close(fig)


def write_readme(results: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    best = results.loc[results["spearman"].idxmax()]
    over = int((results["spearman"] >= 0.9).sum())
    payload = {
        "n_results": int(len(results)),
        "n_summary_rows": int(len(summary)),
        "n_jobs": int(args.n_jobs),
        "n_folds": int(args.n_folds),
        "variants": args.variants,
        "max_spearman": float(best["spearman"]),
        "best_candidate": str(best["candidate"]),
        "best_variant": str(best["variant"]),
        "best_decoder": str(best["decoder"]),
        "best_dimension": str(best["dimension"]),
        "n_results_at_or_above_0.9": over,
    }
    (OUT_DIR / "README.md").write_text(
        "# SWOW Direct Similarity Ceiling\n\n"
        "Clean sandbox output for supervised direct prediction from SWOW similarity matrices.\n\n"
        "This tests whether the similarity matrix itself can predict Glasgow norms on held-out words before SRF fitting.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["raw", "sqrt", "square", "top100", "top300", "profile_cosine"],
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ["results.csv", "summary.csv", "README.md"]:
        path = OUT_DIR / stale
        if path.exists():
            path.unlink()
    log_path = OUT_DIR / "run.log"
    log_path.write_text("starting direct similarity ceiling\n", encoding="utf-8")

    ratings = load_ratings()
    tasks = []
    for cand in FOCUSED_CANDIDATES:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"building {cand.name}\n")
        matrix, aligned = as_rated_matrix(cand, ratings)
        variants = matrix_variants(matrix, args.variants)
        tasks.extend(
            (cand.name, variant_name, variant_matrix, aligned, dimension)
            for variant_name, variant_matrix in variants.items()
            for dimension in PROPERTIES
        )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"dispatching {len(tasks)} candidate/variant/dimension tasks\n")

    chunks = Parallel(n_jobs=args.n_jobs, verbose=10, mmap_mode="r")(
        delayed(evaluate_task)(
            candidate,
            variant_name,
            variant_matrix,
            aligned,
            dimension,
            args.seed,
            args.n_folds,
            np.asarray(args.alphas, dtype=float),
        )
        for candidate, variant_name, variant_matrix, aligned, dimension in tasks
    )
    all_rows = [row for chunk in chunks for row in chunk]

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "results.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    make_plots(results, summary)
    write_readme(results, summary, args)
    print((OUT_DIR / "summary.csv").resolve())


if __name__ == "__main__":
    main()
