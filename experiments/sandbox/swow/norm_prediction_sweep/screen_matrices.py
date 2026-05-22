"""Screen SWOW similarity matrices before fitting SRF.

The screen evaluates whether a candidate word-word matrix can predict held-out
Glasgow norms directly from similarities to training words. This is a cheap
proxy for matrix quality and keeps SRF refits focused on promising sparse
similarities.

Outputs are written to stable paths:
    outputs/matrix_screen/current/results.csv
    outputs/matrix_screen/current/summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.swow import compute_ppmi, load_swow_data  # noqa: E402
from utils.graphs import build_directed_graph, clean_graph, graph_to_matrix, symmetrize_matrix  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
SWOW_DIR = DATA_DIR / "small-world-of-words"
RATINGS_PATH = DATA_DIR / "semantic_norms" / "glasgow_norms.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "matrix_screen" / "current"

PROPERTIES = [
    "arousal",
    "valence",
    "dominance",
    "concreteness",
    "imageability",
    "familiarity",
    "aoa",
    "size",
    "gender",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    use_all_responses: bool
    kind: str
    symmetrization: str = "sum"
    bidirectional_only: bool = False


def load_ratings() -> pd.DataFrame:
    df = pd.read_csv(RATINGS_PATH)
    df["word"] = df["word"].str.lower()
    df = df.rename(columns={"semsize": "size"})
    return df[["word", *PROPERTIES]]


def swow_counts(use_all_responses: bool) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    df = load_swow_data(SWOW_DIR, use_all_responses=use_all_responses)
    g = build_directed_graph(
        df["cue"].tolist(),
        df["response"].tolist(),
        df["count"].tolist(),
    )
    g = clean_graph(g)
    vocabulary = sorted(g.nodes())
    counts = graph_to_matrix(g, vocabulary).astype(np.float32, copy=False)
    np.fill_diagonal(counts, 0.0)
    return counts, vocabulary, df


def sparse_counts(use_all_responses: bool, vocabulary: list[str], df: pd.DataFrame) -> sparse.csr_matrix:
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    valid = df[df["cue"].isin(word_to_idx) & df["response"].isin(word_to_idx)]
    rows = valid["cue"].map(word_to_idx).to_numpy()
    cols = valid["response"].map(word_to_idx).to_numpy()
    data = valid["count"].to_numpy(dtype=np.float32)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(vocabulary), len(vocabulary)))
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    return mat


def sym_counts(counts: np.ndarray, method: str, bidirectional_only: bool) -> np.ndarray:
    directed = counts.copy()
    directed[directed == 0] = np.nan
    np.fill_diagonal(directed, np.nan)
    return symmetrize_matrix(directed, method=method, bidirectional_only=bidirectional_only)


def row_probability(counts: np.ndarray) -> np.ndarray:
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    prob = counts / row_sum
    prob[prob == 0] = np.nan
    np.fill_diagonal(prob, np.nan)
    return prob.astype(np.float32, copy=False)


def npmi_from_counts(counts_sym: np.ndarray) -> np.ndarray:
    counts_clean = np.nan_to_num(counts_sym, nan=0.0)
    total = counts_clean.sum()
    if total <= 0:
        return np.zeros_like(counts_clean, dtype=np.float32)
    p_joint = counts_clean / total
    p_marginal = counts_clean.sum(axis=1) / total
    p_indep = np.outer(p_marginal, p_marginal)
    out = np.zeros_like(counts_clean, dtype=np.float32)
    mask = (p_joint > 0) & (p_indep > 0)
    pmi = np.log(p_joint[mask] / p_indep[mask])
    out[mask] = np.maximum(pmi / (-np.log(p_joint[mask])), 0.0)
    out[np.isnan(counts_sym)] = np.nan
    np.fill_diagonal(out, np.nan)
    return out


def lmi_from_counts(counts_sym: np.ndarray) -> np.ndarray:
    ppmi = compute_ppmi(counts_sym, negative_as_nan=False).astype(np.float32)
    return np.nan_to_num(counts_sym, nan=0.0).astype(np.float32) * np.nan_to_num(ppmi, nan=0.0)


def candidate_matrix(
    cand: Candidate,
    counts: np.ndarray,
    vocabulary: list[str],
    df: pd.DataFrame,
    rated_indices: np.ndarray,
) -> np.ndarray:
    if cand.kind == "count":
        return sym_counts(counts, cand.symmetrization, cand.bidirectional_only)
    if cand.kind == "log_count":
        s = sym_counts(counts, cand.symmetrization, cand.bidirectional_only)
        return np.where(np.isnan(s), np.nan, np.log1p(s)).astype(np.float32)
    if cand.kind == "conditional":
        p = row_probability(counts)
        return symmetrize_matrix(p, method=cand.symmetrization, bidirectional_only=cand.bidirectional_only)
    if cand.kind == "ppmi":
        return compute_ppmi(sym_counts(counts, cand.symmetrization, cand.bidirectional_only), negative_as_nan=False)
    if cand.kind == "npmi":
        return npmi_from_counts(sym_counts(counts, cand.symmetrization, cand.bidirectional_only))
    if cand.kind == "lmi":
        return lmi_from_counts(sym_counts(counts, cand.symmetrization, cand.bidirectional_only))
    if cand.kind in {"profile_out_cosine", "profile_in_cosine", "profile_both_cosine"}:
        csr = sparse_counts(cand.use_all_responses, vocabulary, df)
        if cand.kind == "profile_out_cosine":
            x = normalize(csr, norm="l2", axis=1)
        elif cand.kind == "profile_in_cosine":
            x = normalize(csr.T, norm="l2", axis=1)
        else:
            out = normalize(csr, norm="l2", axis=1)
            inc = normalize(csr.T, norm="l2", axis=1)
            x = sparse.hstack([out, inc], format="csr")
        xr = x[rated_indices]
        return cosine_similarity(xr).astype(np.float32)
    raise ValueError(f"Unknown candidate kind: {cand.kind}")


def aligned_ratings(vocabulary: list[str], ratings: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}
    keep = ratings["word"].isin(word_to_idx)
    aligned = ratings[keep].copy().reset_index(drop=True)
    indices = aligned["word"].map(word_to_idx).to_numpy(dtype=int)
    return indices, aligned


def weighted_predict(k_test_train: np.ndarray, y_train: np.ndarray, top_k: int | None) -> np.ndarray:
    w = np.nan_to_num(k_test_train, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    if top_k is not None and top_k < w.shape[1]:
        keep = np.argpartition(w, -top_k, axis=1)[:, -top_k:]
        mask = np.zeros_like(w, dtype=bool)
        rows = np.arange(w.shape[0])[:, None]
        mask[rows, keep] = True
        w = np.where(mask, w, 0.0)
    denom = w.sum(axis=1)
    pred = np.full(w.shape[0], float(np.mean(y_train)), dtype=float)
    ok = denom > 0
    pred[ok] = (w[ok] @ y_train) / denom[ok]
    return pred


def evaluate_matrix(
    matrix: np.ndarray,
    aligned: pd.DataFrame,
    candidate_name: str,
    seed: int,
    n_folds: int,
    top_ks: list[int | None],
    use_ridge: bool,
) -> list[dict]:
    rows = []
    if matrix.shape[0] != len(aligned):
        indices = aligned["_matrix_index"].to_numpy(dtype=int)
        matrix = matrix[np.ix_(indices, indices)]
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 0.0)

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    alphas = np.logspace(-3, 3, 13)

    for prop in PROPERTIES:
        y = aligned[prop].to_numpy(float)
        valid = np.isfinite(y)
        if valid.sum() < 50:
            continue
        yy = y[valid]
        kk = matrix[np.ix_(valid, valid)]
        for top_k in top_ks:
            pred = np.full(len(yy), np.nan)
            for train, test in cv.split(kk):
                pred[test] = weighted_predict(kk[np.ix_(test, train)], yy[train], top_k)
            stat = spearmanr(pred, yy)
            rows.append({
                "candidate": candidate_name,
                "decoder": f"weighted_top{top_k if top_k is not None else 'all'}",
                "dimension": prop,
                "spearman": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "n": int(len(yy)),
            })

        if use_ridge:
            pred = np.full(len(yy), np.nan)
            for train, test in cv.split(kk):
                model = RidgeCV(alphas=alphas)
                model.fit(kk[np.ix_(train, train)], yy[train])
                pred[test] = model.predict(kk[np.ix_(test, train)])
            stat = spearmanr(pred, yy)
            rows.append({
                "candidate": candidate_name,
                "decoder": "ridge_similarity_features",
                "dimension": prop,
                "spearman": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "n": int(len(yy)),
            })
    return rows


def candidate_grid() -> list[Candidate]:
    cands = []
    for use_all, label in [(True, "R123"), (False, "R1")]:
        for sym in ["sum", "max", "arithmetic_mean", "geometric_mean"]:
            for kind in ["count", "log_count", "conditional", "ppmi", "npmi", "lmi"]:
                cands.append(Candidate(f"{label}_{kind}_{sym}", use_all, kind, sym))
        for kind in ["profile_out_cosine", "profile_in_cosine", "profile_both_cosine"]:
            cands.append(Candidate(f"{label}_{kind}", use_all, kind))
    return cands


def write_summary(df: pd.DataFrame) -> None:
    summary = df.pivot_table(
        index=["candidate", "decoder"],
        columns="dimension",
        values="spearman",
        aggfunc="mean",
    ).reset_index()
    summary["mean_spearman"] = summary[PROPERTIES].mean(axis=1)
    summary["max_spearman"] = summary[PROPERTIES].max(axis=1)
    summary = summary.sort_values("mean_spearman", ascending=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, action="append", default=[10, 25, 50, 100, 200])
    parser.add_argument("--ridge", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings()
    all_rows = []

    for use_all in [True, False]:
        counts, vocabulary, df = swow_counts(use_all)
        rated_indices, aligned = aligned_ratings(vocabulary, ratings)
        aligned["_matrix_index"] = rated_indices
        cands = [cand for cand in candidate_grid() if cand.use_all_responses == use_all]

        def run_one(cand: Candidate) -> list[dict]:
            mat = candidate_matrix(cand, counts, vocabulary, df, rated_indices)
            return evaluate_matrix(
                mat,
                aligned,
                cand.name,
                seed=args.seed,
                n_folds=args.n_folds,
                top_ks=[*args.top_k, None],
                use_ridge=args.ridge,
            )

        chunks = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_one)(cand) for cand in cands
        )
        rows = [row for chunk in chunks for row in chunk]
        all_rows.extend(rows)
        partial = pd.DataFrame(all_rows)
        partial.to_csv(OUT_DIR / "results.csv", index=False)
        write_summary(partial)

    final = pd.DataFrame(all_rows)
    final.to_csv(OUT_DIR / "results.csv", index=False)
    write_summary(final)
    print((OUT_DIR / "summary.csv").resolve())


if __name__ == "__main__":
    main()
