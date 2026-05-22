"""Focused SRF sweep on sparse SWOW similarity matrices.

Outputs are stable and intentionally shallow:

    outputs/srf_sparse/current/
        all_results.csv
        all_summary.csv
        fit_tasks.csv
        run.log
        fits/<spec>/
        plots/*.pdf

The matrices keep unobserved SWOW pairs as NaN, so SRF treats them as missing
instead of as zero-valued similarities.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
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
from scipy import sparse
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
for path in [str(SCRIPT_DIR), str(SRC_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pysrf import SRF  # noqa: E402
from pysrf.consensus import AlignedConsensus, align_embeddings  # noqa: E402
from screen_matrices import (  # noqa: E402
    Candidate,
    PROPERTIES,
    candidate_matrix,
    load_ratings,
    swow_counts,
)
from utils.graphs import build_directed_graph, clean_graph  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "outputs" / "srf_sparse" / "current"
SRF_RESULTS = PROJECT_ROOT / "experiments" / "analyses" / "swow" / "outputs" / "results.csv"


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    candidate: Candidate
    value_transform: str = "raw"


@dataclass(frozen=True)
class FitSpec:
    name: str
    matrix_name: str
    matrix_path: str
    vocab_path: str
    rank: int
    max_outer: int
    max_inner: int
    rho: float
    tol: float
    run_idx: int
    seed: int


def matrix_specs(preset: str) -> list[MatrixSpec]:
    base = [
        MatrixSpec(
            "R123_log_count_sum",
            Candidate("R123_log_count_sum", True, "log_count", "sum"),
        ),
        MatrixSpec(
            "R123_log_count_geometric_mean",
            Candidate("R123_log_count_geometric_mean", True, "log_count", "geometric_mean"),
        ),
        MatrixSpec(
            "R123_npmi_geometric_mean",
            Candidate("R123_npmi_geometric_mean", True, "npmi", "geometric_mean"),
        ),
        MatrixSpec(
            "R123_ppmi_geometric_mean",
            Candidate("R123_ppmi_geometric_mean", True, "ppmi", "geometric_mean"),
        ),
        MatrixSpec(
            "R123_ppmi_sum",
            Candidate("R123_ppmi_sum", True, "ppmi", "sum"),
        ),
        MatrixSpec(
            "R1_log_count_sum",
            Candidate("R1_log_count_sum", False, "log_count", "sum"),
        ),
    ]
    if preset == "quick":
        keep = {
            "R123_log_count_sum",
            "R123_npmi_geometric_mean",
            "R123_ppmi_geometric_mean",
            "R123_ppmi_sum",
        }
        base = [spec for spec in base if spec.name in keep]
    specs: list[MatrixSpec] = []
    for spec in base:
        specs.append(spec)
        if spec.name in {"R123_log_count_sum", "R123_ppmi_sum", "R123_npmi_geometric_mean"}:
            specs.append(MatrixSpec(f"{spec.name}_max01", spec.candidate, "max01"))
        if spec.name == "R123_log_count_sum":
            specs.append(MatrixSpec(f"{spec.name}_sqrt", spec.candidate, "sqrt"))
    return specs


def select_vocabulary(g, top_n_words: int | None) -> list[str]:
    if top_n_words is None or top_n_words <= 0 or top_n_words >= g.number_of_nodes():
        return sorted(g.nodes())

    strengths = g.degree(weight="weight")
    sorted_nodes = sorted(strengths, key=lambda item: (-item[1], item[0]))
    selected = [node for node, _ in sorted_nodes[:top_n_words]]
    return sorted(selected)


def prepare_matrix(spec: MatrixSpec, matrix_dir: Path, top_n_words: int | None) -> dict:
    counts, vocabulary, df = swow_counts(spec.candidate.use_all_responses)
    if top_n_words is not None and top_n_words > 0 and top_n_words < len(vocabulary):
        g = build_directed_graph(df["cue"].tolist(), df["response"].tolist(), df["count"].tolist())
        g = clean_graph(g)
        vocabulary = select_vocabulary(g, top_n_words)
        word_to_idx = {word: i for i, word in enumerate(vocabulary)}
        valid = df[df["cue"].isin(word_to_idx) & df["response"].isin(word_to_idx)]
        rows = valid["cue"].map(word_to_idx).to_numpy()
        cols = valid["response"].map(word_to_idx).to_numpy()
        data = valid["count"].to_numpy(dtype=np.float32)
        counts = sparse.csr_matrix((data, (rows, cols)), shape=(len(vocabulary), len(vocabulary))).toarray()
        counts = counts.astype(np.float32, copy=False)
        np.fill_diagonal(counts, 0.0)

    full_indices = np.arange(len(vocabulary), dtype=int)
    mat = candidate_matrix(spec.candidate, counts, vocabulary, df, full_indices)
    mat = np.asarray(mat, dtype=np.float32)
    finite = np.isfinite(mat)
    np.fill_diagonal(mat, np.nan)

    if spec.value_transform == "max01":
        max_val = float(np.nanmax(mat))
        if max_val > 0:
            mat = mat / max_val
    elif spec.value_transform == "sqrt":
        mat = np.where(np.isfinite(mat), np.sqrt(np.maximum(mat, 0.0)), np.nan).astype(np.float32)
    elif spec.value_transform != "raw":
        raise ValueError(f"Unknown value transform: {spec.value_transform}")

    matrix_path = matrix_dir / f"{spec.name}.npy"
    vocab_path = matrix_dir / f"{spec.name}_vocabulary.json"
    np.save(matrix_path, mat)
    vocab_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    observed = np.isfinite(mat)
    np.fill_diagonal(observed, False)
    return {
        "matrix_name": spec.name,
        "matrix_path": str(matrix_path),
        "vocab_path": str(vocab_path),
        "n_words": int(mat.shape[0]),
        "observed_fraction": float(observed.mean()),
        "missing_fraction": float((~observed).mean()),
        "finite_min": float(np.nanmin(mat)),
        "finite_max": float(np.nanmax(mat)),
        "finite_mean": float(np.nanmean(mat)),
        "source_candidate": spec.candidate.name,
        "value_transform": spec.value_transform,
        "source_observed_fraction": float(finite.mean()),
    }


def build_fit_specs(matrix_rows: list[dict], n_runs: int, seed: int, preset: str) -> list[FitSpec]:
    if preset == "quick":
        train_configs = [
            {"label": "k64_o15_i20", "rank": 64, "max_outer": 15, "max_inner": 20},
            {"label": "k128_o25_i20", "rank": 128, "max_outer": 25, "max_inner": 20},
        ]
    else:
        train_configs = [
            {"label": "k128_o50_i30", "rank": 128, "max_outer": 50, "max_inner": 30},
            {"label": "k128_o150_i60", "rank": 128, "max_outer": 150, "max_inner": 60},
            {"label": "k268_o50_i30", "rank": 268, "max_outer": 50, "max_inner": 30},
            {"label": "k268_o150_i60", "rank": 268, "max_outer": 150, "max_inner": 60},
        ]
    specs: list[FitSpec] = []
    for row in matrix_rows:
        for cfg in train_configs:
            base_name = f"{row['matrix_name']}_{cfg['label']}"
            for run_idx in range(n_runs):
                specs.append(
                    FitSpec(
                        name=base_name,
                        matrix_name=row["matrix_name"],
                        matrix_path=row["matrix_path"],
                        vocab_path=row["vocab_path"],
                        rank=cfg["rank"],
                        max_outer=cfg["max_outer"],
                        max_inner=cfg["max_inner"],
                        rho=3.0,
                        tol=0.0,
                        run_idx=run_idx,
                        seed=seed + run_idx,
                    )
                )
    return specs


def fit_one(spec: FitSpec, out_dir: Path) -> dict:
    start = time.time()
    spec_dir = out_dir / "fits" / spec.name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "spec.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")

    x = np.load(spec.matrix_path)
    model = SRF(
        rank=spec.rank,
        rho=spec.rho,
        max_outer=spec.max_outer,
        max_inner=spec.max_inner,
        tol=spec.tol,
        random_state=spec.seed,
        missing_values=np.nan,
        bounds=(0, None),
        check_input=False,
    )
    embedding = model.fit_transform(x)
    run_path = spec_dir / f"run_{spec.run_idx:03d}.npy"
    np.save(run_path, embedding.astype(np.float32))
    history = {
        key: [float(v) if np.isscalar(v) else v for v in values]
        for key, values in model.history_.items()
    }
    (spec_dir / f"history_{spec.run_idx:03d}.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"done": spec.name, "run_idx": spec.run_idx}) + "\n")
    return {
        **asdict(spec),
        "run_path": str(run_path),
        "elapsed_sec": time.time() - start,
        "n_iter": int(model.n_iter_),
        "final_evar": float(model.history_.get("evar", [np.nan])[-1]),
        "final_rec_error": float(model.history_.get("rec_error", [np.nan])[-1]),
    }


def consensus_from_runs(runs: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    if runs.shape[0] == 1:
        return {"selected": runs[0], "mean": runs[0], "median": runs[0]}
    aligned = align_embeddings(runs, reference_idx=0)
    stacked = aligned.transpose(1, 0, 2).reshape(aligned.shape[1], -1)
    consensus = AlignedConsensus(rank=rank).fit(stacked)
    return {
        "selected": consensus.transform(stacked),
        "mean": aligned.mean(axis=0),
        "median": np.median(aligned, axis=0),
    }


def embedding_transforms() -> list[tuple[str, object]]:
    return [
        ("raw", lambda x: x),
        ("sqrt", lambda x: np.sqrt(np.maximum(x, 0.0))),
        ("log1p", lambda x: np.log1p(x)),
        ("row_l2", lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)),
        ("binary", lambda x: (x > 0).astype(float)),
    ]


def aligned_xy(embedding: np.ndarray, vocabulary: list[str], ratings: pd.DataFrame, dimension: str) -> tuple[np.ndarray, np.ndarray]:
    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}
    valid = ratings[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]
    indices = np.array([word_to_idx[word] for word in valid["word"]], dtype=int)
    return embedding[indices], valid[dimension].to_numpy(float)


def ridge_oof(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    alphas = np.logspace(-5, 4, 25)
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in outer.split(x):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=inner))
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    return pred


def evaluate_embedding(
    embedding: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    fit_name: str,
    consensus: str,
    transform_name: str,
    seed: int,
) -> list[dict]:
    rows = []
    for dimension in PROPERTIES:
        x, y = aligned_xy(embedding, vocabulary, ratings, dimension)
        pred = ridge_oof(x, y, seed)
        stat = spearmanr(pred, y)
        rows.append(
            {
                "fit_spec": fit_name,
                "consensus": consensus,
                "transform": transform_name,
                "dimension": dimension,
                "spearman": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "n": int(len(y)),
            }
        )
    return rows


def evaluate_all(fit_rows: pd.DataFrame, out_dir: Path, n_jobs: int, seed: int) -> pd.DataFrame:
    ratings = load_ratings()
    rows = []
    for fit_name, group in fit_rows.groupby("name"):
        first = group.iloc[0]
        spec_dir = out_dir / "fits" / fit_name
        runs = np.asarray([np.load(path) for path in sorted(spec_dir.glob("run_*.npy"))])
        np.save(spec_dir / "runs.npy", runs)
        vocabulary = json.loads(Path(first["vocab_path"]).read_text(encoding="utf-8"))
        embeddings = consensus_from_runs(runs, int(first["rank"]))
        tasks = []
        for consensus_name, embedding in embeddings.items():
            np.save(spec_dir / f"embedding_{consensus_name}.npy", embedding.astype(np.float32))
            for transform_name, transform in embedding_transforms():
                tasks.append(
                    delayed(evaluate_embedding)(
                        transform(embedding),
                        vocabulary,
                        ratings,
                        fit_name,
                        consensus_name,
                        transform_name,
                        seed,
                    )
                )
        chunks = Parallel(n_jobs=n_jobs, verbose=0)(tasks)
        rows.extend(row for chunk in chunks for row in chunk)
    df = pd.DataFrame(rows)
    meta_cols = [
        "name",
        "matrix_name",
        "rank",
        "max_outer",
        "max_inner",
        "rho",
        "tol",
    ]
    meta = fit_rows[meta_cols].drop_duplicates().rename(columns={"name": "fit_spec"})
    df = df.merge(meta, on="fit_spec", how="left")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.pivot_table(
        index=["fit_spec", "matrix_name", "rank", "max_outer", "max_inner", "consensus", "transform"],
        columns="dimension",
        values="spearman",
        aggfunc="mean",
    ).reset_index()
    summary["mean_spearman"] = summary[PROPERTIES].mean(axis=1)
    summary["max_spearman"] = summary[PROPERTIES].max(axis=1)
    return summary.sort_values(["max_spearman", "mean_spearman"], ascending=False)


def make_plots(results: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    best = (
        results.sort_values("spearman", ascending=False)
        .groupby("dimension", as_index=False)
        .first()
        .sort_values("spearman", ascending=False)
    )
    if SRF_RESULTS.exists():
        ref = pd.read_csv(SRF_RESULTS)
        ref = ref[ref["method"].eq("Ridge_CV")][["dimension", "correlation"]]
        best = best.merge(ref.rename(columns={"correlation": "existing_srf"}), on="dimension", how="left")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(best))
    width = 0.38 if "existing_srf" in best else 0.7
    ax.bar(x - width / 2, best["spearman"], width=width, label="Best sparse SRF sweep")
    if "existing_srf" in best:
        ax.bar(x + width / 2, best["existing_srf"], width=width, label="Existing SRF")
    ax.axhline(0.9, color="black", linewidth=1.0, linestyle=":")
    ax.set_ylabel("Held-out Spearman r")
    ax.set_xticks(x)
    ax.set_xticklabels(best["dimension"], rotation=35, ha="right")
    ax.set_ylim(0, max(0.95, float(best["spearman"].max()) + 0.05))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_dimension.pdf")
    plt.close(fig)

    top = summary.head(25).copy()
    top["label"] = top["fit_spec"] + "\n" + top["consensus"] + " / " + top["transform"]
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    ax.barh(np.arange(len(top)), top["max_spearman"])
    ax.axvline(0.9, color="black", linewidth=1.0, linestyle=":")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Best dimension Spearman r")
    fig.tight_layout()
    fig.savefig(plot_dir / "top_srf_specs.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=112)
    parser.add_argument("--eval-jobs", type=int, default=32)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=["quick", "broad"], default="quick")
    parser.add_argument("--top-n-words", type=int, default=4000)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    if OUT_DIR.exists() and not args.resume:
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_dir = OUT_DIR / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"
    log_path.write_text(f"starting sparse SRF sweep preset={args.preset}\n", encoding="utf-8")

    matrix_rows = []
    for spec in matrix_specs(args.preset):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"preparing matrix {spec.name}\n")
        matrix_rows.append(prepare_matrix(spec, matrix_dir, args.top_n_words))
    pd.DataFrame(matrix_rows).to_csv(OUT_DIR / "matrix_metadata.csv", index=False)

    fit_specs = build_fit_specs(matrix_rows, args.n_runs, args.seed, args.preset)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"dispatching {len(fit_specs)} SRF fits with n_jobs={args.n_jobs}\n")

    chunks = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_one)(spec, OUT_DIR) for spec in fit_specs
    )
    fit_rows = pd.DataFrame(chunks)
    fit_rows.to_csv(OUT_DIR / "fit_tasks.csv", index=False)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("evaluating embeddings\n")
    results = evaluate_all(fit_rows, OUT_DIR, args.eval_jobs, args.seed)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "all_results.csv", index=False)
    summary.to_csv(OUT_DIR / "all_summary.csv", index=False)
    make_plots(results, summary, OUT_DIR)

    best = results.loc[results["spearman"].idxmax()]
    readme = {
        "n_fit_tasks": int(len(fit_rows)),
        "n_results": int(len(results)),
        "max_spearman": float(best["spearman"]),
        "best_fit_spec": str(best["fit_spec"]),
        "best_dimension": str(best["dimension"]),
        "n_results_at_or_above_0.9": int((results["spearman"] >= 0.9).sum()),
    }
    (OUT_DIR / "README.md").write_text(
        "# Sparse SWOW SRF Sweep\n\n```json\n"
        + json.dumps(readme, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print((OUT_DIR / "all_summary.csv").resolve())


if __name__ == "__main__":
    main()
