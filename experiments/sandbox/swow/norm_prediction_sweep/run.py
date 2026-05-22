"""Sweep SWOW embeddings for Glasgow norm prediction.

This sandbox keeps the production SWOW analysis untouched. It evaluates cheap
cached-embedding variants first, then optional SRF refits over alternative SWOW
similarity constructions and optimization budgets.

Example:
    ./.venv/bin/python experiments/sandbox/swow/norm_prediction_sweep/run.py --stage cached --n-jobs 16
    ./.venv/bin/python experiments/sandbox/swow/norm_prediction_sweep/run.py --stage scout --n-jobs 8 --run-jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.swow import load_swow_ppmi, load_swow_similarity  # noqa: E402
from pysrf import SRF  # noqa: E402
from pysrf.consensus import AlignedConsensus, align_embeddings  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
SWOW_DIR = DATA_DIR / "small-world-of-words"
RATINGS_PATH = DATA_DIR / "semantic_norms" / "glasgow_norms.csv"
CONSENSUS_DIR = (
    PROJECT_ROOT / "experiments" / "datasets" / "consensus" / "outputs" / "swow"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / time.strftime("%y%m%d_%H%M%S")

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
class FitSpec:
    name: str
    method: str = "ppmi"
    use_all_responses: bool = True
    symmetrization: str = "sum"
    zeros_as_missing: bool = True
    bidirectional_only: bool = False
    alpha: float = 0.75
    top_n_words: int | None = None
    min_word_length: int = 1
    rank: int = 128
    rho: float = 3.0
    max_outer: int = 25
    max_inner: int = 30
    tol: float = 0.0
    n_runs: int = 3


def load_ratings() -> pd.DataFrame:
    df = pd.read_csv(RATINGS_PATH)
    df["word"] = df["word"].str.lower()
    df = df.rename(columns={"semsize": "size"})
    return df[["word", *PROPERTIES]]


def load_baseline_vocabulary() -> list[str]:
    _, vocabulary, _ = load_swow_ppmi(
        SWOW_DIR,
        use_all_responses=True,
        top_n_words=None,
        min_word_length=1,
        symmetrization="sum",
        bidirectional_only=False,
    )
    return vocabulary


def aligned_xy(
    embedding: np.ndarray, vocabulary: list[str], ratings: pd.DataFrame, dimension: str
) -> tuple[np.ndarray, np.ndarray]:
    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}
    valid = ratings[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]
    indices = np.array([word_to_idx[word] for word in valid["word"]], dtype=int)
    return embedding[indices], valid[dimension].to_numpy(float)


def ridge_oof(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> np.ndarray:
    outer = KFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    inner = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    alphas = np.logspace(-5, 4, 25)
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in outer.split(x):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=inner))
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    return pred


def eval_embedding(
    embedding: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    variant: str,
    decoder: str,
    seed: int = 42,
) -> list[dict]:
    rows = []
    for dimension in PROPERTIES:
        x, y = aligned_xy(embedding, vocabulary, ratings, dimension)
        if len(y) < 50:
            continue
        if decoder == "ridge":
            pred = ridge_oof(x, y, seed=seed)
        elif decoder == "ridge_poly2":
            pred = ridge_oof(
                PolynomialFeatures(degree=2, include_bias=False).fit_transform(x),
                y,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder}")
        stat = spearmanr(pred, y)
        rows.append(
            {
                "variant": variant,
                "decoder": decoder,
                "dimension": dimension,
                "spearman": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "n": int(len(y)),
            }
        )
    return rows


def cached_variants() -> list[tuple[str, np.ndarray]]:
    embedding = np.load(CONSENSUS_DIR / "embedding.npy")
    variants: list[tuple[str, np.ndarray]] = [("cached_selected_raw", embedding)]

    runs_path = CONSENSUS_DIR / "runs.npy"
    if runs_path.exists():
        runs = np.load(runs_path)
        variants.extend(
            [
                ("cached_runs_mean", runs.mean(axis=0)),
                ("cached_runs_median", np.median(runs, axis=0)),
            ]
        )

    reliability_path = CONSENSUS_DIR / "cv_reliability.npy"
    if reliability_path.exists():
        reliability = np.load(reliability_path)
        order = np.argsort(reliability)[::-1]
        variants.extend(
            [
                ("cached_reliability_weighted", embedding * reliability[None, :]),
                ("cached_reliability_top200", embedding[:, order[:200]]),
                ("cached_reliability_top150", embedding[:, order[:150]]),
                ("cached_reliability_top100", embedding[:, order[:100]]),
            ]
        )

    row_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    variants.extend(
        [
            ("cached_log1p", np.log1p(embedding)),
            ("cached_sqrt", np.sqrt(embedding)),
            ("cached_binary", (embedding > 0).astype(float)),
            ("cached_row_l2", embedding / np.maximum(row_norm, 1e-12)),
            (
                "cached_raw_plus_rowstats",
                np.column_stack(
                    [
                        embedding,
                        row_norm,
                        (embedding > 0).sum(axis=1),
                        embedding.mean(axis=1),
                        embedding.max(axis=1),
                    ]
                ),
            ),
        ]
    )
    return variants


def run_cached(n_jobs: int, decoders: list[str]) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings()
    vocabulary = load_baseline_vocabulary()
    variants = cached_variants()

    tasks = [
        delayed(eval_embedding)(embedding, vocabulary, ratings, name, decoder)
        for name, embedding in variants
        for decoder in decoders
    ]
    chunks = Parallel(n_jobs=n_jobs, verbose=10)(tasks)
    rows = [row for chunk in chunks for row in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "cached_results.csv", index=False)
    write_summary(df, OUT_DIR / "cached_summary.csv")
    return df


def load_similarity_for_spec(spec: FitSpec) -> tuple[np.ndarray, list[str], dict]:
    similarity, vocabulary, metadata = load_swow_similarity(
        SWOW_DIR,
        method=spec.method,
        use_all_responses=spec.use_all_responses,
        top_n_words=spec.top_n_words,
        min_word_length=spec.min_word_length,
        symmetrization=spec.symmetrization,
        bidirectional_only=spec.bidirectional_only,
        alpha=spec.alpha,
    )
    similarity = np.asarray(similarity, dtype=np.float64)
    if spec.method == "ppmi" and spec.zeros_as_missing:
        diag = np.diag(similarity).copy()
        similarity[similarity == 0] = np.nan
        np.fill_diagonal(similarity, diag)
    return similarity, vocabulary, metadata


def fit_one_run(similarity: np.ndarray, spec: FitSpec, seed: int) -> np.ndarray:
    model = SRF(
        rank=spec.rank,
        rho=spec.rho,
        max_outer=spec.max_outer,
        max_inner=spec.max_inner,
        tol=spec.tol,
        random_state=seed,
        check_input=False,
    )
    return model.fit_transform(similarity)


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


def run_fit_spec(
    spec: FitSpec,
    out_dir: Path,
    run_jobs: int,
    eval_jobs: int,
    decoders: list[str],
    seed: int,
) -> pd.DataFrame:
    spec_dir = out_dir / "fits" / spec.name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "spec.json").write_text(json.dumps(asdict(spec), indent=2))

    start = time.time()
    similarity, vocabulary, metadata = load_similarity_for_spec(spec)
    (spec_dir / "similarity_metadata.json").write_text(json.dumps(metadata, indent=2))

    seeds = [seed + i for i in range(spec.n_runs)]
    runs = Parallel(n_jobs=run_jobs, verbose=10)(
        delayed(fit_one_run)(similarity, spec, run_seed) for run_seed in seeds
    )
    runs_array = np.asarray(runs)
    np.save(spec_dir / "runs.npy", runs_array)

    ratings = load_ratings()
    embeddings = consensus_from_runs(runs_array, spec.rank)
    rows = []
    for suffix, embedding in embeddings.items():
        np.save(spec_dir / f"embedding_{suffix}.npy", embedding)
        tasks = [
            delayed(eval_embedding)(
                transform(embedding),
                vocabulary,
                ratings,
                f"{spec.name}_{suffix}_{transform_name}",
                decoder,
                seed,
            )
            for transform_name, transform in embedding_transforms()
            for decoder in decoders
        ]
        chunks = Parallel(n_jobs=eval_jobs, verbose=0)(tasks)
        rows.extend(row for chunk in chunks for row in chunk)

    df = pd.DataFrame(rows)
    df["fit_spec"] = spec.name
    df["elapsed_sec"] = time.time() - start
    for key, value in asdict(spec).items():
        df[key] = value
    df.to_csv(spec_dir / "results.csv", index=False)
    return df


def run_fit_seed_task(
    spec: FitSpec,
    out_dir: Path,
    seed: int,
    run_idx: int,
) -> dict:
    """Fit one spec/seed task and write the embedding to disk."""
    spec_dir = out_dir / "fits" / spec.name
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / "spec.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(asdict(spec), indent=2))

    start = time.time()
    similarity, vocabulary, metadata = load_similarity_for_spec(spec)
    metadata_path = spec_dir / "similarity_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2))

    embedding = fit_one_run(similarity, spec, seed)
    run_path = spec_dir / f"run_{run_idx:03d}.npy"
    np.save(run_path, embedding)
    vocab_path = spec_dir / "vocabulary.json"
    if not vocab_path.exists():
        vocab_path.write_text(json.dumps(vocabulary))

    return {
        "fit_spec": spec.name,
        "run_idx": run_idx,
        "seed": seed,
        "path": str(run_path),
        "elapsed_sec": time.time() - start,
    }


def evaluate_saved_spec(
    spec: FitSpec,
    out_dir: Path,
    eval_jobs: int,
    decoders: list[str],
    seed: int,
) -> pd.DataFrame:
    """Build consensus variants from saved per-seed runs and evaluate them."""
    spec_dir = out_dir / "fits" / spec.name
    run_paths = sorted(spec_dir.glob("run_*.npy"))
    if not run_paths:
        raise FileNotFoundError(f"No run files for {spec.name} in {spec_dir}")
    runs_array = np.asarray([np.load(path) for path in run_paths])
    np.save(spec_dir / "runs.npy", runs_array)

    vocabulary = json.loads((spec_dir / "vocabulary.json").read_text())
    ratings = load_ratings()
    embeddings = consensus_from_runs(runs_array, spec.rank)

    rows = []
    for suffix, embedding in embeddings.items():
        np.save(spec_dir / f"embedding_{suffix}.npy", embedding)
        tasks = [
            delayed(eval_embedding)(
                transform(embedding),
                vocabulary,
                ratings,
                f"{spec.name}_{suffix}_{transform_name}",
                decoder,
                seed,
            )
            for transform_name, transform in embedding_transforms()
            for decoder in decoders
        ]
        chunks = Parallel(n_jobs=eval_jobs, verbose=0)(tasks)
        rows.extend(row for chunk in chunks for row in chunk)

    df = pd.DataFrame(rows)
    df["fit_spec"] = spec.name
    for key, value in asdict(spec).items():
        df[key] = value
    df.to_csv(spec_dir / "results.csv", index=False)
    return df


def embedding_transforms() -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("raw", lambda x: x),
        ("log1p", lambda x: np.log1p(x)),
        ("sqrt", lambda x: np.sqrt(x)),
        ("binary", lambda x: (x > 0).astype(float)),
        ("row_l2", lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)),
    ]


def scout_specs() -> list[FitSpec]:
    specs: list[FitSpec] = []
    base = {
        "rank": 128,
        "max_outer": 25,
        "max_inner": 30,
        "tol": 0.0,
        "n_runs": 3,
    }
    for sym, zeros, bidir in [
        ("sum", True, False),
        ("sum", False, False),
        ("arithmetic_mean", True, False),
        ("geometric_mean", True, False),
        ("sum", True, True),
    ]:
        name = f"ppmi_R123_{sym}_zeros{'missing' if zeros else 'zero'}_bidir{int(bidir)}_k128_o25"
        specs.append(
            FitSpec(
                name=name,
                symmetrization=sym,
                zeros_as_missing=zeros,
                bidirectional_only=bidir,
                **base,
            )
        )
    specs.append(
        FitSpec(
            name="ppmi_R1_sum_zerosmissing_bidir0_k128_o25",
            use_all_responses=False,
            **base,
        )
    )
    specs.append(
        FitSpec(
            name="ppmi_R123_sum_zerosmissing_bidir0_k64_o50",
            rank=64,
            max_outer=50,
            max_inner=30,
            tol=0.0,
            n_runs=3,
        )
    )
    specs.append(
        FitSpec(
            name="ppmi_R123_sum_zerosmissing_bidir0_k128_o100_inner100",
            rank=128,
            max_outer=100,
            max_inner=100,
            tol=0.0,
            n_runs=3,
        )
    )
    for alpha in [0.50, 0.75]:
        specs.append(
            FitSpec(
                name=f"rw_R123_alpha{str(alpha).replace('.', '')}_top4000_k96_o25",
                method="rw",
                alpha=alpha,
                top_n_words=4000,
                rank=96,
                max_outer=25,
                max_inner=30,
                tol=0.0,
                n_runs=3,
            )
        )
    return specs


def wide_specs() -> list[FitSpec]:
    specs: list[FitSpec] = []
    ppmi_variants = list(
        product(
            [True, False],
            ["sum", "arithmetic_mean", "geometric_mean", "max"],
            [True, False],
            [64, 128, 268],
        )
    )
    for use_all, sym, zeros, rank in ppmi_variants:
        if sym == "geometric_mean" and not zeros:
            continue
        name = (
            f"ppmi_R{'123' if use_all else '1'}_{sym}_"
            f"zeros{'missing' if zeros else 'zero'}_k{rank}_o50"
        )
        specs.append(
            FitSpec(
                name=name,
                use_all_responses=use_all,
                symmetrization=sym,
                zeros_as_missing=zeros,
                rank=rank,
                max_outer=50,
                max_inner=60,
                n_runs=4,
            )
        )
    for rank, outer, inner in [(128, 100, 100), (268, 100, 100), (268, 150, 100)]:
        specs.append(
            FitSpec(
                name=f"ppmi_R123_sum_zerosmissing_k{rank}_o{outer}_inner{inner}",
                rank=rank,
                max_outer=outer,
                max_inner=inner,
                n_runs=4,
            )
        )
    for alpha, rank in product([0.50, 0.65, 0.75], [64, 128]):
        specs.append(
            FitSpec(
                name=f"rw_R123_alpha{str(alpha).replace('.', '')}_top5000_k{rank}_o50",
                method="rw",
                alpha=alpha,
                top_n_words=5000,
                rank=rank,
                max_outer=50,
                max_inner=60,
                n_runs=4,
            )
        )
    return specs


def write_summary(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    summary = (
        df.pivot_table(
            index=["variant", "decoder"],
            columns="dimension",
            values="spearman",
            aggfunc="mean",
        )
        .reset_index()
    )
    summary["mean_spearman"] = summary[PROPERTIES].mean(axis=1)
    summary["max_spearman"] = summary[PROPERTIES].max(axis=1)
    summary = summary.sort_values("mean_spearman", ascending=False)
    summary.to_csv(path, index=False)


def append_results(df: pd.DataFrame, path: Path) -> None:
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cached", "scout", "wide"], default="cached")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--run-jobs", type=int, default=4)
    parser.add_argument(
        "--spec-jobs",
        type=int,
        default=1,
        help=(
            "Number of fit specs to run concurrently. Total SRF worker count is "
            "approximately spec_jobs * run_jobs, capped by each spec's n_runs."
        ),
    )
    parser.add_argument(
        "--run-level-parallel",
        action="store_true",
        help="Parallelize individual spec/seed fits globally for maximum CPU use.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decoder",
        action="append",
        choices=["ridge", "ridge_poly2"],
        default=None,
        help="Decoder(s) to evaluate. Defaults to ridge.",
    )
    args = parser.parse_args()

    decoders = args.decoder or ["ridge"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "command.json").write_text(json.dumps(vars(args), indent=2))
    latest = Path(__file__).resolve().parent / "outputs" / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(OUT_DIR.name)

    if args.stage == "cached":
        df = run_cached(args.n_jobs, decoders)
        print(f"Wrote {len(df)} cached rows to {OUT_DIR}")
        return

    cached_df = run_cached(args.n_jobs, decoders)
    append_results(cached_df, OUT_DIR / "all_results.csv")

    specs = scout_specs() if args.stage == "scout" else wide_specs()
    if args.run_level_parallel:
        fit_tasks = [
            delayed(run_fit_seed_task)(
                spec=spec,
                out_dir=OUT_DIR,
                seed=args.seed + run_idx,
                run_idx=run_idx,
            )
            for spec in specs
            for run_idx in range(spec.n_runs)
        ]
        print(
            f"Running {len(fit_tasks)} spec/seed fits with n_jobs={args.spec_jobs}",
            flush=True,
        )
        fit_rows = Parallel(n_jobs=args.spec_jobs, verbose=10)(fit_tasks)
        pd.DataFrame(fit_rows).to_csv(OUT_DIR / "fit_tasks.csv", index=False)

        print("Evaluating saved fits", flush=True)
        eval_chunks = Parallel(n_jobs=min(args.n_jobs, len(specs)), verbose=10)(
            delayed(evaluate_saved_spec)(
                spec=spec,
                out_dir=OUT_DIR,
                eval_jobs=1,
                decoders=decoders,
                seed=args.seed,
            )
            for spec in specs
        )
        fit_df = pd.concat(eval_chunks, ignore_index=True) if eval_chunks else pd.DataFrame()
        append_results(fit_df, OUT_DIR / "all_results.csv")
        all_df = pd.read_csv(OUT_DIR / "all_results.csv")
        write_summary(all_df, OUT_DIR / "all_summary.csv")
        best = pd.read_csv(OUT_DIR / "all_summary.csv").head(20)
        print(
            best[["variant", "decoder", "mean_spearman", "max_spearman"]]
            .to_string(index=False)
        )
        return

    if args.spec_jobs <= 1:
        for spec in specs:
            print(f"\n=== Running {spec.name} ===", flush=True)
            df = run_fit_spec(
                spec=spec,
                out_dir=OUT_DIR,
                run_jobs=args.run_jobs,
                eval_jobs=args.n_jobs,
                decoders=decoders,
                seed=args.seed,
            )
            append_results(df, OUT_DIR / "all_results.csv")
            all_df = pd.read_csv(OUT_DIR / "all_results.csv")
            write_summary(all_df, OUT_DIR / "all_summary.csv")
            best = pd.read_csv(OUT_DIR / "all_summary.csv").head(10)
            print(
                best[["variant", "decoder", "mean_spearman", "max_spearman"]]
                .to_string(index=False)
            )
    else:
        print(
            f"Running {len(specs)} specs concurrently with spec_jobs={args.spec_jobs}, "
            f"run_jobs={args.run_jobs}",
            flush=True,
        )
        chunks = Parallel(n_jobs=args.spec_jobs, verbose=10)(
            delayed(run_fit_spec)(
                spec=spec,
                out_dir=OUT_DIR,
                run_jobs=args.run_jobs,
                eval_jobs=1,
                decoders=decoders,
                seed=args.seed,
            )
            for spec in specs
        )
        fit_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        append_results(fit_df, OUT_DIR / "all_results.csv")
        all_df = pd.read_csv(OUT_DIR / "all_results.csv")
        write_summary(all_df, OUT_DIR / "all_summary.csv")
        best = pd.read_csv(OUT_DIR / "all_summary.csv").head(20)
        print(
            best[["variant", "decoder", "mean_spearman", "max_spearman"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
