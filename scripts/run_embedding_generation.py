# %%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pysrf import SRF
from pysrf import cross_val_score
from pysrf.bounds import estimate_p_bound_fast
from tools.rsa import compute_similarity
import matplotlib.pyplot as plt
import json


def estimate_observed_fraction(
    s: np.ndarray,
    strategy: str = "mean",
    random_state: int = 0,
    n_jobs: int = -1,
    **kwargs,
) -> tuple[float, float, float]:
    p_min, p_max, _ = estimate_p_bound_fast(
        s, random_state=random_state, n_jobs=n_jobs, **kwargs
    )
    if strategy == "pmin":
        obs = p_min
    elif strategy == "pmax":
        obs = p_max
    else:
        obs = 0.5 * (p_min + p_max)
    obs = float(np.clip(obs, 0.01, 0.99))
    return obs, float(p_min), float(p_max)


def select_rank(
    s: np.ndarray,
    rank_range: tuple[int, int] = (5, 50),
    step: int = 1,
    n_repeats: int = 5,
    sampling_fraction: float = 0.8,
    random_state: int = 0,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[int, float, pd.DataFrame]:
    ranks = list(range(rank_range[0], rank_range[1] + 1, step))
    grid = {"rank": ranks}
    res = cross_val_score(
        s,
        param_grid=grid,
        n_repeats=n_repeats,
        sampling_fraction=observed_fraction,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        missing_values=np.nan,
        fit_final_estimator=False,
    )
    return int(res.best_params_["rank"]), float(res.best_score_), res.cv_results_


def factorize_runs(
    s: np.ndarray,
    rank: int,
    n_runs: int = 50,
    random_state: int = 0,
    n_jobs: int = -1,
    rho: float = 3.0,
    max_outer: int = 15,
    max_inner: int = 40,
    tol: float = 1e-4,
    init: str = "random_sqrt",
    missing_values: float | None = np.nan,
    bounds: tuple[float | None, float | None] | None = None,
    verbose: int = 0,
) -> np.ndarray:
    vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
    if bounds is None:
        bounds = (vmin, vmax)

    def _fit(seed: int) -> np.ndarray:
        model = SRF(
            rank=rank,
            rho=rho,
            max_outer=max_outer,
            max_inner=max_inner,
            tol=tol,
            init=init,
            random_state=seed,
            missing_values=missing_values,
            bounds=bounds,
            verbose=False,
        )
        return model.fit_transform(s)

    seeds = [random_state + i for i in range(n_runs)]
    ws = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_fit)(sd) for sd in seeds)
    return np.hstack(ws)


def _norm_columns(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _merge_clusters_by_median(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    med = np.array([np.median(x[:, labels == i], axis=1) for i in range(k)]).T
    sums = np.array([np.sum(x[:, labels == i]) for i in range(k)])
    order = np.argsort(-sums)
    return med[:, order]


def cluster_embeddings(
    stacked: np.ndarray,
    name: str = "model",
    min_clusters: int = 2,
    max_clusters: int = 100,
    step: int = 2,
    random_state: int = 0,
    cluster_kwargs: dict | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    x = _norm_columns(stacked).T
    rng = range(min_clusters, max_clusters + 1, step)
    ck = {"random_state": random_state, "init": "k-means++", "n_init": 30}
    if cluster_kwargs:
        ck.update(cluster_kwargs)

    def _score_k(k: int) -> dict:
        km = KMeans(n_clusters=k, **ck)
        km.fit(x)
        lab = km.labels_
        return {
            "model": name,
            "n_clusters": k,
            "silhouette_score": float(silhouette_score(x, lab)),
        }

    records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_score_k)(k) for k in rng
    )
    df = pd.DataFrame(records)
    best_k = int(df.iloc[np.argmax(df["silhouette_score"])]["n_clusters"])
    km = KMeans(n_clusters=best_k, **ck)
    lab = km.fit(x).labels_
    final = _merge_clusters_by_median(stacked, lab, best_k)
    df["best_k"] = best_k
    return df, final, best_k


def run_embedding_pipeline(
    s,
    rank_range: tuple[int, int] = (5, 50),
    step: int = 1,
    n_runs: int = 50,
    n_repeats: int = 5,
    random_state: int = 0,
    n_jobs: int = -1,
    fraction_strategy: str = "mean",
    fraction_kwargs: dict | None = None,
    rho: float = 3.0,
    max_outer: int = 15,
    max_inner: int = 40,
    tol: float = 1e-4,
    init: str = "random_sqrt",
    cluster_min: int | None = None,
    cluster_max: int | None = None,
    cluster_step: int = 2,
    cluster_kwargs: dict | None = None,
    metric: str = "linear",
    output_dir: str | Path | None = None,
    name: str = "model",
    verbose: int = 0,
    estimate_fraction: bool = True,
) -> dict:
    if estimate_fraction:
        obs, p_min, p_max = estimate_observed_fraction(
            s,
            strategy=fraction_strategy,
            random_state=random_state,
            n_jobs=n_jobs,
            **(fraction_kwargs or {}),
        )
    else:
        obs = fraction_kwargs["sampling_fraction"]
        p_min = fraction_kwargs["p_min"]
        p_max = fraction_kwargs["p_max"]
    best_rank, cv_score, cv_df = select_rank(
        s,
        rank_range=rank_range,
        step=step,
        n_repeats=n_repeats,
        sampling_fraction=obs,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    stacked = factorize_runs(
        s,
        rank=best_rank,
        n_runs=n_runs,
        random_state=random_state,
        n_jobs=n_jobs,
        rho=rho,
        max_outer=max_outer,
        max_inner=max_inner,
        tol=tol,
        init=init,
        verbose=verbose,
    )
    cmin = cluster_min if cluster_min is not None else max(2, best_rank // 2)
    cmax = cluster_max if cluster_max is not None else min(best_rank * 2, 100)
    df, final, best_k = cluster_embeddings(
        stacked,
        name=name,
        min_clusters=cmin,
        max_clusters=cmax,
        step=cluster_step,
        random_state=random_state,
        cluster_kwargs=cluster_kwargs,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    out = {
        "sampling_fraction": obs,
        "p_min": p_min,
        "p_max": p_max,
        "optimal_rank": best_rank,
        "cv_score": cv_score,
        "cv_results": cv_df,
        "stacked_embeddings": stacked,
        "cluster_results": df,
        "best_k": best_k,
        "final_embedding": final,
    }
    if output_dir is not None:
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / f"{name}_stacked.npy", stacked)
        np.save(path / f"{name}_embedding.npy", final)
        df.to_csv(path / f"{name}_clustering.csv", index=False)
        if isinstance(cv_df, pd.DataFrame):
            cv_df.to_csv(path / f"{name}_cv.csv", index=False)

        m = cv_df.groupby("rank")["score"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=300)
        ax.plot(m["rank"], m["score"], marker="o", ms=2)
        ax.axvline(best_rank, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Rank", fontsize=7)
        ax.set_ylabel("Score", fontsize=7)
        ax.set_title("Cross-validation score", fontsize=8)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(7)
        fig.tight_layout()
        fig.savefig(path / "cv_score.png", bbox_inches="tight")
        plt.close(fig)

        dd = df if "model" not in df.columns else df[df["model"] == name]
        mm = dd.sort_values("n_clusters")
        fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=300)
        ax.plot(mm["n_clusters"], mm["silhouette_score"], marker="o", ms=2)
        ax.set_xlabel("Clusters", fontsize=7)
        ax.set_ylabel("Silhouette score", fontsize=7)
        ax.set_title("Silhouette score by k", fontsize=8)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(7)
        fig.tight_layout()
        fig.savefig(path / "silhouette_scores.png", bbox_inches="tight")
        plt.close(fig)

        data_pf = {
            "sampling_fraction": float(obs),
            "p_min": float(p_min),
            "p_max": float(p_max),
        }
        with open(path / "observed_fraction.json", "w") as f:
            json.dump(data_pf, f)

        fig, ax = plt.subplots(figsize=(3.0, 1.8), dpi=300)
        ax.bar(
            ["min", "mean", "max"],
            [p_min, obs, p_max],
            color=["#999999", "#1f77b4", "#999999"],
        )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Value", fontsize=7)
        ax.set_title("Observed fraction bounds", fontsize=8)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(7)
        fig.tight_layout()
        fig.savefig(path / "observed_fraction.png", bbox_inches="tight")
        plt.close(fig)
    return out


# %%

from datasets import load_dataset


name = "nsd"
ds = load_dataset(name)

x = ds.betas
rsm = compute_similarity(x, x, "gaussian_kernel")

breakpoint()


# # rsm = ds.group_rsm
# ds = ds.train_data[:300].reshape(ds.train_data[:300].shape[0], -1)
# from experiments.clustering.graph import construct_similarity_graph
# rsm = construct_similarity_graph(ds)

res = run_embedding_pipeline(
    rsm,
    rank_range=(5, 150),
    n_runs=50,
    n_jobs=-1,
    name=name,
    output_dir="/LOCAL/fmahner/similarity-factorization/results/pipeline",
    estimate_fraction=False,
    fraction_kwargs={
        "sampling_fraction": 0.6,
        "p_min": 0.1,
        "p_max": 0.9,
    },
)
