# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.io import (
    load_latest_w_by_model,
)
from tools.utils.multiprocessing import submit_parallel_jobs
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


def zscore_columns(x: np.ndarray) -> np.ndarray:
    means = np.mean(x, axis=0, keepdims=True)
    stds = np.std(x, axis=0, ddof=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (x - means) / stds


def build_global_z(w_by_model: dict[str, np.ndarray], dtype: np.dtype = np.float32):
    keys = list(w_by_model.keys())
    n = next(iter(w_by_model.values())).shape[0]
    for v in w_by_model.values():
        if v.shape[0] != n:
            raise ValueError("All embeddings must have the same number of rows")
    slices = {}
    cols = 0
    for k in keys:
        d = w_by_model[k].shape[1]
        slices[k] = slice(cols, cols + d)
        cols += d
    z_all = np.empty((n, cols), dtype=dtype)
    for k in keys:
        s = slices[k]
        z_all[:, s] = zscore_columns(w_by_model[k]).astype(dtype, copy=False)
    return z_all, slices


def compute_universality_scores(
    w_by_model: dict[str, np.ndarray], use_absolute: bool = True, n_jobs: int = -1
) -> pd.DataFrame:
    z_all, slices = build_global_z(w_by_model)
    n = z_all.shape[0]
    keys = list(w_by_model.keys())

    def per_model(model_key: str):
        s = slices[model_key]
        c = z_all[:, s].T @ z_all / (n - 1)
        if use_absolute:
            c = np.abs(c)
        m = np.ones(c.shape[1], dtype=bool)
        m[s.start : s.stop] = False
        mx = c[:, m].max(axis=1)
        return model_key, float(mx.mean()), mx

    args = [(k,) for k in keys]
    results = submit_parallel_jobs(per_model, args, {"n_jobs": n_jobs, "verbose": 0})

    data = []
    for k, score, per_dim in results:
        data.append({"model": k, "universality": score, "num_dims": per_dim.shape[0]})
    df = (
        pd.DataFrame(data)
        .sort_values("universality", ascending=False)
        .reset_index(drop=True)
    )
    return df


def plot_universality_bar(df: pd.DataFrame, out_dir: Path, fname: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.2, 2.0), dpi=300)
    order = df.sort_values("universality", ascending=False)
    x = np.arange(len(order))
    plt.bar(x, order["universality"].values, color="#4C78A8")
    plt.xticks(x, order["model"].values, rotation=90, fontsize=7)
    plt.ylabel("Universality score", fontsize=7)
    plt.title("Universality score by model", fontsize=7)
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path)
    plt.close()
    return out_path


# %%
def compute_model_similarity_matrix(
    w_by_model: dict[str, np.ndarray], use_absolute: bool = True, n_jobs: int = -1
) -> pd.DataFrame:
    z_all, slices = build_global_z(w_by_model)
    n = z_all.shape[0]
    keys = list(w_by_model.keys())

    def per_model_directional(model_key: str):
        s = slices[model_key]
        c = (z_all[:, s].T @ z_all) / (n - 1)
        if use_absolute:
            c = np.abs(c)
        scores = {}
        for other in keys:
            if other == model_key:
                continue
            so = slices[other]
            mx = c[:, so].max(axis=1).mean()
            scores[other] = float(mx)
        return model_key, scores

    args = [(k,) for k in keys]
    res = submit_parallel_jobs(
        per_model_directional,
        args,
        {"n_jobs": n_jobs, "verbose": 0, "prefer": "threads"},
    )

    dir_mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for k, d in res:
        for other, v in d.items():
            dir_mat.loc[k, other] = v

    sim = (dir_mat + dir_mat.T) / 2.0
    np.fill_diagonal(sim.values, 1.0)
    return sim


def plot_model_similarity_heatmap(sim: pd.DataFrame, out_dir: Path, fname: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = sim.copy()
    d = 1.0 - s.values
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d), method="average")
    order = leaves_list(z)
    ordered_labels = s.index.to_numpy()[order]
    s_ord = s.loc[ordered_labels, ordered_labels]

    plt.figure(figsize=(3.2, 3.2), dpi=300)
    im = plt.imshow(s_ord.values, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.xticks(np.arange(len(s_ord)), s_ord.columns, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(s_ord)), s_ord.index, fontsize=7)
    plt.title("Model similarity", fontsize=7)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_model_map(sim: pd.DataFrame, out_dir: Path, fname: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = sim.values - sim.values.mean(axis=0, keepdims=True)
    p = PCA(n_components=2, random_state=0).fit_transform(x)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(p[:, 0], p[:, 1], s=10, c="#4C78A8")
    for i, label in enumerate(sim.index):
        plt.text(p[i, 0], p[i, 1], label, fontsize=7)
    plt.title("Model map", fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path)
    plt.close()
    return out_path


def compute_dimension_uniqueness(
    w_by_model: dict[str, np.ndarray], use_absolute: bool = True, n_jobs: int = -1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_all, slices = build_global_z(w_by_model)
    n = z_all.shape[0]
    keys = list(w_by_model.keys())

    def per_model(model_key: str):
        s = slices[model_key]
        c = (z_all[:, s].T @ z_all) / (n - 1)
        if use_absolute:
            c = np.abs(c)
        m = np.ones(c.shape[1], dtype=bool)
        m[s.start : s.stop] = False
        mx = c[:, m].max(axis=1)
        idx = np.arange(s.stop - s.start)
        return model_key, idx, mx

    args = [(k,) for k in keys]
    res = submit_parallel_jobs(
        per_model,
        args,
        {"n_jobs": n_jobs, "verbose": 0, "prefer": "threads"},
    )

    rows = []
    for k, idx, mx in res:
        for i, v in zip(idx, mx):
            rows.append({"model": k, "dim": int(i), "best_other_corr": float(v)})
    df = pd.DataFrame(rows)

    agg = (
        df.groupby("model")["best_other_corr"]
        .agg(["mean", "median"])
        .rename(
            columns={"mean": "mean_best_other_corr", "median": "median_best_other_corr"}
        )
        .reset_index()
    )
    return df, agg


def plot_uniqueness_summaries(
    df_summary: pd.DataFrame, out_dir: Path, prefix: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(30, 10.0), dpi=300)
    order = df_summary.sort_values("mean_best_other_corr")
    x = np.arange(len(order))
    plt.bar(x, order["mean_best_other_corr"].values, color="#F58518")
    plt.xticks(x, order["model"].values, rotation=90, fontsize=7)
    plt.ylabel("Mean best-other corr", fontsize=7)
    plt.title("Dimension uniqueness", fontsize=7)
    plt.tight_layout()
    bar_path = out_dir / f"{prefix}_mean.png"
    plt.savefig(bar_path)
    plt.close()

    plt.figure(figsize=(30, 10), dpi=300)
    plt.barh(
        np.arange(len(order)),
        1.0 - order["mean_best_other_corr"].values,
        color="#54A24B",
    )
    plt.yticks(np.arange(len(order)), order["model"].values, fontsize=7)
    plt.xlabel("Uniqueness (1 - mean corr)", fontsize=7)
    plt.title("Dimension uniqueness share", fontsize=7)
    plt.tight_layout()
    share_path = out_dir / f"{prefix}_share.png"
    plt.savefig(share_path)
    plt.close()
    return bar_path, share_path


def plot_uniqueness_hist(df_dims: pd.DataFrame, out_dir: Path, fname: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.0, 2.0), dpi=300)
    v = df_dims["best_other_corr"].values
    bins = np.linspace(0.0, 1.0, 26)
    plt.hist(v, bins=bins, color="#E45756", alpha=0.9)
    plt.xlabel("Best-other corr", fontsize=7)
    plt.ylabel("Count", fontsize=7)
    plt.title("Distribution of best-other corr", fontsize=7)
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path)
    plt.close()
    return out_path


# %%

root = Path("/SSD/datasets/similarity_factorization/factorize_models")
rank = 50
w_by_model = load_latest_w_by_model(root, rank=rank, mmap_mode=None, n_jobs=-1)


# %%
df_univ = compute_universality_scores(w_by_model, use_absolute=True, n_jobs=-1)


# %%
base = Path(__file__).resolve().parents[1]
out_dir = base / "results" / "universality"
plot_universality_bar(df_univ, out_dir, f"universality_rank_{rank}.png")


# %%
sim = compute_model_similarity_matrix(w_by_model, use_absolute=True, n_jobs=-1)
plot_model_similarity_heatmap(sim, out_dir, f"model_similarity_rank_{rank}.png")
plot_model_map(sim, out_dir, f"model_map_rank_{rank}.png")


# %%
df_dims, df_dim_summary = compute_dimension_uniqueness(
    w_by_model, use_absolute=True, n_jobs=-1
)
plot_uniqueness_summaries(df_dim_summary, out_dir, f"uniqueness_rank_{rank}")
plot_uniqueness_hist(df_dims, out_dir, f"uniqueness_hist_rank_{rank}.png")
