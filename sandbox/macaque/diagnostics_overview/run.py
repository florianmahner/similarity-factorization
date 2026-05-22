"""Diagnostics overview for the macaque dimensionality runs.

Reads existing artifacts only. The goal is to make the current failure mode
visible before launching another expensive 22k x 22k validation run.

Outputs:
- macaque_diagnostics_overview.png
- macaque_diagnostics_overview.pdf
- diagnostic_summary.json
- old_cv_fit_summary.csv
- long_partial_fit_summary.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_output_dir

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
DIM_OUT = ROOT / "experiments/datasets/dimensionality/outputs"
MACAQUE_DIR = DIM_OUT / "things_macaque22k"
MACAQUE_SIGMA1_DIR = DIM_OUT / "things_macaque22k_sigma1.0"
LONG_DIR = ROOT / "sandbox/macaque/rank_sweep_max_outer150/outputs/results"
PREVIEW_DIR = ROOT / "sandbox/macaque/preview_u_curve/outputs/results"
CONSENSUS_DIR = ROOT / "experiments/datasets/consensus/outputs/things_macaque22k"
OUTPUT_DIR = get_output_dir()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _last(values, default=np.nan):
    if values is None or len(values) == 0:
        return default
    return values[-1]


def _cv_block(dataset_dir: Path) -> dict | None:
    path = dataset_dir / "cross_validation.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    variant = payload.get("primary_variant", "5fold")
    block = payload["validations"][variant]
    return {"payload": payload, "variant": variant, "block": block}


def _coherence(dataset_dir: Path) -> dict | None:
    path = dataset_dir / "coherence_estimate.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    return {"payload": payload, "estimate": payload["estimate"]}


def _load_fit_records(root: Path, source: str) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*.json")):
        payload = _read_json(path)
        history = payload.get("history", {}) or {}
        srf_kwargs = payload.get("srf_kwargs", {})
        if not srf_kwargs:
            srf_kwargs = {
                "rho": payload.get("rho"),
                "max_inner": payload.get("max_inner"),
                "tol": payload.get("tol"),
                "max_outer": payload.get("max_outer"),
            }
        conv = history.get("converged", [])
        row = {
            "source": source,
            "file": path.name,
            "rank": int(payload["rank"]),
            "fold": int(payload.get("fold", -1)),
            "repeat": int(payload.get("repeat", 0)),
            "seed": int(payload.get("seed", -1)),
            "val_mse": float(payload.get("val_mse", np.nan)),
            "n_entries_val": int(payload.get("n_entries_val", -1)),
            "n_iter": int(payload.get("n_iter", 0)),
            "fit_time_s": float(payload.get("fit_time_s", np.nan)),
            "max_outer": int(srf_kwargs.get("max_outer", payload.get("max_outer", 0))),
            "max_inner": int(srf_kwargs.get("max_inner", payload.get("max_inner", 0))),
            "tol": float(srf_kwargs.get("tol", payload.get("tol", np.nan))),
            "final_converged": bool(_last(conv, False)),
            "final_rec_error": float(_last(history.get("rec_error", []))),
            "final_evar": float(_last(history.get("evar", []))),
            "final_primal": float(_last(history.get("primal_residual", []))),
            "final_dual": float(_last(history.get("dual_residual", []))),
            "history_len": max((len(v) for v in history.values()), default=0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _history_frame(root: Path, source: str) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*.json")):
        payload = _read_json(path)
        history = payload.get("history", {}) or {}
        if "rec_error" not in history:
            continue
        n = len(history["rec_error"])
        for i in range(n):
            rows.append(
                {
                    "source": source,
                    "file": path.name,
                    "rank": int(payload["rank"]),
                    "fold": int(payload.get("fold", -1)),
                    "iter": i + 1,
                    "rec_error": history["rec_error"][i],
                    "evar": history.get("evar", [np.nan] * n)[i],
                    "primal": history.get("primal_residual", [np.nan] * n)[i],
                    "dual": history.get("dual_residual", [np.nan] * n)[i],
                }
            )
    return pd.DataFrame(rows)


def _extract_logged_pstar(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.search(r"sampling_fraction p\*=([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None


def _estimate_train_fraction(n_entries_val: float, n: int, n_folds: int = 5) -> float:
    total_pairs = n * (n - 1) / 2
    pool_fraction = n_entries_val * n_folds / total_pairs
    return pool_fraction * (n_folds - 1) / n_folds


def _lineplot_cv(ax, cv: dict, coherence: dict | None, label: str, color: str) -> None:
    block = cv["block"]
    ranks = np.asarray(block["ranks"], dtype=float)
    means = np.asarray(block["val_mse_mean"], dtype=float)
    sem = np.asarray(block.get("val_mse_sem", np.zeros_like(means)), dtype=float)
    ax.plot(ranks, means, marker="o", color=color, label=label, linewidth=1.8)
    ax.fill_between(ranks, means - sem, means + sem, color=color, alpha=0.15, linewidth=0)
    ax.axvline(block["argmin_rank"], color=color, linestyle="-", alpha=0.35, linewidth=1.0)
    if coherence is not None:
        k = coherence["estimate"]["rank"]
        ax.axvline(k, color=color, linestyle="--", alpha=0.75, linewidth=1.2)
    ax.set_xlabel("rank")
    ax.set_ylabel("validation MSE")


def _plot_history_mean(ax, hist: pd.DataFrame, rank: int, source: str, color: str, label: str) -> None:
    subset = hist[(hist["rank"] == rank) & (hist["source"] == source)]
    if subset.empty:
        return
    stats = subset.groupby("iter")["rec_error"].agg(["mean", "sem"]).reset_index()
    ax.plot(stats["iter"], stats["mean"], color=color, linewidth=1.8, label=label)
    sem = stats["sem"].fillna(0.0).to_numpy()
    ax.fill_between(stats["iter"], stats["mean"] - sem, stats["mean"] + sem, color=color, alpha=0.15)


def _embedding_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    w = np.load(path, mmap_mode="r")
    col_mass = np.asarray(w.sum(axis=0))
    col_nnz = np.asarray((w > 1e-8).sum(axis=0))
    col_max = np.asarray(w.max(axis=0))
    return {
        "shape": list(w.shape),
        "sparsity": float(np.mean(w == 0)),
        "mass": col_mass.tolist(),
        "nnz": col_nnz.tolist(),
        "max": col_max.tolist(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv = _cv_block(MACAQUE_DIR)
    cv_sigma1 = _cv_block(MACAQUE_SIGMA1_DIR)
    coh = _coherence(MACAQUE_DIR)
    coh_sigma1 = _coherence(MACAQUE_SIGMA1_DIR)

    old = _load_fit_records(MACAQUE_DIR / "cv_per_fit", "old_cv")
    long = _load_fit_records(LONG_DIR, "long_partial")
    preview = _load_fit_records(PREVIEW_DIR, "preview_outer10")
    old_hist = _history_frame(MACAQUE_DIR / "cv_per_fit", "old_cv")
    long_hist = _history_frame(LONG_DIR, "long_partial")
    hist = pd.concat([old_hist, long_hist], ignore_index=True)

    old.to_csv(OUTPUT_DIR / "old_cv_fit_summary.csv", index=False)
    long.to_csv(OUTPUT_DIR / "long_partial_fit_summary.csv", index=False)

    n = int(cv["payload"]["n"]) if cv else int(coh["payload"]["n"])
    old_train_fraction = None
    long_train_fraction = None
    if not old.empty:
        old_train_fraction = _estimate_train_fraction(float(old["n_entries_val"].median()), n)
    if not long.empty:
        long_train_fraction = _estimate_train_fraction(float(long["n_entries_val"].median()), n)

    embedding = _embedding_stats(CONSENSUS_DIR / "embedding.npy")
    consensus_summary = _read_json(CONSENSUS_DIR / "summary.json") if (CONSENSUS_DIR / "summary.json").exists() else {}

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.0))
    ax = axes[0, 0]
    if cv:
        _lineplot_cv(ax, cv, coh, "macaque sigma=0.4", "#7f1d1d")
    if cv_sigma1:
        _lineplot_cv(ax, cv_sigma1, coh_sigma1, "macaque sigma=1.0", "#0f766e")
    ax.set_title("Existing CV curves: both hit upper edge")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    if not old.empty:
        grouped = old.groupby("rank").agg(
            n=("rank", "count"),
            n_iter=("n_iter", "mean"),
            converged=("final_converged", "mean"),
            val_mse=("val_mse", "mean"),
        )
        ax.bar(grouped.index.astype(str), grouped["n_iter"], color="#334155", alpha=0.85)
        ax.axhline(50, color="#dc2626", linestyle="--", linewidth=1.1, label="old max_outer=50")
        for x, (_, row) in enumerate(grouped.iterrows()):
            ax.text(x, row["n_iter"] + 1.2, f"{row['converged']:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Old CV: mean n_iter by rank; labels are converged fraction")
    ax.set_xlabel("rank")
    ax.set_ylabel("outer iterations")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    _plot_history_mean(ax, hist, 30, "old_cv", "#64748b", "old rank 30, max_outer=50")
    _plot_history_mean(ax, hist, 30, "long_partial", "#be123c", "partial long rank 30")
    ax.set_title("Training history at rank 30")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("train observed rec_error")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    _plot_history_mean(ax, hist, 50, "old_cv", "#64748b", "old rank 50, max_outer=50")
    _plot_history_mean(ax, hist, 50, "long_partial", "#be123c", "partial long rank 50")
    ax.set_title("Training history at rank 50")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("train observed rec_error")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2, 0]
    if coh:
        eigs = np.asarray(coh["estimate"].get("eigenvalues", []), dtype=float)
        leak = np.asarray(coh["estimate"].get("leakage", []), dtype=float)
        if eigs.size:
            x = np.arange(1, eigs.size + 1)
            ax.plot(x, eigs / eigs[0], color="#1d4ed8", linewidth=1.6, label="eigenvalue / lambda1")
            ax.set_yscale("log")
            ax.set_xscale("log")
        if leak.size:
            ax2 = ax.twinx()
            x2 = np.arange(1, leak.size + 1)
            ax2.plot(x2, leak, color="#f97316", linewidth=1.2, alpha=0.85, label="leakage")
            ax2.set_ylabel("leakage")
            ax2.spines["top"].set_visible(False)
        ax.axvline(coh["estimate"]["rank"], color="#111827", linestyle="--", linewidth=1.1, label="coherence k")
    ax.set_title("Coherence spectrum/leakage")
    ax.set_xlabel("dimension index")
    ax.set_ylabel("normalized eigenvalue")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2, 1]
    if embedding:
        mass = np.asarray(embedding["mass"], dtype=float)
        nnz = np.asarray(embedding["nnz"], dtype=float)
        order = np.argsort(mass)[::-1]
        ax.plot(np.arange(1, len(mass) + 1), mass[order], color="#7f1d1d", marker=".", linewidth=1.2, label="column mass")
        ax.set_xlabel("dimension sorted by mass")
        ax.set_ylabel("column mass")
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(nnz) + 1), nnz[order] / n, color="#0f766e", marker=".", linewidth=1.0, alpha=0.85, label="support fraction")
        ax2.set_ylabel("support fraction")
        ax2.spines["top"].set_visible(False)
    ax.set_title("Consensus rank-100 embedding dimension usage")
    ax.spines[["top", "right"]].set_visible(False)

    cv_rank = cv["block"]["argmin_rank"] if cv else None
    coh_rank = coh["estimate"]["rank"] if coh else None
    logged_long_p = _extract_logged_pstar(ROOT / "sandbox/macaque/rank_sweep_max_outer150/outputs/run.log")
    notes = [
        f"coherence rank={coh_rank}, CV argmin={cv_rank}, CV edge={cv['block'].get('edge_status') if cv else None}",
        f"old median train fraction from val entries={old_train_fraction:.4f}" if old_train_fraction is not None else "old train fraction unavailable",
        f"long partial median train fraction from val entries={long_train_fraction:.4f}" if long_train_fraction is not None else "long train fraction unavailable",
        f"long-run log p*={logged_long_p}" if logged_long_p is not None else "long-run log p* unavailable",
        f"long partial completed={len(long)}/20 fits; ranks present={sorted(int(x) for x in long['rank'].unique()) if not long.empty else []}",
    ]
    fig.text(0.01, 0.01, "\n".join(notes), fontsize=9, family="monospace", va="bottom")
    fig.suptitle("Macaque dimensionality diagnostics from existing artifacts", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.055, 1, 0.975))
    fig.savefig(OUTPUT_DIR / "macaque_diagnostics_overview.png", dpi=180, facecolor="white")
    fig.savefig(OUTPUT_DIR / "macaque_diagnostics_overview.pdf", facecolor="white")
    plt.close(fig)

    summary = {
        "paths": {
            "figure_png": str(OUTPUT_DIR / "macaque_diagnostics_overview.png"),
            "figure_pdf": str(OUTPUT_DIR / "macaque_diagnostics_overview.pdf"),
            "old_cv_fit_summary_csv": str(OUTPUT_DIR / "old_cv_fit_summary.csv"),
            "long_partial_fit_summary_csv": str(OUTPUT_DIR / "long_partial_fit_summary.csv"),
        },
        "current_default": {
            "n": n,
            "coherence_rank": coh_rank,
            "coherence_sampling_fraction": coh["estimate"].get("sampling_fraction") if coh else None,
            "cv_argmin": cv_rank,
            "cv_edge_status": cv["block"].get("edge_status") if cv else None,
            "cv_sampling_fraction": cv["block"].get("params", {}).get("sampling_fraction") if cv else None,
            "cv_srf_kwargs": cv["block"].get("params", {}).get("srf_kwargs") if cv else None,
        },
        "old_cv": {
            "n_fits": int(len(old)),
            "ranks": sorted(int(x) for x in old["rank"].unique()) if not old.empty else [],
            "converged_by_rank": old.groupby("rank")["final_converged"].mean().to_dict() if not old.empty else {},
            "median_val_entries": int(old["n_entries_val"].median()) if not old.empty else None,
            "estimated_train_fraction_from_val_entries": old_train_fraction,
        },
        "long_partial": {
            "n_fits": int(len(long)),
            "ranks": sorted(int(x) for x in long["rank"].unique()) if not long.empty else [],
            "converged_by_rank": long.groupby("rank")["final_converged"].mean().to_dict() if not long.empty else {},
            "median_val_entries": int(long["n_entries_val"].median()) if not long.empty else None,
            "estimated_train_fraction_from_val_entries": long_train_fraction,
            "logged_sampling_fraction_pstar": logged_long_p,
        },
        "preview_outer10": {
            "n_fits": int(len(preview)),
            "rows": preview[["rank", "val_mse", "n_iter", "max_outer", "max_inner"]].to_dict("records") if not preview.empty else [],
        },
        "consensus_rank100": consensus_summary,
        "embedding_stats": {
            "shape": embedding.get("shape"),
            "sparsity": embedding.get("sparsity"),
            "mass_min": float(np.min(embedding["mass"])) if embedding else None,
            "mass_median": float(np.median(embedding["mass"])) if embedding else None,
            "mass_max": float(np.max(embedding["mass"])) if embedding else None,
        },
        "candidate_causes": [
            "Old CV did not converge for ranks >=20: almost every fit hit max_outer=50.",
            "The partial long run is confounded with a different effective training fraction; its validation entries imply p_train about 0.66, while the current default CV uses about 0.593.",
            "The rank-100 consensus embedding was selected from the upper-edge CV result, so interpretability of that embedding is downstream of a suspect rank-selection curve.",
        ],
    }
    (OUTPUT_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {OUTPUT_DIR / 'macaque_diagnostics_overview.png'}")
    print(f"wrote {OUTPUT_DIR / 'diagnostic_summary.json'}")
    for note in notes:
        print(note)


if __name__ == "__main__":
    main()
