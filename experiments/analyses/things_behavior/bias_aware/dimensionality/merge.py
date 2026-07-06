"""Merge partial_*.json + the original cross_validation.json into one CV file.

Re-derives argmin_rank and one_se_rank across the union of ranks. Sources are
all expected to share identical params (verified) — we just concatenate the
per-rank val_mse arrays.

Also writes a quick PDF/PNG of the merged CV curve next to the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SOURCES = [
    OUT / "cross_validation.json",      # original (10, 20, 25)
    OUT / "partial_precuneus.json",     # 34, 35
    OUT / "partial_prefrontal.json",    # 38, 39, 42, 43, 44, 45, 55, 70, 100, 150
    OUT / "partial_fusiform.json",      # 30, 31, 32, 33, 36, 37, 40, 41, 50, 60, 80, 120
]


def _one_se_rank(ranks: list[int], mean: np.ndarray, sem: np.ndarray, argmin_idx: int) -> int:
    threshold = float(mean[argmin_idx] + sem[argmin_idx])
    for r, m in zip(ranks, mean):
        if np.isfinite(m) and m <= threshold:
            return int(r)
    return int(ranks[argmin_idx])


def main() -> None:
    by_rank: dict[int, dict] = {}
    matrix_meta = None
    params = None
    for path in SOURCES:
        if not path.exists():
            print(f"  skipping (missing): {path}")
            continue
        p = json.loads(path.read_text())
        if matrix_meta is None:
            matrix_meta = p.get("matrix")
            params = p.get("params")
        for i, r in enumerate(p["ranks"]):
            by_rank[int(r)] = {
                "mean": float(p["val_mse_mean"][i]),
                "sem": float(p["val_mse_sem"][i]) if i < len(p.get("val_mse_sem", [])) else 0.0,
                "scores": list(p["scores"][str(r)]) if str(r) in p.get("scores", {}) else [],
            }
        print(f"  loaded {path.name}: ranks={p['ranks']}")

    ranks = sorted(by_rank.keys())
    mean = np.array([by_rank[r]["mean"] for r in ranks])
    sem = np.array([by_rank[r]["sem"] for r in ranks])
    counts = [len(by_rank[r]["scores"]) for r in ranks]

    argmin_idx = int(np.nanargmin(mean))
    argmin_rank = ranks[argmin_idx]
    one_se = _one_se_rank(ranks, mean, sem, argmin_idx)

    payload = {
        "matrix": matrix_meta,
        "params": params,
        "ranks": ranks,
        "val_mse_mean": mean.tolist(),
        "val_mse_sem": sem.tolist(),
        "val_mse_count": counts,
        "scores": {str(r): by_rank[r]["scores"] for r in ranks},
        "argmin_rank": int(argmin_rank),
        "one_se_rank": int(one_se),
        "status": "complete",
    }
    merged_path = OUT / "cross_validation.json"
    merged_path.write_text(json.dumps(payload, indent=2))
    print(f"Merged {len(ranks)} ranks -> {merged_path}")
    print(f"  argmin_rank = {argmin_rank}  (val_mse = {mean[argmin_idx]:.6e})")
    print(f"  one_se_rank = {one_se}")
    print(f"\n  rank   val_mse_mean +/- sem    count")
    for r, m, s, c in zip(ranks, mean, sem, counts):
        marker = " <- argmin" if r == argmin_rank else (" <- one_se" if r == one_se else "")
        print(f"  {r:4d}   {m:.4e} +/- {s:.2e}   {c}{marker}")

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(ranks, mean, yerr=sem, marker="o", markersize=4,
                linewidth=1.2, color="#332288", capsize=2)
    ax.axvline(argmin_rank, color="#882255", linestyle="--", linewidth=0.8,
               label=f"argmin = {argmin_rank}")
    if one_se != argmin_rank:
        ax.axvline(one_se, color="#117733", linestyle=":", linewidth=0.8,
                   label=f"one-SE = {one_se}")
    ax.set_xlabel("rank")
    ax.set_ylabel("Validation MSE")
    ax.set_title("THINGS-behav, bias-aware matrix (Fisher, λ=10) — 5-fold × 10-repeat CV")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "cv_curve.pdf")
    fig.savefig(OUT / "cv_curve.png", dpi=200)
    plt.close(fig)
    print(f"  wrote cv_curve.{{pdf,png}}")


if __name__ == "__main__":
    main()
