"""Spectral diagnostic for macaque from already-saved coherence outputs.

Reads existing ``coherence_estimate.json`` for macaque + comparison datasets
and plots three signals that are all independent of the failing CV:

1. **Spectrum** — top eigenvalues of S. Steep decay = low intrinsic rank.
2. **Leakage** — per-dim instability score from the coherence bootstrap.
   Small for signal-bearing dims, large where noise dominates. The k_est
   is where leakage's changepoint lives.
3. **Recovery loss vs sampling fraction** — fraction of spectral mass the
   bootstrap-recovered low-rank model fails to capture. Tells us how
   reconstructible the matrix is.

For each panel we overlay the coherence rank_estimate (k*) and the CV
argmin to make agreement / disagreement visible. The macaque story:
spectrum + leakage say rank ≈ 30–50, but the old CV (max_outer=50) says
rank=100 (edge). That mismatch is exactly what we expect when the CV
fits are undertrained.

Output: ``outputs/spectral_diagnostic.png`` — no compute, just plotting,
runs safely alongside the rank sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.colors import GRAY_LIGHT, INDIGO, ROSE, SAND, TEAL, WINE  # noqa: F401
from src.utils import get_output_dir

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
CV_DIR = ROOT / "experiments/datasets/dimensionality/outputs"

OUTPUT_DIR = get_output_dir()

DATASETS = [
    # (key, label, color)
    ("things_macaque22k", "Macaque IT (default σ)", WINE),
    ("things_macaque22k_tight", "Macaque IT (tight σ)", ROSE),
    ("nsd_subj01", "NSD subj01", INDIGO),
    ("things_behavior", "THINGS behavior", SAND),
    ("clip_vit_l14", "CLIP ViT-L/14", TEAL),
]


def _load(name: str) -> dict | None:
    p = CV_DIR / name / "coherence_estimate.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    est = payload["estimate"]
    cv_path = CV_DIR / name / "cross_validation.json"
    cv_argmin = None
    if cv_path.exists():
        try:
            cv = json.loads(cv_path.read_text())
            cv_argmin = cv["validations"]["5fold"].get("argmin_rank")
        except (KeyError, json.JSONDecodeError):
            cv_argmin = None
    return {
        "name": name,
        "n": payload.get("n"),
        "k_est": int(est["rank"]),
        "p_star": float(est["sampling_fraction"]),
        "eigenvalues": np.array(est.get("eigenvalues", []), dtype=float),
        "leakage": np.array(est.get("leakage", []), dtype=float),
        "sampling_grid": np.array(est.get("sampling_grid", []), dtype=float),
        "recovery_loss": np.array(est.get("recovery_loss_monotone", []), dtype=float),
        "cv_argmin": cv_argmin,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for key, label, color in DATASETS:
        d = _load(key)
        if d is None:
            print(f"[skip] {key}: no coherence_estimate.json")
            continue
        d["label"] = label
        d["color"] = color
        loaded.append(d)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # ---- Panel 1: spectrum scree (log y) ----
    ax = axes[0]
    for d in loaded:
        eigs = d["eigenvalues"]
        if len(eigs) == 0:
            continue
        x = np.arange(1, len(eigs) + 1)
        ax.plot(x, eigs, color=d["color"], linewidth=1.4, label=d["label"])
        if d["k_est"] <= len(eigs):
            ax.axvline(d["k_est"], color=d["color"], linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Spectrum of S (log–log)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- Panel 2: leakage profile ----
    ax = axes[1]
    for d in loaded:
        leak = d["leakage"]
        if len(leak) == 0:
            continue
        x = np.arange(1, len(leak) + 1)
        ax.plot(x, leak, color=d["color"], linewidth=1.4, label=d["label"])
        if d["k_est"] <= len(leak):
            ax.axvline(d["k_est"], color=d["color"], linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("dim index")
    ax.set_ylabel("leakage")
    ax.set_title("Per-dim leakage (k* = changepoint)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- Panel 3: recovery loss vs sampling fraction ----
    ax = axes[2]
    for d in loaded:
        rg = d["sampling_grid"]
        rl = d["recovery_loss"]
        if len(rg) == 0 or len(rl) == 0:
            continue
        ax.plot(rg, rl, color=d["color"], linewidth=1.4, marker="o", markersize=3, label=d["label"])
        ax.axvline(d["p_star"], color=d["color"], linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("sampling fraction p")
    ax.set_ylabel("recovery loss (monotone)")
    ax.set_title("Recovery loss vs p (p* = chosen)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- Annotation table at bottom ----
    rows_text = []
    rows_text.append(f"{'dataset':<25} {'n':>6} {'k_est':>6} {'p*':>5} {'CV argmin':>10}")
    rows_text.append("-" * 60)
    for d in loaded:
        am = d["cv_argmin"] if d["cv_argmin"] is not None else "—"
        rows_text.append(f"{d['label']:<25} {d['n']:>6} {d['k_est']:>6} {d['p_star']:>5.2f} {str(am):>10}")
    fig.text(
        0.02, -0.18, "\n".join(rows_text),
        family="monospace", fontsize=7.5, va="top", ha="left",
    )

    fig.suptitle(
        "Macaque vs comparison datasets — spectrum + leakage + recovery agree on low rank, only the old CV disagrees",
        fontsize=10, y=1.02,
    )

    fig.subplots_adjust(left=0.05, right=0.985, top=0.88, bottom=0.16, wspace=0.30)
    out = OUTPUT_DIR / "spectral_diagnostic.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")

    # ---- Also print the table as text ----
    print()
    print("\n".join(rows_text))


if __name__ == "__main__":
    main()
