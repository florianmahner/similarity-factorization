"""Entropy-peak σ selection for the RBF kernel across datasets.

For each dataset, sweeps α = σ / (median pairwise distance) and computes
Shannon entropy of the off-diagonal kernel value distribution. The α* that
maximizes entropy is the "most informative" bandwidth (Yang & Oja-style
information-theoretic bandwidth selection).

The pairwise-distance information is encoded in the σ_scale=1.0 cached kernel
(K_ij = exp(-D_ij / (2 σ_med²))), so we avoid reloading raw features by
computing

    K(α) = K_1.0 ** (1 / α²).

Outputs (all in outputs/):
    per-dataset NPZ: entropy_<dataset>.npz
    per-dataset PNG: entropy_<dataset>.png
    combined PNG:    entropy_combined.png
    summary JSON:    summary.json

Run:
    poetry run python experiments/datasets/sigma_entropy/run.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "32")

import matplotlib.pyplot as plt
import numpy as np

from src.colors import CYCLE, INDIGO

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
CACHE = ROOT / "experiments/datasets/dimensionality/outputs/cache"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# (label, cache filename @ σ=1.0).  Only datasets with a σ=1.0 cache work —
# the entropy method needs the median-heuristic kernel to recover relative
# distances.
DATASETS = [
    ("clip_vit_l14", "clip_vit_l14_sigma1.0.npy"),
    ("nsd_subj01", "nsd_subj01_sigma1.0.npy"),
    ("things_macaque22k", "things_macaque22k_sigma1.0.npy"),
]

ALPHA_GRID = np.geomspace(0.05, 5.0, 80)
N_BINS = 128
N_SUBSAMPLE = 2_000_000
RNG_SEED = 0


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shannon_entropy(values: np.ndarray, n_bins: int = N_BINS) -> float:
    hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def sweep_one(sim_path: Path) -> dict:
    """Load σ=1.0 kernel, sweep α, return entropy curve + peak."""
    k1 = np.load(sim_path).astype(np.float32)
    n = k1.shape[0]

    rng = np.random.default_rng(RNG_SEED)
    i_idx, j_idx = np.triu_indices(n, k=1)
    total = len(i_idx)
    if N_SUBSAMPLE < total:
        sel = rng.choice(total, N_SUBSAMPLE, replace=False)
        i_idx, j_idx = i_idx[sel], j_idx[sel]
    k1_off = k1[i_idx, j_idx].astype(np.float64)
    k1_off = np.clip(k1_off, 1e-300, 1.0)
    log_k1 = np.log(k1_off)

    entropies = np.zeros(len(ALPHA_GRID))
    means = np.zeros(len(ALPHA_GRID))
    fracs_near_one = np.zeros(len(ALPHA_GRID))
    fracs_near_zero = np.zeros(len(ALPHA_GRID))
    for idx, alpha in enumerate(ALPHA_GRID):
        log_k_a = log_k1 / (alpha * alpha)
        k_a = np.exp(np.clip(log_k_a, -50.0, 0.0))
        entropies[idx] = shannon_entropy(k_a)
        means[idx] = float(k_a.mean())
        fracs_near_one[idx] = float((k_a > 0.95).mean())
        fracs_near_zero[idx] = float((k_a < 0.05).mean())

    peak = int(np.argmax(entropies))
    return {
        "n": int(n),
        "n_offdiag_samples": int(len(k1_off)),
        "alpha": ALPHA_GRID,
        "entropy_bits": entropies,
        "mean_kernel": means,
        "frac_near_one": fracs_near_one,
        "frac_near_zero": fracs_near_zero,
        "alpha_star": float(ALPHA_GRID[peak]),
        "peak_entropy": float(entropies[peak]),
        "mean_at_peak": float(means[peak]),
        "k1_off_median": float(np.median(k1_off)),
        "k1_off_min": float(k1_off.min()),
    }


def plot_one(label: str, res: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ax1.plot(res["alpha"], res["entropy_bits"], "-o", ms=2.4, color=INDIGO)
    ax1.axvline(res["alpha_star"], color="0.5", linestyle="--", lw=0.8)
    ax1.axvline(1.0, color="0.85", linestyle=":", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\alpha = \sigma / \sigma_{\mathrm{med}}$  (log)")
    ax1.set_ylabel("Shannon entropy of off-diag K (bits)")
    ax1.set_title(f"{label}  —  α* = {res['alpha_star']:.3f}", fontsize=9)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    ax2.plot(res["alpha"], res["mean_kernel"], "-", color=INDIGO, label="mean K")
    ax2.plot(res["alpha"], res["frac_near_zero"], "--", color="0.4", lw=1.0, label="frac < 0.05")
    ax2.plot(res["alpha"], res["frac_near_one"], ":", color="0.1", lw=1.0, label="frac > 0.95")
    ax2.axvline(res["alpha_star"], color="0.5", linestyle="--", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\alpha$  (log)")
    ax2.set_ylabel("statistic")
    ax2.set_title("kernel distribution diagnostics", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out_path = OUT / f"entropy_{label}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_combined(results: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (label, res) in enumerate(results.items()):
        color = CYCLE[i % len(CYCLE)]
        # Normalize entropy to its own max so all curves share a [0, 1] y-axis —
        # makes the peak locations visually comparable across n.
        h = res["entropy_bits"]
        h_norm = h / h.max()
        ax.plot(res["alpha"], h_norm, "-", color=color, lw=1.5,
                label=f"{label}  (n={res['n']}, α*={res['alpha_star']:.2f})")
        ax.axvline(res["alpha_star"], color=color, linestyle="--", lw=0.6, alpha=0.5)
    ax.axvline(1.0, color="0.85", linestyle=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha = \sigma / \sigma_{\mathrm{med}}$  (log)")
    ax.set_ylabel("Shannon entropy  /  max(H)")
    ax.set_title("RBF entropy-peak bandwidth across datasets", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path = OUT / "entropy_combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    results: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for label, fname in DATASETS:
        sim_path = CACHE / fname
        if not sim_path.exists():
            stamp(f"SKIP  {label}: {sim_path} not found")
            continue
        stamp(f"=== {label}  ({sim_path.name}) ===")
        t0 = time.time()
        res = sweep_one(sim_path)
        stamp(
            f"  n={res['n']}  off-diag samples={res['n_offdiag_samples']:,}  "
            f"sweep={time.time()-t0:.1f}s  "
            f"α*={res['alpha_star']:.3f}  H*={res['peak_entropy']:.3f} bits  "
            f"mean K at peak={res['mean_at_peak']:.3f}"
        )
        results[label] = res

        np.savez(
            OUT / f"entropy_{label}.npz",
            alpha=res["alpha"], entropy_bits=res["entropy_bits"],
            mean_kernel=res["mean_kernel"],
            frac_near_one=res["frac_near_one"],
            frac_near_zero=res["frac_near_zero"],
            alpha_star=res["alpha_star"],
            peak_entropy=res["peak_entropy"],
            n=res["n"],
        )
        plot_path = plot_one(label, res)
        stamp(f"  saved {plot_path.name}")

        summary[label] = {
            "n": res["n"],
            "alpha_star": res["alpha_star"],
            "peak_entropy_bits": res["peak_entropy"],
            "mean_kernel_at_peak": res["mean_at_peak"],
            "kernel_off_median_at_alpha1": res["k1_off_median"],
            "kernel_off_min_at_alpha1": res["k1_off_min"],
        }

    if results:
        path = plot_combined(results)
        stamp(f"saved combined plot {path.name}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    stamp(f"saved summary.json")
    stamp("\nα* summary:")
    for label, s in summary.items():
        stamp(f"  {label:24s}  α*={s['alpha_star']:.3f}  H*={s['peak_entropy_bits']:.3f} bits  (n={s['n']})")


if __name__ == "__main__":
    main()
