"""Step 1: just look at the spectrum / leakage / recovery for vgg16.

No compute — reads the existing experiment JSON and plots
    - top eigenvalues (log-y)
    - leakage profile (per-dim instability)
    - recovery loss vs sampling fraction

This tells us whether k_cut=45 is even plausible from the spectrum, and
whether the leakage profile has a clean elbow at 45 or a flat ramp.

Run:
    poetry run python sandbox/dimensionality/vgg16_spectrum/run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.utils import get_output_dir
from src.utils.figure_theme import despine
from src.colors import GRAY, INDIGO, ROSE, TEAL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()

DATASETS = [
    ("peterson_animals", "U-shape works (k_cut=9, argmin=11)"),
    ("vgg16",            "U-shape broken (k_cut=45, argmin=90)"),
]
EXP_OUT = (
    PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_payload(name: str) -> dict:
    return json.loads((EXP_OUT / name / "result.json").read_text())


def plot_dataset(payload: dict, label: str, axes) -> None:
    est = payload["estimate"]
    rank = int(est["rank"])
    p_star = float(est["sampling_fraction"])
    eig = np.asarray(est["eigenvalues"])
    leakage = np.asarray(est["leakage"])
    grid = np.asarray(est["sampling_grid"])
    raw = np.asarray(est["recovery_loss_raw"])
    monotone = np.asarray(est["recovery_loss_monotone"])

    ax = axes[0]
    ax.plot(np.arange(1, len(eig) + 1), eig, color=INDIGO)
    ax.axvline(rank, color=ROSE, linestyle="--", label=f"k_cut={rank}")
    ax.set_yscale("log")
    ax.set_xlabel("rank index")
    ax.set_ylabel("eigenvalue (log)")
    ax.set_title(f"{label} — spectrum")
    ax.legend(frameon=False)
    despine(ax)

    ax = axes[1]
    ax.plot(np.arange(1, len(leakage) + 1), leakage, color=INDIGO)
    ax.axvline(rank, color=ROSE, linestyle="--", label=f"k_cut={rank}")
    ax.set_xlabel("rank index")
    ax.set_ylabel("leakage")
    ax.set_title(f"{label} — leakage")
    ax.legend(frameon=False)
    despine(ax)

    ax = axes[2]
    ax.plot(grid, raw, color=GRAY, marker=".", label="raw")
    ax.plot(grid, monotone, color=INDIGO, marker="o", label="monotone")
    ax.axhline(est["params"].get("recovery_tolerance", 0.1), color=ROSE,
               linestyle="--", label="tolerance")
    ax.axvline(p_star, color=TEAL, linestyle=":", label=f"p*={p_star:.3f}")
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("recovery loss")
    ax.set_title(f"{label} — recovery")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)


def main() -> None:
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(15, 4 * len(DATASETS)))
    if len(DATASETS) == 1:
        axes = axes[None, :]

    for row, (name, note) in enumerate(DATASETS):
        log(f"loading {name} ({note})")
        payload = load_payload(name)
        n = payload.get("n", "?")
        est = payload["estimate"]
        eig = np.asarray(est["eigenvalues"])
        rank = int(est["rank"])
        log(f"  n={n}  k_cut={rank}  p*={est['sampling_fraction']:.3f}")
        log(f"  top eigs: {eig[:5]}")
        log(f"  eigs around k_cut: {eig[max(0, rank-3):rank+3]}")
        log(f"  ratio eig[k_cut] / eig[0]: {eig[rank-1] / eig[0]:.3e}")
        log(f"  ratio eig[k_cut] / eig[k_cut*2]: "
            f"{eig[rank-1] / eig[min(2*rank-1, len(eig)-1)]:.3e}")
        plot_dataset(payload, f"{name} (n={n})", axes[row])

    fig.tight_layout()
    out = OUTPUT_DIR / "spectrum.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
