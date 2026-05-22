"""Replay the F-stat changepoint over the leakage profile.

Re-runs the same algorithm pysrf uses internally (`_changepoint`) but exposes
the F-stat at every candidate split, so we can see whether k_cut=45 for vgg16
is a sharp winner or just barely the argmax of a flat curve. If the F-stat is
flat near 45, then "the data does not strongly identify 45" — CV picking 90
isn't necessarily wrong.

Also tries a few alternative configs (different min_segment, different
high_band_quantile) to see how robust k_cut is.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_OUT = (
    PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def f_statistic(left, right):
    n_left, n_right = len(left), len(right)
    within = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
    pooled = within / max(n_left + n_right - 2, 1)
    if pooled <= 1e-12:
        return np.inf
    harmonic = n_left * n_right / (n_left + n_right)
    gap2 = (right.mean() - left.mean()) ** 2
    return harmonic * gap2 / pooled


def fstat_curve(leakage, min_segment=2):
    n = len(leakage)
    scores = np.full(n - 1, np.nan)
    for split in range(min_segment - 1, n - min_segment):
        scores[split] = f_statistic(leakage[: split + 1], leakage[split + 1 :])
    return scores


def main() -> None:
    for name in ["peterson_animals", "vgg16"]:
        payload = json.loads((EXP_OUT / name / "result.json").read_text())
        est = payload["estimate"]
        leakage = np.asarray(est["leakage"])
        rank = int(est["rank"])
        n_grid = len(leakage)

        log(f"\n=== {name}  (n={payload['n']}  k_cut={rank}  max_rank={n_grid}) ===")

        scores = fstat_curve(leakage, min_segment=2)
        argmax_split = int(np.nanargmax(scores))
        argmax_rank = argmax_split + 1  # _changepoint adds 1
        peak = scores[argmax_split]
        log(f"  argmax F-stat: split={argmax_split}  -> rank={argmax_rank}  "
            f"F={peak:.2f}")

        # How sharp is the peak? Compare to neighbours.
        log(f"  F-stat in window around k_cut:")
        lo = max(0, argmax_split - 5)
        hi = min(len(scores), argmax_split + 6)
        for s in range(lo, hi):
            mark = "  <- argmax" if s == argmax_split else ""
            log(f"    split={s:>4}  rank={s + 1:>4}  F={scores[s]:>10.2f}"
                f"{mark}")

        # Top-5 splits and their F values
        order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
        log(f"  top-10 splits by F-stat:")
        for s in order[:10]:
            log(f"    rank={s + 1:>4}  F={scores[s]:>10.2f}")

        # Try with tighter min_segment requirement (forces longer "noise" tail)
        for ms in [5, 10, 20]:
            if 2 * ms >= n_grid:
                continue
            sc = fstat_curve(leakage, min_segment=ms)
            r = int(np.nanargmax(sc)) + 1
            log(f"  min_segment={ms:>3}  -> k_cut={r}")


if __name__ == "__main__":
    main()
