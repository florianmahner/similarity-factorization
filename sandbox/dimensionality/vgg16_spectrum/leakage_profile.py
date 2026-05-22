"""Print the leakage profile + cumulative variance explained for vgg16 vs peterson_animals.

If vgg16's leakage profile has no clean elbow at 45 (just a soft ramp), and the
cumulative variance keeps climbing well past 45, then there is no "true rank" —
CV will keep improving with more dimensions because there is still real signal
above the changepoint pysrf picks.
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

DATASETS = ["peterson_animals", "vgg16"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    for name in DATASETS:
        payload = json.loads((EXP_OUT / name / "result.json").read_text())
        est = payload["estimate"]
        rank = int(est["rank"])
        eig = np.asarray(est["eigenvalues"])
        leakage = np.asarray(est["leakage"])
        n = payload.get("n", "?")

        log(f"\n=== {name}  (n={n}  k_cut={rank}) ===")

        # cumulative variance explained by top-k (treating eig as variance)
        cum = np.cumsum(eig) / np.sum(eig)
        for k in [rank // 2, rank, int(1.5 * rank), 2 * rank, 3 * rank]:
            if k < len(eig):
                log(f"  cum_var@k={k:>4}: {cum[k - 1]:.4f}")

        # leakage around k_cut
        lo = max(0, rank - 5)
        hi = min(len(leakage), rank + 6)
        log(f"  leakage around k_cut={rank}:")
        for i in range(lo, hi):
            mark = "  <- k_cut" if (i + 1) == rank else ""
            log(f"    idx={i + 1:>4}  leakage={leakage[i]:.4f}{mark}")

        # leakage at wider range
        log(f"  leakage at strided idx:")
        for i in [1, 5, 10, rank // 2, rank, int(1.5 * rank), 2 * rank, 3 * rank, len(leakage)]:
            if 1 <= i <= len(leakage):
                log(f"    idx={i:>4}  leakage={leakage[i - 1]:.4f}")


if __name__ == "__main__":
    main()
