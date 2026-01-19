"""Factorial Design: RSA vs SRF Power Comparison."""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from pysrf import SRF

# --- design ---


def create_design(n_levels: list[int]) -> tuple[np.ndarray, list[slice]]:
    """One-hot encode full factorial design: n_levels=[2,3] -> 6 items, 5 columns."""
    items = list(itertools.product(*[range(n) for n in n_levels]))
    slices, col = [], 0
    for n in n_levels:
        slices.append(slice(col, col + n))
        col += n

    x = np.zeros((len(items), col))
    for i, item in enumerate(items):
        for f, level in enumerate(item):
            x[i, slices[f].start + level] = 1.0
    return x, slices


def apply_weights(
    x: np.ndarray, slices: list[slice], weights: list[float]
) -> np.ndarray:
    return np.hstack([x[:, sl] * w for sl, w in zip(slices, weights)])


# --- noise ---


def add_column_noise(x: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Replace (1-snr) fraction of columns with scaled random noise."""
    if snr >= 1:
        return x.copy()
    if snr <= 0:
        return np.abs(rng.standard_normal(x.shape))

    n_signal = max(1, round(snr * x.shape[1]))
    order = rng.permutation(x.shape[1])

    out = np.zeros_like(x)
    out[:, order[:n_signal]] = x[:, order[:n_signal]]

    n_noise = x.shape[1] - n_signal
    if n_noise > 0:
        noise = np.abs(rng.standard_normal((x.shape[0], n_noise)))
        out[:, order[n_signal:]] = noise * np.mean(np.std(x, axis=0)) / np.std(noise)
    return out


def add_elementwise_noise(
    x: np.ndarray, snr: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise to achieve target SNR via binary search on scale."""
    if snr >= 1:
        return x.copy()
    if snr <= 0:
        return np.clip(rng.normal(0, 1, x.shape), 0, None)

    target = snr / (1 - snr)
    noise = rng.normal(0, 1, x.shape)
    lo, hi = 0.0, 100.0

    for _ in range(100):
        scale = (lo + hi) / 2
        noisy = np.clip(x + scale * noise, 0, None)
        ratio = np.var(x) / np.var(noisy - x) if np.var(noisy - x) > 0 else np.inf
        lo, hi = (scale, hi) if ratio > target else (lo, scale)
        if abs(ratio - target) < 1e-4 * target:
            break

    return np.clip(x + scale * noise, 0, None)


def add_noise(
    x: np.ndarray, snr: float, model: str, rng: np.random.Generator
) -> np.ndarray:
    return (
        add_column_noise(x, snr, rng)
        if model == "column"
        else add_elementwise_noise(x, snr, rng)
    )


# --- similarity ---


def compute_similarity(x: np.ndarray) -> np.ndarray:
    return x @ x.T


def fit_srf(s: np.ndarray, rank: int, seed: int) -> np.ndarray:
    return SRF(rank=rank, random_state=seed, verbose=False).fit_transform(
        np.maximum(s, 0)
    )


# --- tests ---


def corr_triu(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = squareform(a, checks=False), squareform(b, checks=False)
    return (
        np.corrcoef(va, vb)[0, 1] if np.std(va) > 1e-10 and np.std(vb) > 1e-10 else 0.0
    )


def mantel_test(
    s_data: np.ndarray, s_hyp: np.ndarray, n_perm: int, rng: np.random.Generator
) -> float:
    """Permutation test for correlation between two similarity matrices."""
    r_obs = corr_triu(s_data, s_hyp)
    r_null = [
        corr_triu(s_data, s_hyp[p := rng.permutation(len(s_hyp))][:, p])
        for _ in range(n_perm)
    ]
    return (sum(r >= r_obs for r in r_null) + 1) / (n_perm + 1)


def hungarian_match(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Find optimal column permutation of w to match x via correlation."""
    k = x.shape[1]
    cost = np.array(
        [
            [-(abs(np.corrcoef(x[:, i], w[:, j])[0, 1]) or 0) for j in range(k)]
            for i in range(k)
        ]
    )
    return linear_sum_assignment(cost)[1]


def loo_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Predict x from w using LOO Hungarian matching on held-out items."""
    n = len(x)
    x_pred = np.zeros_like(x)
    for i in range(n):
        mask = np.arange(n) != i
        perm = hungarian_match(w[mask], x[mask])
        x_pred[i] = w[i, perm]
    return x_pred


def loo_test(
    w: np.ndarray, x: np.ndarray, sl: slice, n_perm: int, rng: np.random.Generator
) -> float:
    """Permutation test for LOO prediction; null redoes matching each time."""
    x_pred = loo_predict(w, x)
    r_obs = np.corrcoef(x_pred[:, sl].ravel(), x[:, sl].ravel())[0, 1]

    r_null = []
    for _ in range(n_perm):
        x_perm = x[rng.permutation(len(x))]
        x_pred_perm = loo_predict(w, x_perm)
        r_null.append(
            np.corrcoef(x_pred_perm[:, sl].ravel(), x_perm[:, sl].ravel())[0, 1]
        )

    return (sum(r >= r_obs for r in r_null) + 1) / (n_perm + 1)


# --- simulation ---


def run_single(
    x: np.ndarray, slices: list[slice], snr: float, noise: str, n_perm: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)

    x_noisy = add_noise(x, snr, noise, rng)
    s = compute_similarity(x_noisy)
    w = fit_srf(s, x.shape[1], seed)
    s_denoised = compute_similarity(w)

    results = {"rsa": [], "denoise": [], "loo": []}
    for f, sl in enumerate(slices):
        s_hyp = compute_similarity(x[:, sl])
        results["rsa"].append(
            mantel_test(s, s_hyp, n_perm, np.random.default_rng(seed + 1000 * f)) < 0.05
        )
        results["denoise"].append(
            mantel_test(
                s_denoised, s_hyp, n_perm, np.random.default_rng(seed + 2000 * f)
            )
            < 0.05
        )
        results["loo"].append(
            loo_test(w, x, sl, n_perm, np.random.default_rng(seed + 3000 * f)) < 0.05
        )
    return results


def run_simulation(
    n_levels: list[int],
    weights: list[float],
    snr_levels: np.ndarray,
    noise: str,
    n_reps: int,
    n_perm: int,
) -> dict:
    x, slices = create_design(n_levels)
    x_weighted = apply_weights(x, slices, weights)

    power = {
        m: np.zeros((len(snr_levels), len(weights))) for m in ["rsa", "denoise", "loo"]
    }

    for i, snr in enumerate(snr_levels):
        results = Parallel(n_jobs=-1)(
            delayed(run_single)(
                x_weighted, slices, snr, noise, n_perm, rep * 10000 + i * 100000
            )
            for rep in range(n_reps)
        )
        for m in power:
            power[m][i] = np.mean(
                [[r[m][f] for f in range(len(weights))] for r in results], axis=0
            )

    return {"power": power, "snr_levels": snr_levels, "weights": weights}


# --- plotting ---


def plot_power(res_col: dict, res_elem: dict, weights: list[float], path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    colors = ["#E24A33", "#348ABD", "#988ED5", "#777777"]
    methods = [("rsa", "RSA"), ("denoise", "SRF-Denoise"), ("loo", "SRF-LOO")]

    for row, (res, noise_label) in enumerate(
        [(res_col, "Column"), (res_elem, "Element-wise")]
    ):
        for col, (key, title) in enumerate(methods):
            ax = axes[row, col]
            for f, w in enumerate(weights):
                ax.plot(
                    res["snr_levels"],
                    res["power"][key][:, f] * 100,
                    "-o",
                    color=colors[f],
                    lw=1.5,
                    ms=4,
                    label=f"F{f+1} (w={w})",
                )

            ax.axhline(5, color="gray", ls="--", lw=1, alpha=0.5)
            ax.set(ylim=(-5, 105), xlim=(-0.05, 1.05))
            ax.spines[["top", "right"]].set_visible(False)

            if row == 1:
                ax.set_xlabel("SNR")
            if col == 0:
                ax.set_ylabel(f"{noise_label}\nPower (%)")
            if row == 0:
                ax.set_title(title)
            if row == 0 and col == 2:
                ax.legend(fontsize=6, loc="lower right")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- main ---


def main():
    out = (
        Path(__file__).parent
        / "outputs"
        / "factorial_new"
        / datetime.now().strftime("%y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    n_levels = [2, 3, 2, 4]
    snr_levels = np.linspace(0, 1, 6)

    experiments = [
        ("equal", [1.0, 1.0, 1.0, 1.0]),
        ("differential", [1.0, 0.7, 0.6, 0.8]),
    ]

    for name, weights in experiments:
        print(f"\n=== {name}: {weights} ===")
        results = {
            noise: run_simulation(
                n_levels, weights, snr_levels, noise, n_reps=200, n_perm=500
            )
            for noise in ["column", "elementwise"]
            if print(f"  {noise}...") or True
        }

        with open(out / f"{name}.json", "w") as f:
            json.dump(
                {
                    k: {
                        kk: vv.tolist() if hasattr(vv, "tolist") else vv
                        for kk, vv in v.items()
                    }
                    for k, v in results.items()
                },
                f,
            )

        plot_power(
            results["column"], results["elementwise"], weights, out / f"{name}.pdf"
        )

    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
