#!/usr/bin/env python3
"""Combine MonkeyN recordings and compute split-half reliability."""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit, prange
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
N_SPLITS = 1000


@njit(parallel=True)
def fast_split_half(tensor, n_reps, n_splits, seed=42):
    """Blazingly fast split-half reliability with numba."""
    n_stim, max_reps, n_ch = tensor.shape
    split_corrs = np.zeros((n_splits, n_ch))

    for s in prange(n_splits):
        np.random.seed(seed + s)
        half1 = np.zeros((n_stim, n_ch))
        half2 = np.zeros((n_stim, n_ch))

        for i in range(n_stim):
            n = n_reps[i]
            idx = np.random.permutation(n)
            if n % 2 != 0:
                n = n - 1

            for ch in range(n_ch):
                sum1, sum2 = 0.0, 0.0
                for j in range(n // 2):
                    sum1 += tensor[i, idx[j], ch]
                    sum2 += tensor[i, idx[n // 2 + j], ch]
                half1[i, ch] = sum1 / (n // 2)
                half2[i, ch] = sum2 / (n // 2)

        for ch in range(n_ch):
            h1 = half1[:, ch]
            h2 = half2[:, ch]
            m1 = np.mean(h1)
            m2 = np.mean(h2)
            num, d1, d2 = 0.0, 0.0, 0.0
            for i in range(n_stim):
                x = h1[i] - m1
                y = h2[i] - m2
                num += x * y
                d1 += x * x
                d2 += y * y
            split_corrs[s, ch] = num / (np.sqrt(d1 * d2) + 1e-10)

    return split_corrs


def compute_reliability(tensor, n_reps, n_splits=1000):
    split_corrs = fast_split_half(tensor.astype(np.float64), n_reps, n_splits)
    z = np.arctanh(np.clip(split_corrs, -0.9999, 0.9999))
    avg_r = np.tanh(z.mean(axis=0))
    return 2 * avg_r / (1 + avg_r)


def load_dataset(path: Path, roi: str = "it"):
    with h5py.File(path / "THINGS_normMUA_raw.mat", "r") as f:
        data = np.array(f[f"data_{roi}"])
    zi_list = pd.read_csv(path / "zi_list.csv", header=None)[0].values

    # Remove trials with NaN
    valid = ~np.isnan(data).any(axis=1)
    if valid.sum() < len(valid):
        print(f"    Removing {(~valid).sum()} NaN trials")
    return data[valid], zi_list[valid]


def build_tensor(data, zi_list, classes):
    """Build tensor for specified classes in order."""
    n_ch = data.shape[1]
    reps_per_class = [np.sum(zi_list == c) for c in classes]
    max_reps = max(reps_per_class)

    tensor = np.zeros((len(classes), max_reps, n_ch), dtype=np.float64)
    n_reps = np.zeros(len(classes), dtype=np.int32)

    for i, c in enumerate(classes):
        trials = data[zi_list == c]
        n_reps[i] = len(trials)
        tensor[i, :n_reps[i], :] = trials

    return tensor, n_reps


def combine_tensors(t1, n1, t2, n2):
    n_stim, _, n_ch = t1.shape
    max_reps = n1.max() + n2.max()
    combined = np.zeros((n_stim, max_reps, n_ch), dtype=np.float64)

    for i in range(n_stim):
        combined[i, :n1[i], :] = t1[i, :n1[i], :]
        combined[i, n1[i]:n1[i]+n2[i], :] = t2[i, :n2[i], :]

    return combined, n1 + n2


def main():
    roi = "it"
    data_dir = Path(__file__).parent.parent.parent / "data" / "things-monkey" / "THINGS-2k"

    print("Loading data...")
    data_n, zi_n = load_dataset(data_dir / "monkeyN", roi)
    data_nnew, zi_nnew = load_dataset(data_dir / "monkeyNnew", roi)
    print(f"  MonkeyN: {data_n.shape}, MonkeyNnew: {data_nnew.shape}")

    # Find common classes
    classes_n = np.unique(zi_n)
    classes_nnew = np.unique(zi_nnew)
    common = np.array(sorted(set(classes_n) & set(classes_nnew)))
    print(f"  Common stimuli: {len(common)}")

    print("Building tensors...")
    tensor_n, reps_n = build_tensor(data_n, zi_n, common)
    tensor_nnew, reps_nnew = build_tensor(data_nnew, zi_nnew, common)
    print(f"  MonkeyN: {tensor_n.shape}, reps {reps_n.min()}-{reps_n.max()}")
    print(f"  MonkeyNnew: {tensor_nnew.shape}, reps {reps_nnew.min()}-{reps_nnew.max()}")

    print("Combining tensors...")
    tensor_combined, reps_combined = combine_tensors(tensor_n, reps_n, tensor_nnew, reps_nnew)
    print(f"  Combined: {tensor_combined.shape}, reps {reps_combined.min()}-{reps_combined.max()}")

    print(f"\nComputing reliability ({N_SPLITS} splits)...")
    print("  Compiling numba...")

    reliab_n = compute_reliability(tensor_n, reps_n, N_SPLITS)
    print(f"  MonkeyN: median={np.median(reliab_n):.3f}")

    reliab_nnew = compute_reliability(tensor_nnew, reps_nnew, N_SPLITS)
    print(f"  MonkeyNnew: median={np.median(reliab_nnew):.3f}")

    reliab_combined = compute_reliability(tensor_combined, reps_combined, N_SPLITS)
    print(f"  Combined: median={np.median(reliab_combined):.3f}, >{0.3}: {np.sum(reliab_combined > 0.3)}/256")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(reliab_n, bins=40, alpha=0.6, label=f"N (med={np.median(reliab_n):.2f})")
    axes[0].hist(reliab_nnew, bins=40, alpha=0.6, label=f"Nnew (med={np.median(reliab_nnew):.2f})")
    axes[0].set_xlabel("Reliability")
    axes[0].set_ylabel("Channels")
    axes[0].set_title("Individual")
    axes[0].legend()

    axes[1].hist(reliab_combined, bins=40, alpha=0.7)
    axes[1].axvline(0.3, color='r', ls='--', label='0.3')
    axes[1].set_xlabel("Reliability")
    axes[1].set_title(f"Combined (med={np.median(reliab_combined):.2f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reliability.png", dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'reliability.png'}")

    np.savez(OUTPUT_DIR / "reliability.npz",
             reliab_n=reliab_n, reliab_nnew=reliab_nnew, reliab_combined=reliab_combined)


if __name__ == "__main__":
    main()
