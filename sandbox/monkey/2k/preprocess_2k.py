#!/usr/bin/env python3
"""Preprocess THINGS-2k monkey MUA data. Time windows from TVSD norm_MUA.m."""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROI_SLICES = {
    "N": {"v1": (0, 512), "v4": (512, 768), "it": (768, 1024)},
    "F": {"v1": (0, 512), "it": (512, 832), "v4": (832, 1024)},
}
TIME_WINDOWS = {"v1": (25, 125), "v4": (50, 150), "it": (75, 175)}


def load_raw(mat_path: Path, monkey: str, roi: str):
    ch_start, ch_end = ROI_SLICES[monkey][roi]
    t_start, t_end = TIME_WINDOWS[roi]

    with h5py.File(mat_path, "r") as f:
        tb = np.array(f["tb"]).flatten()
        allmat = np.array(f["ALLMAT"])

        baseline_idx = np.where((tb >= -100) & (tb < 0))[0]
        response_idx = np.where((tb > t_start) & (tb <= t_end))[0]

        baseline = np.array(
            f["ALLMUA"][baseline_idx[0] : baseline_idx[-1] + 1, :, ch_start:ch_end]
        )
        response = np.array(
            f["ALLMUA"][response_idx[0] : response_idx[-1] + 1, :, ch_start:ch_end]
        )

    baseline_avg = np.nanmean(baseline, axis=0)
    response_avg = np.nanmean(response, axis=0).astype(np.float32)
    data = response_avg - baseline_avg.astype(np.float32)

    return {
        "data": data,
        "stim_ids": allmat[1].astype(np.int32),
        "rep_ids": allmat[2].astype(np.int32),
        "tb": tb,
        "baseline": baseline_avg,
        "response": response_avg,
    }


def average_reps(trial_data: dict):
    data, stim_ids = trial_data["data"], trial_data["stim_ids"]
    unique_stim = np.unique(stim_ids)
    n_stim, n_ch = len(unique_stim), data.shape[1]
    max_reps = max(np.sum(stim_ids == s) for s in unique_stim)

    data_avg = np.zeros((n_stim, n_ch), dtype=np.float32)
    data_reps = np.full((n_stim, n_ch, max_reps), np.nan, dtype=np.float32)
    n_reps = np.zeros(n_stim, dtype=np.int32)

    for i, stim in enumerate(unique_stim):
        reps = data[stim_ids == stim]
        n_reps[i] = len(reps)
        data_avg[i] = np.nanmean(reps, axis=0)
        data_reps[i, :, : n_reps[i]] = reps.T

    return {
        "data": data_avg,
        "data_reps": data_reps,
        "stim_ids": unique_stim,
        "n_reps": n_reps,
    }


def compute_reliability(data_reps: np.ndarray, n_splits: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_stim, n_ch, _ = data_reps.shape
    reliab = np.zeros((n_ch, n_splits), dtype=np.float32)

    for split in range(n_splits):
        half1 = np.zeros((n_stim, n_ch), dtype=np.float32)
        half2 = np.zeros((n_stim, n_ch), dtype=np.float32)

        for s in range(n_stim):
            valid = np.where(~np.isnan(data_reps[s, 0, :]))[0]
            if len(valid) >= 2:
                rng.shuffle(valid)
                h1_idx, h2_idx = valid[: len(valid) // 2], valid[len(valid) // 2 :]
                half1[s] = np.nanmean(data_reps[s][:, h1_idx], axis=1)
                half2[s] = np.nanmean(data_reps[s][:, h2_idx], axis=1)
            else:
                half1[s], half2[s] = np.nan, np.nan

        for ch in range(n_ch):
            valid = ~np.isnan(half1[:, ch]) & ~np.isnan(half2[:, ch])
            if valid.sum() > 10:
                reliab[ch, split] = np.corrcoef(half1[valid, ch], half2[valid, ch])[
                    0, 1
                ]

    return np.nanmean(reliab, axis=1)


def split_categories(
    data: np.ndarray, data_reps: np.ndarray, reliab: np.ndarray, classes: list
):
    things_idx, monkey_idx, human_idx = [], [], []
    for i, c in enumerate(classes):
        cl = c.lower()
        if cl.startswith("monkey") and cl[6:].isdigit():
            monkey_idx.append(i)
        elif cl.startswith("person") and cl[6:].replace("jpg", "").isdigit():
            human_idx.append(i)
        else:
            things_idx.append(i)

    def subset(idx):
        idx = np.array(idx)
        return {"data": data[idx], "reliab": reliab, "classes": np.array(classes)[idx]}

    return {
        "things": subset(things_idx),
        "monkey_faces": subset(monkey_idx),
        "human_faces": subset(human_idx),
    }


def load_stim_info(stim_path: Path):
    with h5py.File(stim_path, "r") as f:
        n = f["test_imgs/class"].shape[0]
        classes = []
        for i in range(n):
            ref = f["test_imgs/class"][i, 0]
            classes.append(
                "".join(chr(int(c)) for c in f[ref][()].flatten() if c < 128)
            )
    return classes


# === Plotting functions ===


def plot_step1(trial_data: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Baseline vs response distribution
    ax = axes[0]
    ax.hist(
        trial_data["baseline"].flatten(),
        bins=100,
        alpha=0.6,
        label="Baseline",
        density=True,
    )
    ax.hist(
        trial_data["response"].flatten(),
        bins=100,
        alpha=0.6,
        label="Response",
        density=True,
    )
    ax.set_xlabel("MUA")
    ax.set_ylabel("Density")
    ax.set_title("Baseline vs Response")
    ax.legend()

    # After baseline subtraction
    ax = axes[1]
    ax.hist(trial_data["data"].flatten(), bins=100, alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Baseline-subtracted MUA")
    ax.set_ylabel("Count")
    ax.set_title("After baseline subtraction")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_step2(stim_data: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Distribution of stimulus means
    ax = axes[0]
    stim_means = np.nanmean(stim_data["data"], axis=1)
    ax.hist(stim_means, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mean response across channels")
    ax.set_ylabel("Stimulus count")
    ax.set_title(f"Stimulus means (n={len(stim_means)})")

    # Reps per stimulus
    ax = axes[1]
    ax.hist(
        stim_data["n_reps"],
        bins=range(stim_data["n_reps"].min(), stim_data["n_reps"].max() + 2),
        edgecolor="black",
        alpha=0.7,
        align="left",
    )
    ax.set_xlabel("Number of repetitions")
    ax.set_ylabel("Stimulus count")
    ax.set_title("Repetitions per stimulus")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_step3(reliab: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(reliab[~np.isnan(reliab)], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        0.3,
        color="red",
        linestyle="--",
        lw=2,
        label=f"threshold=0.3 (n={np.sum(reliab>0.3)})",
    )
    ax.axvline(
        np.nanmedian(reliab),
        color="blue",
        linestyle="--",
        lw=2,
        label=f"median={np.nanmedian(reliab):.2f}",
    )
    ax.set_xlabel("Split-half reliability")
    ax.set_ylabel("Channel count")
    ax.set_title(f"Reliability distribution (n={len(reliab)} channels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_step4(split_data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [
        len(split_data[k]["classes"]) for k in ["things", "monkey_faces", "human_faces"]
    ]
    labels = ["THINGS", "Monkey faces", "Human faces"]
    ax.bar(labels, counts, edgecolor="black")
    ax.set_ylabel("Number of stimuli")
    ax.set_title("Category split")
    for i, c in enumerate(counts):
        ax.text(i, c + 10, str(c), ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey", type=str, choices=["F", "N"], required=True)
    parser.add_argument("--roi", type=str, choices=["v1", "v4", "it"], default="it")
    args = parser.parse_args()

    monkey, roi = args.monkey, args.roi
    raw_dir = (
        Path(__file__).parent.parent.parent.parent / "data" / "things-monkey" / "2k"
    )
    out_dir = Path(__file__).parent / "outputs"
    plot_dir = out_dir / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    mat_path = raw_dir / ("monkeyF" if monkey == "F" else "monkeyN_session2")
    mat_path = mat_path / "THINGS_normMUA_time_resolved.mat"
    stim_path = raw_dir / "things2_imgs.mat"

    prefix = f"monkey{monkey}_{roi}"

    # Step 1: Time-average with baseline subtraction
    print(f"\n=== Step 1: Load and time-average ({monkey}, {roi.upper()}) ===")
    trial_data = load_raw(mat_path, monkey, roi)
    print(
        f"  Trials: {trial_data['data'].shape[0]}, Channels: {trial_data['data'].shape[1]}"
    )
    print(f"  Range: [{trial_data['data'].min():.2f}, {trial_data['data'].max():.2f}]")
    plot_step1(trial_data, plot_dir / f"{prefix}_step1.png")

    # Step 2: Average repetitions
    print(f"\n=== Step 2: Average repetitions ===")
    stim_data = average_reps(trial_data)
    print(
        f"  Stimuli: {stim_data['data'].shape[0]}, Reps: {stim_data['n_reps'].min()}-{stim_data['n_reps'].max()}"
    )
    plot_step2(stim_data, plot_dir / f"{prefix}_step2.png")

    # Step 3: Compute reliability
    print(f"\n=== Step 3: Compute reliability ===")
    reliab = compute_reliability(stim_data["data_reps"])
    print(f"  Mean: {np.nanmean(reliab):.3f}, Median: {np.nanmedian(reliab):.3f}")
    print(f"  Channels > 0.3: {np.sum(reliab > 0.3)}/{len(reliab)}")
    plot_step3(reliab, plot_dir / f"{prefix}_step3.png")

    # Step 4: Split by category
    print(f"\n=== Step 4: Split by category ===")
    classes = load_stim_info(stim_path)
    classes = [classes[s - 1] for s in stim_data["stim_ids"]]
    split_data = split_categories(
        stim_data["data"], stim_data["data_reps"], reliab, classes
    )
    for cat, d in split_data.items():
        print(f"  {cat}: {len(d['classes'])} stimuli")
    plot_step4(split_data, plot_dir / f"{prefix}_step4.png")

    # Save final THINGS-only output
    out_path = out_dir / f"{prefix}_things.npz"
    np.savez_compressed(out_path, **split_data["things"])
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
