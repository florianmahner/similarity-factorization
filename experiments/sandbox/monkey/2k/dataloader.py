#!/usr/bin/env python3
"""Dataloader for THINGS-2k monkey data."""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "things-monkey" / "2k"
THINGS_IMG_DIR = Path("/SSD/datasets/things/behav1854")

ROI_SLICES = {
    "N": {"v1": (0, 512), "v4": (512, 768), "it": (768, 1024)},
    "F": {"v1": (0, 512), "it": (512, 832), "v4": (832, 1024)},
}
TIME_WINDOWS = {"v1": (25, 125), "v4": (50, 150), "it": (75, 175)}


@dataclass
class StimulusInfo:
    """Stimulus metadata from things2_imgs.mat."""
    stim_id: np.ndarray       # 1-indexed IDs (1-2000)
    classes: np.ndarray       # Class names
    things_path: np.ndarray   # Image filenames
    is_things: np.ndarray     # True if THINGS object, False if face


def load_stimulus_info() -> StimulusInfo:
    """Load stimulus metadata, marking THINGS vs faces."""
    stim_path = DATA_DIR / "things2_imgs.mat"

    with h5py.File(stim_path, "r") as f:
        n = f["test_imgs/class"].shape[0]
        classes = []
        paths = []

        for i in range(n):
            ref = f["test_imgs/class"][i, 0]
            name = "".join(chr(int(c)) for c in f[ref][()].flatten() if c < 128)
            classes.append(name)

            ref = f["test_imgs/things_path"][i, 0]
            path = "".join(chr(int(c)) for c in f[ref][()].flatten() if c < 128)
            paths.append(path)

    classes = np.array(classes)
    paths = np.array(paths)

    # Classify: faces are monkey01-73, person01-73
    is_things = np.array([
        not (c.startswith("monkey") and len(c) > 6) and not c.startswith("person")
        for c in classes
    ])

    return StimulusInfo(
        stim_id=np.arange(1, n + 1),
        classes=classes,
        things_path=paths,
        is_things=is_things,
    )


def get_things_image_path(class_name: str) -> Path:
    """Get full path to THINGS image for a class."""
    return THINGS_IMG_DIR / class_name / f"{class_name}_01b.jpg"


@dataclass
class NeuralRecording:
    """Neural data for one recording."""
    data: np.ndarray          # (n_trials, n_channels) time-averaged
    stim_ids: np.ndarray      # (n_trials,) stimulus IDs (1-indexed)
    rep_ids: np.ndarray       # (n_trials,) repetition number
    roi: str
    monkey: str
    session: str


def load_neural_data(
    monkey: str,
    session: str,
    roi: str = "it",
    things_only: bool = True,
) -> NeuralRecording:
    """Load time-averaged neural data for a recording.

    Args:
        monkey: "N" or "F"
        session: "1" or "2" for monkey N, ignored for F
        roi: "it", "v1", or "v4"
        things_only: If True, filter out face trials
    """
    if monkey == "N":
        folder = f"monkeyN_session{session}"
    else:
        folder = "monkeyF"
        session = "1"  # F only has one session

    rec_dir = DATA_DIR / folder

    # Check if we have time-resolved or already time-averaged
    time_resolved_path = rec_dir / "THINGS_normMUA_time_resolved.mat"
    time_averaged_path = rec_dir / "THINGS_normMUA_raw.mat"

    if time_averaged_path.exists():
        # Already time-averaged (session1)
        with h5py.File(time_averaged_path, "r") as f:
            data = np.array(f[f"data_{roi}"]).astype(np.float32)

        # Load labels from zi_list
        labels = pd.read_csv(rec_dir / "zi_list.csv", header=None)[0].values

        # Map labels to stim_ids using stimulus info
        stim_info = load_stimulus_info()
        class_to_id = {c: i for i, c in zip(stim_info.stim_id, stim_info.classes)}
        stim_ids = np.array([class_to_id.get(l, -1) for l in labels])

        # Rep IDs not available for time-averaged, estimate from order
        rep_ids = np.zeros(len(stim_ids), dtype=np.int32)
        for stim in np.unique(stim_ids):
            mask = stim_ids == stim
            rep_ids[mask] = np.arange(1, mask.sum() + 1)

    elif time_resolved_path.exists():
        # Time-resolved - need to average
        with h5py.File(time_resolved_path, "r") as f:
            tb = np.array(f["tb"]).flatten()
            stim_ids = np.array(f["ALLMAT"][1]).astype(np.int32)
            rep_ids = np.array(f["ALLMAT"][2]).astype(np.int32)

            # Get ROI slice and time windows
            ch_start, ch_end = ROI_SLICES[monkey][roi]
            t_start, t_end = TIME_WINDOWS[roi]

            # Time indices
            baseline_idx = np.where((tb >= -100) & (tb < 0))[0]
            response_idx = np.where((tb > t_start) & (tb <= t_end))[0]

            # Load and average (chunked to save memory)
            allmua = f["ALLMUA"]
            baseline = np.nanmean(allmua[baseline_idx[0]:baseline_idx[-1]+1, :, ch_start:ch_end], axis=0)
            response = np.nanmean(allmua[response_idx[0]:response_idx[-1]+1, :, ch_start:ch_end], axis=0)

        data = (response - baseline).astype(np.float32)

    else:
        raise FileNotFoundError(f"No data found in {rec_dir}")

    # Filter to THINGS only
    if things_only:
        stim_info = load_stimulus_info()
        things_ids = set(stim_info.stim_id[stim_info.is_things])
        mask = np.array([s in things_ids for s in stim_ids])
        data = data[mask]
        stim_ids = stim_ids[mask]
        rep_ids = rep_ids[mask]

    return NeuralRecording(
        data=data,
        stim_ids=stim_ids,
        rep_ids=rep_ids,
        roi=roi,
        monkey=monkey,
        session=session,
    )


if __name__ == "__main__":
    # Test stimulus info
    info = load_stimulus_info()
    print(f"Total stimuli: {len(info.classes)}")
    print(f"THINGS: {info.is_things.sum()}")
    print(f"Faces: {(~info.is_things).sum()}")

    # Test neural data loading (session1 only - it's small)
    print("\n--- Neural data test (MonkeyN session1, IT) ---")
    rec = load_neural_data("N", "1", roi="it", things_only=True)
    print(f"Data shape: {rec.data.shape}")
    print(f"Unique stimuli: {len(np.unique(rec.stim_ids))}")
    print(f"Reps per stim: {rec.rep_ids.max()}")

    # Map stim_id to class name
    id_to_class = {i: c for i, c in zip(info.stim_id, info.classes)}
    first_stim = rec.stim_ids[0]
    print(f"First trial: stim_id={first_stim} -> {id_to_class[first_stim]}")

    # Uncomment to test time-resolved (slow, loads 60GB file):
    # print("\n--- Neural data test (MonkeyN session2, IT) ---")
    # rec2 = load_neural_data("N", "2", roi="it", things_only=True)
    # print(f"Data shape: {rec2.data.shape}")
