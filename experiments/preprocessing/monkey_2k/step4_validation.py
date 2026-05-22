"""Step 4: Validate preprocessed data in thingsprimate format.

Usage:
    poetry run python experiments/preprocessing/monkey_2k/step4_validation.py --recording F
    poetry run python experiments/preprocessing/monkey_2k/step4_validation.py --recording N1
    poetry run python experiments/preprocessing/monkey_2k/step4_validation.py --recording N2

Checks:
  1. Data shape and format matches thingsprimate
  2. Stimulus ordering matches canonical THINGS classes
  3. Image paths exist for all stimuli
  4. Reliability values are sensible
  5. No NaN/Inf values
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.monkey import get_things_classes, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROCESSED_DIR / "final"
THINGS_IMG_DIR = Path("/SSD/datasets/things/behav1854")


def check_data_format(data_dict: dict, recording: str) -> bool:
    """Check data matches thingsprimate format."""
    log.info(f"\n{'='*60}")
    log.info("1. DATA FORMAT (thingsprimate compatibility)")
    log.info(f"{'='*60}")

    # Check required keys
    required_keys = ["train_MUA", "reliab"]
    missing = [k for k in required_keys if k not in data_dict]
    if missing:
        log.error(f"Missing keys: {missing}")
        return False

    train_mua = data_dict["train_MUA"]
    reliab = data_dict["reliab"]

    log.info(f"Recording: {recording}")
    log.info(f"train_MUA: {train_mua.shape} (channels x stimuli)")
    log.info(f"reliab: {reliab.shape} (channels,)")

    # Check shapes are consistent
    n_channels = train_mua.shape[0]
    if reliab.shape[0] != n_channels:
        log.error(f"Channel mismatch: train_MUA has {n_channels}, reliab has {reliab.shape[0]}")
        return False

    # Check for NaN/Inf
    has_nan = np.isnan(train_mua).any()
    has_inf = np.isinf(train_mua).any()
    log.info(f"NaN values: {has_nan}")
    log.info(f"Inf values: {has_inf}")

    # Data statistics
    log.info(f"Data range: [{train_mua.min():.2f}, {train_mua.max():.2f}]")
    log.info(f"Data mean: {train_mua.mean():.4f}")

    return not has_nan and not has_inf


def check_stiminfo(stiminfo: pd.DataFrame) -> bool:
    """Check stiminfo CSV format."""
    log.info(f"\n{'='*60}")
    log.info("2. STIMINFO FORMAT")
    log.info(f"{'='*60}")

    if "exemplar" not in stiminfo.columns:
        log.error("Missing 'exemplar' column")
        return False

    stimuli = stiminfo["exemplar"].values
    canonical = get_things_classes()

    log.info(f"Stimuli in file: {len(stimuli)}")
    log.info(f"Canonical THINGS: {len(canonical)}")

    match = np.array_equal(stimuli, canonical)
    log.info(f"Exact match: {match}")

    if not match:
        diff = set(stimuli) ^ set(canonical)
        log.warning(f"Differences: {list(diff)[:10]}")

    log.info(f"First 5: {list(stimuli[:5])}")
    log.info(f"Last 5: {list(stimuli[-5:])}")

    return match


def check_image_paths(stimuli: np.ndarray) -> bool:
    """Check all stimulus images exist."""
    log.info(f"\n{'='*60}")
    log.info("3. IMAGE PATHS")
    log.info(f"{'='*60}")

    missing = []
    for obj in stimuli:
        img_path = THINGS_IMG_DIR / obj / f"{obj}_01b.jpg"
        if not img_path.exists():
            missing.append(obj)

    log.info(f"Total stimuli: {len(stimuli)}")
    log.info(f"Images found: {len(stimuli) - len(missing)}")
    log.info(f"Images missing: {len(missing)}")

    if missing:
        log.warning(f"Missing: {missing[:10]}")

    return len(missing) == 0


def check_reliability(reliab: np.ndarray, min_thresh: float = 0.3) -> bool:
    """Check reliability values."""
    log.info(f"\n{'='*60}")
    log.info("4. RELIABILITY")
    log.info(f"{'='*60}")

    log.info(f"Channels: {len(reliab)}")
    log.info(f"Range: [{reliab.min():.3f}, {reliab.max():.3f}]")
    log.info(f"Mean: {reliab.mean():.3f}")

    n_good = (reliab >= min_thresh).sum()
    log.info(f"Channels >= {min_thresh}: {n_good}/{len(reliab)}")

    valid = (reliab >= -0.1).all() and (reliab <= 1.1).all()
    log.info(f"Values in valid range: {valid}")

    return valid


def check_loader_compatibility(recording: str, roi: str) -> bool:
    """Test that the data can be loaded like thingsprimate."""
    log.info(f"\n{'='*60}")
    log.info("5. LOADER COMPATIBILITY")
    log.info(f"{'='*60}")

    npy_path = DATA_DIR / f"monkey{recording}_{roi}.npy"
    csv_path = DATA_DIR / f"monkey{recording}_{roi}_stiminfo.csv"

    # Load like thingsprimate does
    try:
        monkey_data = np.load(npy_path, allow_pickle=True).item()
        data = monkey_data["train_MUA"].astype("float32").T  # (n_stim, n_channels)
        reliab = monkey_data["reliab"]

        stiminfo = pd.read_csv(csv_path)
        stims = stiminfo["exemplar"].tolist()

        log.info(f"Loaded data: {data.shape} (stimuli x channels)")
        log.info(f"Loaded stimuli: {len(stims)}")

        if len(stims) != data.shape[0]:
            log.error(f"Mismatch: {len(stims)} stimuli but {data.shape[0]} rows")
            return False

        # Test reliability filtering
        min_reliab = 0.3
        reliab_mask = reliab >= min_reliab
        data_filtered = data[:, reliab_mask]
        log.info(f"After reliability filter (>={min_reliab}): {data_filtered.shape}")

        return True

    except Exception as e:
        log.error(f"Loader failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        required=True,
        choices=["N1", "N2", "F"],
    )
    parser.add_argument("--roi", default="it", choices=["it", "v1", "v4"])
    args = parser.parse_args()

    # Load data
    npy_path = DATA_DIR / f"monkey{args.recording}_{args.roi}.npy"
    csv_path = DATA_DIR / f"monkey{args.recording}_{args.roi}_stiminfo.csv"

    if not npy_path.exists():
        log.error(f"Data not found: {npy_path}")
        log.error("Run step3_average_stimuli.py first")
        return

    log.info(f"Loading: {npy_path}")
    data_dict = np.load(npy_path, allow_pickle=True).item()
    stiminfo = pd.read_csv(csv_path)

    # Run checks
    checks = {
        "data_format": check_data_format(data_dict, args.recording),
        "stiminfo": check_stiminfo(stiminfo),
        "image_paths": check_image_paths(stiminfo["exemplar"].values),
        "reliability": check_reliability(data_dict["reliab"]),
        "loader": check_loader_compatibility(args.recording, args.roi),
    }

    # Summary
    log.info(f"\n{'='*60}")
    log.info("VALIDATION SUMMARY")
    log.info(f"{'='*60}")

    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        log.info(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        log.info("\nAll checks passed!")
    else:
        log.error("\nSome checks failed. Fix issues before proceeding.")


if __name__ == "__main__":
    main()
