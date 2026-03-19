#!/usr/bin/env python3
"""Validate THINGS-2k monkey data preprocessing pipeline."""

import numpy as np
from pathlib import Path
from preprocessing import (
    load_recording, remove_nan_trials, filter_to_stimuli, get_common_stimuli,
    zscore_channels, average_per_stimulus, combine_recordings,
    check_zscore, check_stimulus_alignment, check_no_nan,
    summarize_recording, summarize_averaged
)
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "things-monkey" / "2k"


def main():
    print("=" * 60)
    print("THINGS-2k Monkey Data Validation")
    print("=" * 60)

    # === STEP 1: Load raw data ===
    print("\n[1] LOADING RAW DATA")
    rec_n = load_recording(DATA_DIR / "monkeyN_session1", name="MonkeyN_session1")
    rec_nnew = load_recording(DATA_DIR / "monkeyN_session2", name="MonkeyN_session2")

    summarize_recording(rec_n)
    summarize_recording(rec_nnew)

    # === STEP 2: Remove NaN trials ===
    print("\n[2] REMOVING NaN TRIALS")
    rec_n = remove_nan_trials(rec_n)
    rec_nnew = remove_nan_trials(rec_nnew)

    # === STEP 3: Find common stimuli ===
    print("\n[3] FINDING COMMON STIMULI")
    common = get_common_stimuli(rec_n, rec_nnew)
    print(f"  MonkeyN stimuli: {rec_n.n_stimuli}")
    print(f"  MonkeyNnew stimuli: {rec_nnew.n_stimuli}")
    print(f"  Common stimuli: {len(common)}")

    extra_nnew = set(np.unique(rec_nnew.labels)) - set(common)
    if extra_nnew:
        print(f"  Extra in MonkeyNnew (will be removed): {len(extra_nnew)}")

    # === STEP 4: Filter to common stimuli BEFORE z-scoring ===
    print("\n[4] FILTERING TO COMMON STIMULI")
    rec_n = filter_to_stimuli(rec_n, common)
    rec_nnew = filter_to_stimuli(rec_nnew, common)

    # Verify
    assert rec_n.n_stimuli == len(common), "MonkeyN stimulus count mismatch"
    assert rec_nnew.n_stimuli == len(common), "MonkeyNnew stimulus count mismatch"
    print(f"  ✓ Both recordings now have {len(common)} stimuli")

    # === STEP 5: Z-score each recording ===
    print("\n[5] Z-SCORING (per channel)")
    rec_n_z = zscore_channels(rec_n)
    rec_nnew_z = zscore_channels(rec_nnew)

    # Verify z-scoring
    print("  Checking z-score quality...")
    ok_n = check_zscore(rec_n_z.data, rec_n_z.name)
    ok_nnew = check_zscore(rec_nnew_z.data, rec_nnew_z.name)
    if ok_n and ok_nnew:
        print("  ✓ Z-scoring correct (mean=0, std=1 per channel)")

    # === STEP 6: Average per stimulus ===
    print("\n[6] AVERAGING PER STIMULUS")
    X_n, stimuli, reps_n = average_per_stimulus(rec_n_z, common)
    X_nnew, _, reps_nnew = average_per_stimulus(rec_nnew_z, common)

    summarize_averaged(X_n, "MonkeyN")
    summarize_averaged(X_nnew, "MonkeyNnew")

    # Check means are now close to zero
    print("\n  Checking averaged data means...")
    mean_n = X_n.mean()
    mean_nnew = X_nnew.mean()
    print(f"  MonkeyN overall mean: {mean_n:.6f}")
    print(f"  MonkeyNnew overall mean: {mean_nnew:.6f}")
    if abs(mean_n) < 0.01 and abs(mean_nnew) < 0.01:
        print("  ✓ Both means close to zero")
    else:
        print("  ⚠ Warning: means not close to zero!")

    # === STEP 7: Check alignment between recordings ===
    print("\n[7] CHECKING RECORDING ALIGNMENT")
    r = check_stimulus_alignment(X_n, X_nnew, common, "MonkeyN", "MonkeyNnew")

    if r > 0.5:
        print(f"  ✓ Good alignment (r={r:.3f})")
    elif r > 0.3:
        print(f"  ⚠ Moderate alignment (r={r:.3f})")
    else:
        print(f"  ✗ Poor alignment (r={r:.3f}) - check data!")

    # === STEP 8: Combine recordings ===
    print("\n[8] COMBINING RECORDINGS")
    rec_combined = combine_recordings(rec_n_z, rec_nnew_z, name="Combined")
    X_combined, _, reps_combined = average_per_stimulus(rec_combined, common)

    summarize_averaged(X_combined, "Combined")

    print(f"\n  Reps per stimulus:")
    print(f"    MonkeyN: {reps_n.min()}-{reps_n.max()} (mean={reps_n.mean():.1f})")
    print(f"    MonkeyNnew: {reps_nnew.min()}-{reps_nnew.max()} (mean={reps_nnew.mean():.1f})")
    print(f"    Combined: {reps_combined.min()}-{reps_combined.max()} (mean={reps_combined.mean():.1f})")

    # === STEP 9: Final checks ===
    print("\n[9] FINAL VALIDATION")
    all_ok = True

    # No NaN
    if not check_no_nan(X_n, "X_n"):
        all_ok = False
    if not check_no_nan(X_nnew, "X_nnew"):
        all_ok = False
    if not check_no_nan(X_combined, "X_combined"):
        all_ok = False

    # Shape consistency
    assert X_n.shape == X_nnew.shape == X_combined.shape
    print(f"  ✓ All shapes consistent: {X_combined.shape}")

    # Stimuli order consistency
    assert np.array_equal(stimuli, common)
    print(f"  ✓ Stimuli order consistent")

    if all_ok:
        print("\n" + "=" * 60)
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ SOME CHECKS FAILED - REVIEW ABOVE")
        print("=" * 60)

    # === Save validated data ===
    print("\n[10] SAVING VALIDATED DATA")
    np.savez(
        OUTPUT_DIR / "validated_data.npz",
        X_n=X_n,
        X_nnew=X_nnew,
        X_combined=X_combined,
        stimuli=stimuli,
        reps_n=reps_n,
        reps_nnew=reps_nnew,
        reps_combined=reps_combined
    )
    print(f"  Saved: {OUTPUT_DIR / 'validated_data.npz'}")


if __name__ == "__main__":
    main()
