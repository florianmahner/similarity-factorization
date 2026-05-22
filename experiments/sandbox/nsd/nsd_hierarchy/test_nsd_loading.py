"""
Test NSD data loading against Johannes's reference implementation.

Run with: poetry run pytest test_nsd_loading.py -v
"""
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from joblib import Parallel, delayed
import glob
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from datasets.nsd_utils import NSD_DIR_IRIS, get_roi, load_nsd_betas


def load_betas_johannes(subject_id: int = 1, roi_name: str = "floc-faces"):
    """
    Johannes's exact implementation from johannes_nsd/data.py.
    This is the reference/ground truth.
    """
    experiment = loadmat(
        NSD_DIR_IRIS / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    )

    # Lines 129-132 from data.py
    stim_indices = (
        experiment["subjectim"][:, experiment["masterordering"][0] - 1][0] - 1
    )
    extraction_indices = np.unique(stim_indices)  # SORTED

    # Get ROI mask
    roi_mask = get_roi(subject_id, roi_name)
    mask_flat = (roi_mask > 0).flatten()

    # Load betas (Johannes does NOT zscore)
    beta_fps = sorted(glob.glob(str(
        NSD_DIR_IRIS / "nsddata_betas" / "ppdata" / f"subj{str(subject_id).zfill(2)}"
        / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR" / "betas_session*.nii.gz"
    )))

    def load_session(fp):
        betas_4d = nib.load(fp).get_fdata().astype(np.float32) / 300
        betas_2d = betas_4d.reshape(-1, betas_4d.shape[-1])
        return betas_2d[mask_flat]

    betas_list = Parallel(n_jobs=8)(delayed(load_session)(fp) for fp in beta_fps)
    raw_betas = np.concatenate(betas_list, axis=-1)  # (n_voxels, n_trials)
    selected_betas = raw_betas.T  # Johannes: (n_trials, n_voxels)

    # Lines 247-257: averaging
    averaged_betas = np.zeros((len(extraction_indices), selected_betas.shape[1]))
    for i, unique_img_idx in enumerate(extraction_indices):
        img_idx = np.where(stim_indices == unique_img_idx)[0]
        img_idx = [_ for _ in img_idx if _ < selected_betas.shape[0]]
        img_betas = selected_betas[img_idx]
        img_betas = np.nanmean(img_betas, axis=0)
        averaged_betas[i] = img_betas
    averaged_betas = np.nan_to_num(averaged_betas)

    return averaged_betas, extraction_indices


class TestNSDLoading:
    """Test that our NSD loading matches Johannes's implementation."""

    @pytest.fixture(scope="class")
    def johannes_data(self):
        """Load data using Johannes's method (reference)."""
        print("\nLoading Johannes reference data...")
        betas, trials = load_betas_johannes(subject_id=1, roi_name="floc-faces")
        return {"betas": betas, "trials": trials}

    @pytest.fixture(scope="class")
    def our_data(self):
        """Load data using our method."""
        print("\nLoading our data...")
        roi_mask = get_roi(1, "floc-faces")
        betas, trials = load_nsd_betas(1, voxel_indices=roi_mask > 0, zscore_betas=False, max_workers=8)
        return {"betas": betas, "trials": trials}

    def test_trials_are_sorted(self, our_data):
        """Our trials should be sorted (matching Johannes)."""
        trials = our_data["trials"]
        assert np.all(np.diff(trials) > 0), "Trials should be sorted in ascending order"

    def test_trials_match(self, johannes_data, our_data):
        """Trial IDs should match exactly."""
        n = len(our_data["trials"])
        np.testing.assert_array_equal(
            our_data["trials"],
            johannes_data["trials"][:n],
            err_msg="Trial IDs don't match Johannes's implementation"
        )

    def test_betas_shape_match(self, johannes_data, our_data):
        """Betas should have same shape."""
        n = len(our_data["trials"])
        assert our_data["betas"].shape == johannes_data["betas"][:n].shape, \
            f"Shape mismatch: ours={our_data['betas'].shape}, johannes={johannes_data['betas'][:n].shape}"

    def test_betas_values_match(self, johannes_data, our_data):
        """Betas values should match exactly."""
        n = len(our_data["trials"])
        np.testing.assert_allclose(
            our_data["betas"],
            johannes_data["betas"][:n],
            rtol=1e-6,
            atol=1e-6,
            err_msg="Beta values don't match Johannes's implementation"
        )

    def test_specific_stimulus(self, johannes_data, our_data):
        """Spot check a specific stimulus."""
        # Check first stimulus (smallest ID)
        first_stim = johannes_data["trials"][0]
        our_idx = np.where(our_data["trials"] == first_stim)[0]
        assert len(our_idx) == 1, f"Stimulus {first_stim} not found in our data"

        np.testing.assert_allclose(
            our_data["betas"][our_idx[0]],
            johannes_data["betas"][0],
            rtol=1e-6,
            err_msg=f"Betas for stimulus {first_stim} don't match"
        )


def run_quick_check():
    """Quick sanity check without pytest."""
    print("=" * 60)
    print("Quick verification: Our code vs Johannes")
    print("=" * 60)

    print("\nLoading Johannes reference...")
    betas_j, trials_j = load_betas_johannes()
    print(f"  Shape: {betas_j.shape}, Trials: {len(trials_j)}")
    print(f"  First 5 trials: {trials_j[:5]}")

    print("\nLoading our implementation...")
    roi_mask = get_roi(1, "floc-faces")
    betas_o, trials_o = load_nsd_betas(1, voxel_indices=roi_mask > 0, zscore_betas=False, max_workers=8)
    print(f"  Shape: {betas_o.shape}, Trials: {len(trials_o)}")
    print(f"  First 5 trials: {trials_o[:5]}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Note: Johannes includes stimuli with no betas (NaN->0), we skip them
    # So we need to compare only stimuli that exist in both
    print(f"Johannes has {len(trials_j)} stimuli (includes empty)")
    print(f"We have {len(trials_o)} stimuli (only with valid betas)")

    # Find common stimuli
    common_trials = np.intersect1d(trials_o, trials_j)
    print(f"Common stimuli: {len(common_trials)}")

    # Compare for common stimuli
    trials_match = True
    betas_match = True
    max_diff = 0

    for trial in common_trials[:100]:  # Check first 100
        j_idx = np.where(trials_j == trial)[0][0]
        o_idx = np.where(trials_o == trial)[0][0]

        # Johannes might have NaN/0 for some, skip those
        if np.isnan(betas_j[j_idx]).any() or (betas_j[j_idx] == 0).all():
            continue

        diff = np.abs(betas_o[o_idx] - betas_j[j_idx]).max()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            betas_match = False
            print(f"  Mismatch at trial {trial}: diff={diff:.6f}")

    print(f"\nTrials sorted correctly: {np.all(np.diff(trials_o) > 0)}")
    print(f"First trial matches: {trials_o[0] == trials_j[0]}")
    print(f"Betas match (first 100 common): {betas_match}")
    print(f"Max diff: {max_diff:.10f}")

    return betas_match


if __name__ == "__main__":
    success = run_quick_check()
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
