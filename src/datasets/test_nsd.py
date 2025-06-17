# %%
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from h5py import File
from scipy.io import loadmat
from scipy.stats import zscore
from tqdm.auto import tqdm

NSD_DIR_IRIS = Path("/LOCAL/LABSHARE/natural-scenes-dataset/")


def load_betas_from_fp(
    fp: str, standardize: bool = True, voxel_indices: np.ndarray = None
) -> np.ndarray:
    # Betas need to be divided by 300, as they've been multiplied by 300 and stored
    # as int16 to save space (https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD)
    betas = nib.load(fp).get_fdata().astype(np.float32) / 300

    if voxel_indices is not None:
        betas = betas[voxel_indices]

    if standardize:
        betas = zscore(betas, axis=-1)

    return betas


def get_available_subjects(nsd_dir: Path = NSD_DIR_IRIS) -> list[int]:
    subject_dirs = sorted(
        glob.glob(str(nsd_dir / "nsddata_betas" / "ppdata" / "subj*"))
    )
    subject_ids = [int(Path(d).name[4:]) for d in subject_dirs]

    return subject_ids


def get_spaces(subject_id: int, nsd_dir: Path = NSD_DIR_IRIS) -> list[str]:
    return [
        Path(fp).stem
        for fp in sorted(
            glob.glob(
                str(
                    nsd_dir
                    / "nsddata_betas"
                    / "ppdata"
                    / f"subj{str(subject_id).zfill(2)}"
                    / "*"
                )
            )
        )
    ]


def get_available_rois(
    subject_id: int, nsd_dir: Path = NSD_DIR_IRIS, space: str = "func1pt8mm"
) -> list[str]:
    assert space in get_spaces(subject_id, nsd_dir)

    return [
        Path(fp).name.replace(".nii.gz", "")
        for fp in sorted(
            glob.glob(
                str(
                    nsd_dir
                    / "nsddata"
                    / "ppdata"
                    / f"subj{str(subject_id).zfill(2)}"
                    / space
                    / "roi"
                    / "*.nii.gz"
                )
            )
        )
    ]


def get_roi(
    subject_id: int,
    roi_name: str,
    nsd_dir: Path = NSD_DIR_IRIS,
    space: str = "func1pt8mm",
) -> np.ndarray:
    roi_fp = (
        nsd_dir
        / "nsddata"
        / "ppdata"
        / f"subj{str(subject_id).zfill(2)}"
        / space
        / "roi"
        / f"{roi_name}.nii.gz"
    )
    roi = nib.load(roi_fp).get_fdata().astype(np.float32)

    return roi


def load_nsd_betas(
    subject_id: int,
    zscore_betas: bool = True,
    nsd_dir: Path = NSD_DIR_IRIS,
    space: str = "func1pt8mm",
    voxel_indices: np.ndarray = None,
    max_workers: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    experiment_design = loadmat(
        nsd_dir / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    )
    sub_betas_fps = sorted(
        glob.glob(
            str(
                nsd_dir
                / "nsddata_betas"
                / "ppdata"
                / f"subj{str(subject_id).zfill(2)}"
                / space
                / "betas_fithrf_GLMdenoise_RR"
                / "betas_session*.nii.gz"
            )
        )
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                load_betas_from_fp,
                fp,
                zscore_betas,
                voxel_indices,
            ): i
            for i, fp in enumerate(sub_betas_fps)
        }

        betas = [None] * len(sub_betas_fps)
        for future in tqdm(
            as_completed(future_to_idx), total=len(future_to_idx), desc="Loading betas"
        ):
            idx = future_to_idx[future]
            betas[idx] = future.result()

    betas = np.concatenate(betas, axis=-1)

    trial_ordering = (
        experiment_design["subjectim"][
            subject_id - 1, experiment_design["masterordering"].squeeze() - 1
        ]
        - 1
    )

    unique_trials_and_indices = {}
    for i, trial in enumerate(trial_ordering):
        if trial not in unique_trials_and_indices:
            unique_trials_and_indices[trial] = [i]
        else:
            unique_trials_and_indices[trial].append(i)

    averaged_betas = []
    trials_with_betas = []
    for trial, indices in unique_trials_and_indices.items():
        stim_indices = [i for i in indices if i < betas.shape[1]]
        if len(stim_indices) > 0:
            averaged_betas.append(np.nanmean(betas[:, stim_indices], axis=1))
            trials_with_betas.append(trial)
    averaged_betas = np.array(averaged_betas)

    return averaged_betas, np.array(trials_with_betas)


def load_nsd_images(trials: list[int], nsd_dir: Path = NSD_DIR_IRIS):
    images_hdf5 = File(
        nsd_dir / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    )["imgBrick"]
    sorted_indices = np.argsort(trials)
    sorted_trials = np.array(trials)[sorted_indices]

    images = images_hdf5[sorted_trials, ...]
    images = images[np.argsort(sorted_indices)]

    return images


def load_nsd_data(
    subject_id: int,
    roi_name: str = "streams",
    space: str = "func1pt8mm",
    zscore_betas: bool = True,
    return_images: bool = True,
):

    subjects = get_available_subjects()
    subject_id = subjects[0]

    floc_faces = get_roi(subject_id, "streams")
    floc_faces_indices = floc_faces == 5

    betas, trials_with_betas = load_nsd_betas(
        subject_id, voxel_indices=floc_faces_indices
    )
    if not return_images:
        return betas

    images = load_nsd_images(trials_with_betas)
    return betas, images
