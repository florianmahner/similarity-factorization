from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from pathlib import Path
from scipy.io import loadmat
from glob import glob
from .helpers import group_level_rsa


@register_dataset("mur92")
class Mur92(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("mur92", root)

    def load(self) -> BaseDataset:
        image_folder = Path(self.root) / "images"
        images = sorted(glob(f"{image_folder}/*.jpg"))
        mri_dir = Path(self.root) / "fmri_roidata_new_all"
        mri_file = Path(mri_dir) / "92_fmri_hvc_raw_new_unconstrained_single.mat"
        mri_data = loadmat(mri_file)["data"].ravel()
        mri_data = [d.T for d in mri_data]
        # this is the problem for the optimal rank that it is super high
        group_level_rsm, per_subject_rsms = group_level_rsa(mri_data, metric="pearson")
        return BaseDataset(
            name="mur92",
            images=images,
            mri_data=mri_data,
            group_rsm=group_level_rsm,
            subject_rsms=per_subject_rsms,
        )
