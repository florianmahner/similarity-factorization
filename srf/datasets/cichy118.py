from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from tools.rsa import compute_rsm
from tools.stats import apply_transform
from pathlib import Path
from scipy.io import loadmat
from glob import glob
from .helpers import group_level_rsa

@register_dataset("cichy118")
class Cichy118(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("cichy118", root)

    def load(self) -> BaseDataset:
        image_folder = Path(self.root) / "images"
        images = sorted(glob(f"{image_folder}/*.jpg"))
        mri_dir = Path(self.root) / "fmri_roidata_new_all"
        mri_file = Path(mri_dir) / "118_fmri_hvc_raw_new_unconstrained_single.mat"
        mri_data = loadmat(mri_file)["data"].ravel()
        mri_data = [d.T for d in mri_data]
        group_level_rsm, per_subject_rsms = group_level_rsa(mri_data, metric="cosine")

        # behavior
        behavior_file = Path(self.root) / "behavior" / "118_behavior_rdm_single.mat"
        behavior_data = loadmat(behavior_file)["RDM118_arrange"].T
        behavior_single_rsm = 1 - behavior_data

        behavior_group_file = (
            Path(self.root) / "behavior" / "118_behavior_rdm_group.mat"
        )
        behavior_group_data = loadmat(behavior_group_file)["RDM118_arrange_reduc"].T

        behavior_average_rsm = 1 - behavior_group_data

        return BaseDataset(
            name="cichy118",
            images=images,
            mri_data=mri_data,
            group_rsm=group_level_rsm,
            subject_rsms=per_subject_rsms,
            behavior_average_rsm=behavior_average_rsm,
            behavior_single_rsm=behavior_single_rsm,
        )


