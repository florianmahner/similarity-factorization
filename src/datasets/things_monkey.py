import numpy as np
import joblib
from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from tools.rsa import compute_similarity
from pathlib import Path


def get_channel_mask(monkey_type: str, roi: str | None = None):
    if roi is None:
        return slice(None)

    if (
        monkey_type == "N"
    ):  # https://gin.g-node.org/paolo_papale/TVSD/src/master/_code/norm_MUA.m
        masks = {"v1": slice(0, 512), "v4": slice(512, 768), "it": slice(768, 1024)}
    else:  # monkey F
        masks = {"v1": slice(0, 512), "it": slice(512, 832), "v4": slice(832, 1024)}
    return masks[roi.lower()]


def load_monkey(
    monkey_path: str = "/SSD/projects/deepsim/raw/macaque",
    monkey_type: str = "F",
    roi: str | None = "it",
    min_reliab: float | None = None,
):
    path = Path(monkey_path) / f"monkey{monkey_type}.npy"
    monkey_data = np.load(path, allow_pickle=True).item()
    filenames = monkey_data["train_things_path"]
    filenames = np.char.split(filenames, "\\")
    filenames = [i[-1] for i in filenames]
    data = monkey_data["train_MUA"].astype("float32").T
    reliab = monkey_data["reliab"].mean(axis=1)

    # First select roi and track kept channels
    roi_mask = get_channel_mask(monkey_type, roi)
    if isinstance(roi_mask, slice):
        # Convert slice to boolean array
        full_mask = np.zeros(data.shape[1], dtype=bool)
        full_mask[roi_mask] = True
        inc_chans = np.arange(data.shape[1])[roi_mask]
    else:
        full_mask = roi_mask
        inc_chans = np.where(roi_mask)[0]

    data = data[:, roi_mask]
    reliab = reliab[roi_mask]

    # Then apply reliability threshold
    if min_reliab is not None:
        reliab_mask = reliab >= min_reliab
        data = data[:, reliab_mask]
        inc_chans = inc_chans[reliab_mask]

    return data, filenames


@register_dataset("things-monkey-22k")
class ThingsMonkey22k(BaseDatasetLoader):
    def __init__(
        self,
        root: str = "/SSD/projects/deepsim/raw/macaque",
        min_reliab: float = 0.6,
        monkey_type: str = "F",
        roi: str | None = "it",
    ):
        super().__init__("things-monkey-22k", root)
        self.min_reliab = min_reliab
        self.monkey_type = monkey_type
        self.roi = roi

    def load(self) -> BaseDataset:
        data, filenames = load_monkey(
            monkey_path=self.root,
            monkey_type=self.monkey_type,
            roi=self.roi,
            min_reliab=self.min_reliab,
        )

        return BaseDataset(
            name="things-monkey-22k",
            it=data,
            filenames=filenames,
            rsm=compute_similarity(data, data, "gaussian_kernel"),
        )
