import numpy as np
import joblib
from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from tools.rsa import compute_similarity


@register_dataset("things-monkey-22k")
class ThingsMonkey22k(BaseDatasetLoader):
    def __init__(self, root: str = None, reliability_threshold: float = 0.6):
        super().__init__("things-monkey-22k", root)
        self.reliability_threshold = reliability_threshold

    def load(self) -> BaseDataset:
        data = joblib.load(
            "/LOCAL/fmahner/monkey-dimensions/data/monkey/22k/f/THINGS_normMUA_processed.pkl"
        )
        x_it = data["data_filtered_it_0.3"]
        filenames = np.loadtxt(
            "/LOCAL/fmahner/monkey-dimensions/data/monkey/22k/f/index_to_image.txt",
            dtype=str,
        )

        return BaseDataset(
            name="things-monkey-22k",
            it=x_it,
            filenames=filenames,
        )


@register_dataset("things-monkey-2k")
class ThingsMonkey2k(BaseDatasetLoader):
    def __init__(self, root: str = None, reliability_threshold: float = 0.6):
        super().__init__("things-monkey-2k", root)
        self.reliability_threshold = reliability_threshold

    def load(self) -> BaseDataset:
        data = joblib.load(
            "/LOCAL/fmahner/monkey-dimensions/data/monkey/2k/f/THINGS_normMUA_processed.pkl"
        )
        x = data["train_MUA_averaged"]
        it_localizer = 768
        reliab = data["reliab"].mean(0)[it_localizer:]
        x_it = x[:, it_localizer:]
        x_it = x_it[:, reliab > self.reliability_threshold]
        filenames = np.loadtxt(
            "/LOCAL/fmahner/monkey-dimensions/data/monkey/2k/f/index_to_image.txt",
            dtype=str,
        )
        rsm = compute_similarity(x_it, x_it, "pearson")

        return BaseDataset(
            name="things-monkey-2k",
            it=x_it,
            filenames=filenames,
            rsm=rsm,
        )
