from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from tools.rsa import compute_rsm
from PIL import Image
import numpy as np
from nilearn import datasets
import pandas as pd
from nilearn.maskers import NiftiMasker
import warnings

# TODO - not good yet.


@register_dataset("haxby-faces")
class HaxbyFaces(BaseDatasetLoader):
    def __init__(self, root: str = None, sub_no: int = 4):
        super().__init__("haxby-faces", root)
        self.sub_no = sub_no

    def load_rsm(self) -> np.ndarray:

        # Fetch Haxby dataset (fMRI data with clear cognitive labels)
        haxby_dataset = datasets.fetch_haxby()

        # Load behavioral labels
        labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
        conditions = labels["labels"]

        # We take only faces and houses for binary classification
        # condition_mask = conditions.isin(["face", "house", "bottle", "face", "cat", "shoe"])
        condition_mask = conditions.isin(["face", "house"])
        y = conditions[condition_mask]

        # Encode targets as integers (0, 1)
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Create a masker object to vectorize fMRI images
        masker = NiftiMasker(mask_img=haxby_dataset.mask_vt[0], standardize=True)

        # Apply masker and get data as numpy array
        X = masker.fit_transform(haxby_dataset.func[0])

        # Apply the condition mask to the data
        X = X[condition_mask]

        return BaseDataset(
            name="haxby-faces",
            images=None,
            rsm=X,
            targets=y_encoded,
            cls_labels=y,
            mri_data=X,
        )


if __name__ == "__main__":
    dataset = HaxbyFaces("/SSD/datasets/haxby-2001")
    dataset.load()
