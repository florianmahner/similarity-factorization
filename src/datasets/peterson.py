import os
import shutil
from glob import glob
from PIL import Image
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import re

from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader

RELIABILITIES = {
    "animals": 0.90,
    "automobiles": 0.83,
    "fruits": 0.57,
    "furniture": 0.65,
    "various": 0.70,
    "vegetables": 0.62,
}


def load_peterson(root: str) -> np.ndarray:
    image_folder = Path(root) / "images"
    images = sorted(glob(f"{image_folder}/*.png") + glob(f"{image_folder}/*.jpg"))
    rsm_file = Path(root) / "rsm.npy"
    rsm = np.load(rsm_file)
    unique_images = [re.sub(r"\d+", "", img) for img in images]
    unique_images = list(set(unique_images))
    targets = [unique_images.index(re.sub(r"\d+", "", img)) for img in images]
    rsm = rsm / rsm.max()
    return images, rsm, targets


@register_dataset("peterson-various")
class PetersonVarious(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("peterson-various", root)

    def load(self) -> BaseDataset:
        images, rsm, targets = load_peterson(self.root)

        return BaseDataset(
            name="peterson-various",
            images=images,
            rsm=rsm,
            targets=targets,
            reliabilities=RELIABILITIES["various"],
        )


@register_dataset("peterson-animals")
class PetersonAnimals(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("peterson-animals", root)

    def load(self) -> BaseDataset:
        images, rsm, targets = load_peterson(self.root)

        return BaseDataset(
            name="peterson-animals",
            images=images,
            rsm=rsm,
            targets=targets,
            reliabilities=RELIABILITIES["animals"],
        )


def process_peterson(
    input_folder: str = "./data/peterson_raw/datasets",
    output_folder: str = "./data/peterson_processed",
):
    images_folders = {
        "animals": "images",
        "automobiles": "alt_auto_v2b_edited",
        "fruits": "joshs_3per_cat_picks/fruit_sim_set",
        "furniture": "furn-120-sim",
        "various": "various_objects_v4b_edited",
        "vegetables": "josh-veggies-picks-120-similarity-set",
    }

    mat_files = {
        "animals": "turkSimMatrix2016.mat",
        "automobiles": "simMatrix_mturkvehicles.mat",
        "fruits": "fruits_mturk_sim.mat",
        "furniture": "simMatrix_mturkfurniture.mat",
        "various": "simMatrix_mturkmurreplication.mat",
        "vegetables": "simMatrix_mturkveggies.mat",
    }

    rsm_names = {
        "animals": "simMatrixturk2016",
        "automobiles": "simMatrix_mturkvehicles",
        "fruits": "simmat",
        "furniture": "simMatrix_mturkfurniture",
        "various": "simMatrix_mturkmurreplication",
        "vegetables": "simMatrix_mturkveggies",
    }

    domains = images_folders.keys()

    domain_folders = {domain: domain for domain in domains}
    domain_folders["various"] = (
        "mur_120_replication"  # for some reason, the data for various is not in a folder called various
    )

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    for domain in domains:
        print(domain)
        domain_in_folder = f"{input_folder}/{domain_folders[domain]}"
        domain_in_img_folder = f"{domain_in_folder}/{images_folders[domain]}"
        domain_in_rsm_path = f"{domain_in_folder}/{mat_files[domain]}"

        domain_out_folder = f"{output_folder}/{domain}"
        domain_out_img_folder = f"{domain_out_folder}/images"
        domain_out_rsm_path = f"{domain_out_folder}/rsm.npy"

        os.makedirs(domain_out_img_folder)
        image_files = glob(f"{domain_in_img_folder}/*.png") + glob(
            f"{domain_in_img_folder}/*.jpg"
        )

        assert len(image_files) == 120
        for image_file in image_files:
            print(image_file)
            img_out_path = f'{domain_out_img_folder}/{image_file.split("/")[-1]}'

            if image_file.endswith(".jpg"):
                Image.open(image_file).save(img_out_path.replace(".jpg", ".png"))
            else:
                shutil.copyfile(image_file, img_out_path)

        rsm = loadmat(domain_in_rsm_path)[rsm_names[domain]]
        assert rsm.shape == (120, 120)
        assert isinstance(rsm, np.ndarray)
        np.save(domain_out_rsm_path, rsm)
