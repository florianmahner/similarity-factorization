"""Extract DINOv3 features for THINGS+ images."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

THINGS_PLUS_DIR = Path("/SSD/datasets/things/behav1854")
IMAGE_INFO_PATH = "/SSD/projects/deepsim/raw/features/image_info.csv"
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # ViT-L variant
BATCH_SIZE = 16


def get_things_plus_images() -> tuple[list[Path], list[str]]:
    """Get paths and categories for THINGS+ images."""
    info = pd.read_csv(IMAGE_INFO_PATH)
    plus_mask = info["filename"].str.contains("_plus")
    categories = info.loc[plus_mask, "category"].tolist()

    paths = []
    for cat in categories:
        img_path = THINGS_PLUS_DIR / cat / f"{cat}_01b.jpg"
        paths.append(img_path)

    return paths, categories


def extract_features(
    paths: list[Path],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Extract DINOv3 features for all images."""
    print(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_features = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
            batch_paths = paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)

            # Use pooler_output (CLS token)
            features = outputs.pooler_output.cpu().numpy()
            all_features.append(features)

    return np.vstack(all_features)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Getting THINGS+ image paths...")
    paths, categories = get_things_plus_images()
    print(f"Found {len(paths)} images")

    # Verify paths exist
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Warning: {len(missing)} images not found")
        print(f"First missing: {missing[0]}")

    print(f"\nExtracting DINOv3 features...")
    features = extract_features(paths, MODEL_NAME, BATCH_SIZE, device)
    print(f"Features shape: {features.shape}")

    # Save features
    np.save(OUTPUT_DIR / "dinov3_features.npy", features)

    # Save metadata
    metadata = pd.DataFrame({
        "category": categories,
        "path": [str(p) for p in paths],
    })
    metadata.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
