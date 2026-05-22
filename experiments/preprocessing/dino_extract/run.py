"""Extract CLIP ResNet50 features for THINGS+ images.

Uses the THINGS+ image set (one image per category, 1854 images)
and extracts visual features from CLIP RN50 (OpenAI).
Features are non-negative (post-ReLU), suitable for linear kernel + SRF.

Saves to data/features/clip_rn50/ for the pipeline.

Usage:
    ./scripts/submit experiments/preprocessing/dino_extract/run.py --bg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

THINGS_PLUS_DIR = Path("/SSD/datasets/things/plus")
MODEL_NAME = "RN50"
PRETRAINED = "openai"
BATCH_SIZE = 64


def get_images() -> tuple[list[Path], list[str]]:
    paths = sorted(THINGS_PLUS_DIR.glob("*.jpg"))
    categories = [p.stem for p in paths]
    return paths, categories


def extract_features(
    paths: list[Path],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device,
    )
    model.eval()

    all_features = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
            batch_paths = paths[i:i + batch_size]
            images = torch.stack([
                preprocess(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)

            features = model.encode_image(images)
            features = features.cpu().numpy()
            all_features.append(features)

    features = np.vstack(all_features)
    features = np.maximum(features, 0)
    return features


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    paths, categories = get_images()
    print(f"Found {len(paths)} THINGS+ images")

    features = extract_features(paths, MODEL_NAME, PRETRAINED, BATCH_SIZE, device)
    print(f"Features: {features.shape}")
    print(f"Min: {features.min():.4f}, Max: {features.max():.4f}")
    print(f"Fraction zero: {(features == 0).mean():.3f}")

    np.save(OUTPUT_DIR / "clip_rn50_features.npy", features)

    metadata = pd.DataFrame({
        "category": categories,
        "path": [str(p) for p in paths],
    })
    metadata.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    target = Path(__file__).resolve().parents[3] / "data" / "features" / "clip_rn50"
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "clip_rn50_features.npy", features)
    metadata.to_csv(target / "metadata.csv", index=False)
    print(f"Saved to {OUTPUT_DIR} and {target}")


if __name__ == "__main__":
    main()
