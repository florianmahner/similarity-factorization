#!/usr/bin/env python
"""Extract CLIP RN50 features from Things+ images.

This script extracts visual features from the penultimate layer (visual encoder)
of CLIP's ResNet-50 model for all images in the Things+ dataset.

Features are extracted from the visual encoder output (after attention pooling,
before the final projection head), which produces 1024-dimensional features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ThingsPlusDataset(Dataset):
    """Dataset wrapper for Things+ images."""

    def __init__(self, image_dir: Path, preprocess):
        self.image_paths = sorted(image_dir.glob("*.jpg"))
        self.preprocess = preprocess
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        processed = self.preprocess(image)
        return {
            "image": processed,
            "filename": image_path.stem,
        }


def extract_visual_features(model, images: torch.Tensor) -> torch.Tensor:
    """Extract features from the visual encoder of CLIP ResNet-50.

    Parameters
    ----------
    model : CLIP model
        CLIP model with ResNet-50 visual encoder
    images : torch.Tensor
        Preprocessed image tensor (batch_size, 3, 224, 224)

    Returns
    -------
    torch.Tensor
        Features from visual encoder, shape (batch_size, 1024)
    """
    visual_output = None

    def hook_fn(_module, _input, output):
        nonlocal visual_output
        visual_output = output

    # Register hook on the visual encoder output
    handle = model.visual.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model.encode_image(images)

    # Remove hook
    handle.remove()

    return visual_output


def main():
    """Extract CLIP RN50 features from Things+ images."""
    # Configuration
    image_dir = Path("/SSD/datasets/things/plus")
    output_dir = Path("/SSD/fmahner/dnn_activations/things_plus/clip_rn50")
    layer_name = "visual"
    batch_size = 64
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    # Create output directory
    layer_dir = output_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {layer_dir}")

    # Load CLIP RN50 model
    logger.info("Loading CLIP RN50 model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50", pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # Create dataset and dataloader
    dataset = ThingsPlusDataset(image_dir, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract features
    all_features = []
    all_filenames = []

    logger.info("Extracting features...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = batch["image"].to(device)
            filenames = batch["filename"]

            # Extract visual features
            features = extract_visual_features(model, images)

            all_features.append(features.cpu().numpy())
            all_filenames.extend(filenames)

    # Concatenate all features
    all_features = np.vstack(all_features)

    logger.info(f"Extracted features shape: {all_features.shape}")
    logger.info(f"Feature dimension: {all_features.shape[1]}")

    # Save full feature matrix
    logger.info("Saving feature matrix...")
    features_path = output_dir / "features.npy"
    np.save(features_path, all_features)
    logger.info(f"Saved feature matrix to {features_path}")

    # Save metadata
    metadata = {
        "model": "clip-rn50",
        "model_pretrained": "openai",
        "layer": layer_name,
        "layer_description": "Visual encoder output (penultimate layer before projection)",
        "n_images": len(all_filenames),
        "feature_dim": int(all_features.shape[1]),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "filenames": all_filenames,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("=" * 60)
    logger.info("Feature extraction complete!")
    logger.info(f"Total images processed: {len(all_filenames)}")
    logger.info(f"Feature dimension: {all_features.shape[1]}")
    logger.info(f"Feature matrix saved to: {features_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
