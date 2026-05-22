"""Export the vgg16 similarity matrix as a single .npy for sharing.

Builds the same matrix used by `experiments/datasets/dimensionality` and
saves it alongside a small README documenting how it was built so a
collaborator can reproduce / interpret it.

Run:
    poetry run python sandbox/dimensionality/vgg16_npy/run.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from similarity import build_similarity
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    cfg_path = PROJECT_ROOT / "configs" / "dataset" / "vgg16.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": OmegaConf.load(PROJECT_ROOT / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(PROJECT_ROOT),
    })
    OmegaConf.resolve(parent)
    cfg = parent.dataset

    log(f"Building vgg16 similarity from {cfg.path}/{cfg.features_file}")
    log(f"  similarity_fn={cfg.similarity_fn}  bounds_task={cfg.bounds_task}")
    t0 = time.time()
    s = build_similarity(cfg)
    s = np.asarray(s, dtype=np.float64)
    log(f"  built in {time.time() - t0:.1f}s  shape={s.shape}  "
        f"dtype={s.dtype}  finite_all={np.all(np.isfinite(s))}")

    out_path = OUTPUT_DIR / "vgg16_similarity.npy"
    np.save(out_path, s)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log(f"Saved {out_path}  ({size_mb:.1f} MB)")

    readme = OUTPUT_DIR / "README.md"
    readme.write_text(
        "# vgg16_similarity.npy\n\n"
        f"- shape: {s.shape}\n"
        "- dtype: float64\n"
        "- symmetric, full (no missing values)\n"
        f"- source: VGG16 penultimate features ({cfg.features_file}) "
        "for the 1854 THINGS+ categories\n"
        f"- similarity: linear kernel with /max normalization (config "
        f"`configs/dataset/vgg16.yaml`)\n"
        f"- value range: [{s.min():.4f}, {s.max():.4f}], "
        f"mean off-diag: {s[np.triu_indices(s.shape[0], 1)].mean():.4f}\n\n"
        "Load with:\n"
        "```python\n"
        "import numpy as np\n"
        "S = np.load('vgg16_similarity.npy')\n"
        "```\n"
    )
    log(f"Saved {readme}")


if __name__ == "__main__":
    main()
