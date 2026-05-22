"""Plot top-k THINGS images for saved macaque embeddings.

This is a plotting-only helper. It does not fit SRF; it reads an existing
embedding.npy or W_*.npy and writes a dimensions x top-k thumbnail grid.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.monkey import load_macaque
from src.utils import get_output_dir

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
DEFAULT_EMBEDDING = ROOT / "experiments/datasets/consensus/outputs/things_macaque22k/embedding.npy"
DEFAULT_FILENAMES = (
    ROOT
    / "experiments/sandbox/things/things_monkey_22k_embedding/outputs/22k_gaussian/filenames.txt"
)
SSD_ROOT = Path("/data/labshare/_stachelschwein/SSD/datasets/things/core")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def category_from_filename(filename: str) -> str:
    return Path(filename).name.rsplit("_", 1)[0]


def build_paths_from_filenames(filenames_path: Path, stimuli: list[str]) -> list[Path]:
    if not filenames_path.exists():
        raise FileNotFoundError(f"Exact THINGS filename list not found: {filenames_path}")

    filenames = [line.strip() for line in filenames_path.read_text().splitlines() if line.strip()]
    if len(filenames) != len(stimuli):
        raise ValueError(
            f"filename count {len(filenames)} does not match macaque rows {len(stimuli)}"
        )

    mismatches = [
        (i, filename, stimulus)
        for i, (filename, stimulus) in enumerate(zip(filenames, stimuli, strict=True))
        if category_from_filename(filename) != stimulus
    ]
    if mismatches:
        preview = ", ".join(f"row {i}: {filename} vs {stimulus}" for i, filename, stimulus in mismatches[:5])
        raise ValueError(f"filename list is not aligned with macaque stimuli ({preview})")

    paths = [SSD_ROOT / category_from_filename(filename) / filename for filename in filenames]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} exact THINGS image files are missing, first paths:\n{preview}")

    return paths


def load_font():
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).exists():
            return ImageFont.truetype(cand, 10), ImageFont.truetype(cand, 13)
    return ImageFont.load_default(), ImageFont.load_default()


def make_grid(
    w: np.ndarray,
    paths: list[Path],
    categories: list[str],
    out_path: Path,
    title: str,
    top_k: int,
    thumb: int,
    max_dims: int | None,
) -> None:
    col_mass = np.asarray(w.sum(axis=0))
    order = np.argsort(col_mass)[::-1]
    if max_dims is not None:
        order = order[:max_dims]

    pad = 3
    row_label_w = 92
    title_h = 34
    width = row_label_w + top_k * thumb + (top_k + 1) * pad
    row_h = thumb + 2 * pad
    height = title_h + len(order) * row_h + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, title_font = load_font()

    draw.text((pad, 8), title, fill=(20, 20, 20), font=title_font)

    for r, dim in enumerate(order):
        y0 = title_h + r * row_h
        loadings = w[:, dim]
        top_idx = np.argsort(loadings)[::-1][:top_k]
        nnz = int((loadings > 1e-8).sum())
        draw.text(
            (4, y0 + 5),
            f"d{int(dim) + 1}\nmass={col_mass[dim]:.1f}\nnnz={nnz}\nmax={loadings.max():.2f}",
            fill=(20, 20, 20),
            font=font,
        )
        for c, idx in enumerate(top_idx):
            x = row_label_w + pad + c * (thumb + pad)
            y = y0 + pad
            path = paths[int(idx)]
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((thumb, thumb))
                canvas.paste(img, (x + (thumb - img.width) // 2, y + (thumb - img.height) // 2))
            except Exception:
                draw.rectangle([x, y, x + thumb, y + thumb], outline=(150, 150, 150))
                draw.text((x + 4, y + 4), "miss", fill=(120, 120, 120), font=font)

            cat = str(categories[int(idx)])[:14]
            draw.rectangle([x, y + thumb - 13, x + thumb, y + thumb - 1], fill=(255, 255, 255))
            draw.text((x + 2, y + thumb - 12), cat, fill=(20, 20, 20), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    log(f"saved {out_path} ({width}x{height}px)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--filenames", type=Path, default=DEFAULT_FILENAMES)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--thumb", type=int, default=84)
    parser.add_argument("--max-dims", type=int, default=None)
    args = parser.parse_args()

    out_dir = get_output_dir()
    embedding_path = args.embedding.resolve()
    w = np.load(embedding_path)
    log(f"loaded {embedding_path} shape={w.shape}")

    _, stimuli, _ = load_macaque(
        "22k",
        "F",
        root="/data/labshare/_stachelschwein/SSD/datasets/things/macaque",
        roi="it",
        min_reliab=0.3,
        average_exemplars=False,
    )
    if len(stimuli) != w.shape[0]:
        raise ValueError(f"stimuli length {len(stimuli)} does not match embedding rows {w.shape[0]}")
    paths = build_paths_from_filenames(args.filenames.resolve(), list(stimuli))
    log(f"loaded exact THINGS filenames from {args.filenames.resolve()} images={len(paths)} missing=0")

    if args.out is None:
        stem = embedding_path.parent.name if embedding_path.name == "embedding.npy" else embedding_path.stem
        suffix = f"rank{w.shape[1]}_k{args.top_k}_exact_images"
        if args.max_dims is not None:
            suffix += f"_top{args.max_dims}dims"
        out_path = out_dir / f"{stem}_dim_topk_{suffix}.png"
    else:
        out_path = args.out

    title = f"macaque IT saved embedding - rank={w.shape[1]} - top-{args.top_k} highest-loading images per dim"
    make_grid(w, paths, list(stimuli), out_path, title, args.top_k, args.thumb, args.max_dims)


if __name__ == "__main__":
    main()
