"""Visualize the top-200 VGG16 Gram embedding and closest object pairs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.colors import GRAY, ROSE, setup_style
from src.utils.figure_theme import despine

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs" / "vgg16_200d"
OUT.mkdir(parents=True, exist_ok=True)

K_PATH = ROOT / "experiments/datasets/dimensionality/outputs/cache/vgg16.npy"
META_PATH = ROOT / "data/features/vgg16/metadata.csv"
SSD_ROOT = Path("/data/labshare/_stachelschwein/SSD")

N_DIMS = 200
N_PAIRS = 40


def ssd_path(path: str) -> Path:
    if path.startswith("/SSD/"):
        return SSD_ROOT / path.removeprefix("/SSD/")
    return Path(path)


def gram_embedding(k: np.ndarray, n_dims: int) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(np.asarray(k, dtype=np.float64))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    pos = np.maximum(vals[:n_dims], 0.0)
    return vecs[:, :n_dims] * np.sqrt(pos), vals


def nearest_pairs(z: np.ndarray, meta: pd.DataFrame, k: np.ndarray) -> pd.DataFrame:
    sq = np.sum(z * z, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    tri_i, tri_j = np.triu_indices_from(d2, k=1)
    order = np.argsort(d2[tri_i, tri_j])[:N_PAIRS]
    rows = []
    diag = np.sqrt(np.maximum(np.diag(k), 1e-12))
    cosine = k / np.outer(diag, diag)
    for rank, idx in enumerate(order, 1):
        i = int(tri_i[idx])
        j = int(tri_j[idx])
        rows.append({
            "rank": rank,
            "i": i,
            "j": j,
            "category_i": str(meta.loc[i, "category"]),
            "category_j": str(meta.loc[j, "category"]),
            "distance_200d": float(np.sqrt(d2[i, j])),
            "kernel_i_j": float(k[i, j]),
            "cosine_i_j": float(cosine[i, j]),
            "norm_i": float(np.sqrt(k[i, i])),
            "norm_j": float(np.sqrt(k[j, j])),
            "image_i": str(ssd_path(str(meta.loc[i, "path"]))),
            "image_j": str(ssd_path(str(meta.loc[j, "path"]))),
        })
    return pd.DataFrame(rows)


def plot_scatter(z: np.ndarray, meta: pd.DataFrame, k: np.ndarray, pairs: pd.DataFrame) -> None:
    setup_style()
    norm = np.sqrt(np.diag(k))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sc = ax.scatter(
        z[:, 0], z[:, 1],
        c=np.log10(norm),
        s=10,
        cmap="viridis",
        linewidths=0,
        alpha=0.78,
    )
    for row in pairs.head(12).itertuples(index=False):
        ax.plot(
            [z[row.i, 0], z[row.j, 0]],
            [z[row.i, 1], z[row.j, 1]],
            color=ROSE,
            linewidth=0.8,
            alpha=0.75,
        )
    ax.set_title("VGG16 top-200 Gram embedding")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("log10 feature norm")
    despine(ax)
    fig.savefig(OUT / "embedding_2d_top200.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "embedding_2d_top200.pdf", bbox_inches="tight")
    plt.close(fig)


def contact_sheet(pairs: pd.DataFrame) -> None:
    thumb = 120
    pad = 12
    label_h = 34
    rows = min(20, len(pairs))
    width = 2 * thumb + 3 * pad
    height = rows * (thumb + label_h + pad) + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for r, row in enumerate(pairs.head(rows).itertuples(index=False)):
        y = pad + r * (thumb + label_h + pad)
        for col, path in enumerate([row.image_i, row.image_j]):
            x = pad + col * (thumb + pad)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((thumb, thumb))
                bx = x + (thumb - img.width) // 2
                by = y + (thumb - img.height) // 2
                canvas.paste(img, (bx, by))
            except Exception:
                draw.rectangle([x, y, x + thumb, y + thumb], outline=GRAY)
                draw.text((x + 5, y + 45), "missing", fill=(80, 80, 80))
        text = (
            f"{row.rank}. {row.category_i} | {row.category_j}  "
            f"d={row.distance_200d:.2f}, cos={row.cosine_i_j:.3f}"
        )
        draw.text((pad, y + thumb + 4), text[:72], fill=(20, 20, 20))

    canvas.save(OUT / "nearest_pairs_contact.png")


def main() -> None:
    k = np.asarray(np.load(K_PATH), dtype=np.float64)
    meta = pd.read_csv(META_PATH)
    if len(meta) != k.shape[0]:
        raise ValueError(f"metadata rows {len(meta)} != matrix n {k.shape[0]}")

    z, vals = gram_embedding(k, N_DIMS)
    pairs = nearest_pairs(z, meta, k)
    pairs.to_csv(OUT / "nearest_pairs_top200.csv", index=False)
    np.save(OUT / "embedding_top200.npy", z)
    plot_scatter(z, meta, k, pairs)
    contact_sheet(pairs)

    summary = {
        "n": int(k.shape[0]),
        "n_dims": N_DIMS,
        "metadata": str(META_PATH),
        "matrix": str(K_PATH),
        "positive_eigenvalues": int(np.sum(vals > 1e-8)),
        "top_200_spectral_fraction": float(np.maximum(vals[:N_DIMS], 0).sum() / np.maximum(vals, 0).sum()),
        "nearest_pairs_head": pairs.head(10).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
