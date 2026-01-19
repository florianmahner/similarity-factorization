from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig

from datasets import load_swow_ppmi

from .embedding import (
    fit_srf,
    get_top_words,
)


def run(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)

    similarity, vocabulary, metadata = load_swow_ppmi(
        data_dir,
        use_all_responses=cfg.use_all_responses,
        top_n_words=cfg.top_n_words,
        min_word_length=cfg.min_word_length,
        symmetrization=cfg.symmetrization,
        bidirectional_only=cfg.bidirectional_only,
    )

    model = fit_srf(similarity, cfg.rank, max_outer=cfg.max_outer)
    word_embedding = model.w_
    top_words = get_top_words(word_embedding, vocabulary)

    out_dir = Path.cwd()
    np.save(out_dir / "word_embedding.npy", word_embedding)
    np.save(out_dir / "similarity.npy", similarity)
    joblib.dump(model, out_dir / "model.joblib")

    words_to_idx = {word: i for i, word in enumerate(vocabulary)}

    full_metadata = {
        "vocabulary": vocabulary,
        "words_to_idx": words_to_idx,
        "graph_metadata": metadata,
        "top_words": top_words.to_dict("list"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(full_metadata, f)
