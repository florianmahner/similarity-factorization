from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig

from ..lib.data import (
    load_swow_data,
    filter_by_word_length,
)
from ..lib.ppmi import make_ppmi_graph
from ..lib.embedding import (
    fit_srf,
    get_top_words,
)


def run(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)

    df = load_swow_data(data_dir, cfg.use_all_responses)
    cues = df["cue"].tolist()
    responses = df["response"].tolist()
    counts = df["count"].tolist()

    if cfg.min_word_length > 1:
        cues, responses, counts = filter_by_word_length(
            cues, responses, counts, cfg.min_word_length
        )

    similarity, vocabulary, metadata = make_ppmi_graph(
        cues,
        responses,
        counts,
        symmetrization=cfg.symmetrization,
        top_n=cfg.top_n_words,
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
