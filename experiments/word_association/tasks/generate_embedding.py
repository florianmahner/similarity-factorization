from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cli import ExperimentContext

from ..lib.data import (
    load_swow_data,
    filter_by_word_length,
)
from ..lib.ppmi import make_ppmi_graph
from ..lib.embedding import (
    fit_srf,
    compute_coherence,
    compute_sparsity,
    get_top_words,
    compute_reconstruction_metrics,
)

if TYPE_CHECKING:
    from ..exp import Params


def run(context: ExperimentContext, params: Params) -> Path:
    df = load_swow_data(params.data_dir, params.use_all_responses)
    cues = df["cue"].tolist()
    responses = df["response"].tolist()
    counts = df["count"].tolist()

    if params.min_word_length > 1:
        cues, responses, counts = filter_by_word_length(
            cues, responses, counts, params.min_word_length
        )

    similarity, vocabulary, metadata = make_ppmi_graph(
        cues,
        responses,
        counts,
        symmetrization=params.symmetrization,
        top_n=params.top_n_words,
    )

    word_embedding = fit_srf(similarity, params.rank)
    coherence = compute_coherence(word_embedding, similarity)
    sparsity = compute_sparsity(word_embedding)
    top_words = get_top_words(word_embedding, vocabulary)
    recon = compute_reconstruction_metrics(word_embedding, similarity)

    context.save_npy(word_embedding, "word_embedding")
    context.save_npy(similarity, "similarity")
    (context.run_dir / "vocabulary.txt").write_text("\n".join(vocabulary))
    context.save_csv(sparsity.assign(coherence=coherence), "sparsity")
    context.save_csv(top_words, "top_words")
    context.save_csv(
        pd.DataFrame(
            [
                {
                    **metadata,
                    **recon,
                    "rank": params.rank,
                    "mean_gini": sparsity["gini"].mean(),
                    "mean_coherence": coherence.mean(),
                }
            ]
        ),
        "summary",
    )

    context.logger.info(
        "Fitted SRF: rank=%d, n_words=%d, ppmi=[%.1f, %.1f]",
        params.rank,
        len(vocabulary),
        metadata["ppmi_min"],
        metadata["ppmi_max"],
    )
    return context.run_dir
