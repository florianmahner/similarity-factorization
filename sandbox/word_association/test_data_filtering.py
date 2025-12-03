#!/usr/bin/env python3
from __future__ import annotations

import itertools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from experiments.word_association.lib import data as data_lib
from experiments.word_association.lib import ppmi as ppmi_lib
from experiments.word_association.lib import embedding as embedding_lib
from experiments.word_association.lib.semantic_eval import validate_unsupervised
from experiments.word_association.lib.validation import load_behavioral_ratings

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "small-world-of-words"
RATINGS_DIR = REPO_ROOT / "data"
OUTPUT_ROOT = (
    REPO_ROOT / "experiments" / "word_association" / "development" / "filtering"
)

USE_ALL_OPTIONS = [True, False]
BIDIRECTIONAL_OPTIONS = [False, True]
TOP_N_OPTIONS = [1000]
DEFAULT_RANK = 50
MAX_OUTER = 400


def build_config_grid() -> list[dict[str, object]]:
    grid = []
    for use_all, bidir, top_n in itertools.product(
        USE_ALL_OPTIONS, BIDIRECTIONAL_OPTIONS, TOP_N_OPTIONS
    ):
        resp = "R123" if use_all else "R1"
        edge = "bidir" if bidir else "all"
        grid.append(
            {
                "name": f"{resp}+{edge}_top{top_n}",
                "use_all_responses": use_all,
                "bidirectional_only": bidir,
                "top_n": top_n,
            }
        )
    return grid


def build_embedding(config: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    df = data_lib.load_swow_data(DATA_DIR, config["use_all_responses"])
    cues = df["cue"].tolist()
    responses = df["response"].tolist()
    counts = df["count"].tolist()

    cues, responses, counts = data_lib.filter_by_word_length(cues, responses, counts, 2)

    similarity, vocabulary, _ = ppmi_lib.make_ppmi_graph(
        cues,
        responses,
        counts,
        symmetrization="geometric_mean",
        top_n=config["top_n"],
        bidirectional_only=config["bidirectional_only"],
    )

    model = embedding_lib.fit_srf(similarity, DEFAULT_RANK, max_outer=MAX_OUTER)
    return model.w_, [word.lower() for word in vocabulary]


def evaluate_configuration(config: dict[str, object]) -> pd.DataFrame:
    embedding, vocabulary = build_embedding(config)
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    ratings = load_behavioral_ratings(RATINGS_DIR)
    stats = validate_unsupervised(
        word_embedding=embedding,
        vocabulary=vocabulary,
        word_to_idx=word_to_idx,
        ratings=ratings,
        top_n=50,
        use_cosine=False,
    )
    stats["configuration"] = config["name"]
    stats["top_n"] = config["top_n"]
    stats["use_all_responses"] = config["use_all_responses"]
    stats["bidirectional_only"] = config["bidirectional_only"]
    return stats


def plot_results(df: pd.DataFrame, output_dir: Path, run_id: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="axis",
        y="correlation",
        hue="configuration",
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Spearman correlation")
    plt.xlabel("Semantic axis")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"filtering_comparison_{run_id}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(6, 4))
    animacy = df[df["axis"] == "animacy"]
    sns.lineplot(
        data=animacy,
        x="top_n",
        y="correlation",
        hue="configuration",
        marker="o",
    )
    plt.ylabel("Animacy correlation")
    plt.xlabel("top_n words")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"animacy_trend_{run_id}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    configs = build_config_grid()
    print(f"Evaluating {len(configs)} configurations in parallel...")
    frames = Parallel(n_jobs=-1)(
        delayed(evaluate_configuration)(cfg) for cfg in configs
    )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / f"filtering_comparison_{run_id}.csv", index=False)

    plot_results(combined, output_dir, run_id)

    summary = (
        combined.groupby(["configuration", "axis"])["correlation"]
        .mean()
        .unstack(0)
        .round(3)
    )
    summary.to_csv(output_dir / f"filtering_summary_{run_id}.csv")

    print(f"Done. Outputs saved to {output_dir} (run {run_id})")


if __name__ == "__main__":
    main()
