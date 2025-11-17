from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cli import ExperimentContext, register_experiment

from .tasks import generate_embedding, predict_semantic_axis

TaskName = Literal["generate_embedding", "semantic_axes"]


@dataclass(slots=True)
class Params:
    task: TaskName
    rank: int
    use_all_responses: bool
    top_n_words: int | None
    min_word_length: int
    symmetrization: str
    data_dir: Path
    ratings_data_dir: Path
    embedding_dir: Path | None = None


def add_arguments(parser) -> None:
    parser.add_argument("--task", choices=["generate_embedding", "semantic_axes"])
    parser.add_argument("--rank", type=int)
    parser.add_argument("--use-all-responses", action="store_true")
    parser.add_argument("--top-n-words", type=int)
    parser.add_argument("--min-word-length", type=int)
    parser.add_argument(
        "--symmetrization", choices=["geometric_mean", "arithmetic_mean", "max"]
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--ratings-data-dir", type=Path)


@register_experiment(
    name="word_association",
    params_type=Params,
    add_arguments=add_arguments,
    description="SWOW-based word association analyses",
)
def run(context: ExperimentContext, params: Params) -> None:
    if params.task == "generate_embedding":
        generate_embedding.run(context, params)
    elif params.task == "semantic_axes":
        if not params.embedding_dir:
            raise ValueError("--embedding-dir required")
        predict_semantic_axis.run(context, params)
