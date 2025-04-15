import numpy as np
from dataclasses import dataclass


SPOSE_PATH = "./data/misc/spose_embeddings.txt"


@dataclass
class CVParams:
    n_splits: int = 5
    n_repeats: int = 5
    max_iter: int = 500
    init: str = "random"
    random_state: int = 0
    candidate_ranks: list[int] = range(2, 7)
    verbose: int = 10


def load_spose_embedding(path=SPOSE_PATH, max_objects=None, max_dims=None):
    x = np.maximum(np.loadtxt(path), 0)
    if max_objects:
        x = x[:max_objects]
    if max_dims:
        x = x[:, :max_dims]
    return x
