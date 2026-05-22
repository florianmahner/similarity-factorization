from __future__ import annotations

import warnings
from typing import Union

import numpy as np

RandomStateLike = Union[
    None,
    int,
    np.integer,
    np.random.SeedSequence,
    np.random.Generator,
    np.random.RandomState,
]


def as_seed_sequence(random_state: RandomStateLike) -> np.random.SeedSequence:
    if isinstance(random_state, np.random.SeedSequence):
        return random_state
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.SeedSequence(random_state)
    if isinstance(random_state, np.random.Generator):
        seed_seq = getattr(random_state.bit_generator, "seed_seq", None)
        if isinstance(seed_seq, np.random.SeedSequence):
            return seed_seq
        return np.random.SeedSequence(int(random_state.integers(0, 2**31 - 1)))
    if isinstance(random_state, np.random.RandomState):
        return np.random.SeedSequence(int(random_state.randint(0, 2**31 - 1)))
    raise TypeError(
        f"random_state must be None, int, SeedSequence, Generator, or "
        f"RandomState; got {type(random_state).__name__}"
    )


def spawn_ints(random_state: RandomStateLike, n: int) -> np.ndarray:
    children = as_seed_sequence(random_state).spawn(n)
    return np.array([int(s.generate_state(1)[0]) for s in children], dtype=np.uint32)


def is_nan_marker(value: float | None) -> bool:
    return value is np.nan or (isinstance(value, float) and np.isnan(value))


def observation_mask(x: np.ndarray, missing_values: float | None = np.nan) -> np.ndarray:
    if missing_values is None or is_nan_marker(missing_values):
        return np.isfinite(x)
    return np.not_equal(x, missing_values)


def symmetrize_observations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.stack([x, x.T]), axis=0)
    diagonal = np.diag(out).copy()
    diagonal[~np.isfinite(diagonal)] = 0.0
    np.fill_diagonal(out, diagonal)
    return out
