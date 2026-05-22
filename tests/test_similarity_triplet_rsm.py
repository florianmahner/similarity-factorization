from __future__ import annotations

import numpy as np

from src.similarity.triplet_rsm import (
    build_bias_aware_triplet_matrix,
    fit_triplet_bias_prior,
)


def test_fit_triplet_bias_prior_returns_symmetric_outputs() -> None:
    triplets = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
        ]
    )

    wins, trials, prior = fit_triplet_bias_prior(4, triplets, weight_mode="shown")

    assert np.allclose(wins, wins.T)
    assert np.allclose(trials, trials.T)
    assert np.allclose(prior, prior.T)


def test_zero_strength_matches_laplace_matrix_on_observed_entries() -> None:
    triplets = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
        ]
    )

    wins, trials, prior = fit_triplet_bias_prior(4, triplets, alpha=1.0, weight_mode="shown")
    matrix = build_bias_aware_triplet_matrix(
        wins=wins,
        trials=trials,
        prior=prior,
        alpha=1.0,
        prior_strength=0.0,
        fill_missing_with_prior=False,
    )

    expected = np.divide(
        wins + 1.0,
        trials + 2.0,
        out=np.full_like(wins, np.nan, dtype=np.float64),
        where=trials > 0,
    )
    np.fill_diagonal(expected, 1.0)

    observed = trials > 0
    assert np.allclose(matrix[observed], expected[observed])


def test_missing_entries_can_be_filled_from_prior() -> None:
    wins = np.array(
        [
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    trials = np.array(
        [
            [0.0, 6.0, 0.0],
            [6.0, 0.0, 4.0],
            [0.0, 4.0, 0.0],
        ]
    )
    prior = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ]
    )

    matrix = build_bias_aware_triplet_matrix(
        wins=wins,
        trials=trials,
        prior=prior,
        alpha=1.0,
        prior_strength=10.0,
        fill_missing_with_prior=True,
    )

    assert matrix[0, 2] == prior[0, 2]
    assert matrix[2, 0] == prior[2, 0]
