"""Held-out predictive losses for SBM cross-validation.

Implements the loss families from plan_SBM_clustering.md §3 and §7:

  - ``gaussian_mse``    : MSE for continuous similarities (§7.1).
  - ``gaussian_nll``    : Gaussian negative log-likelihood with per-rank
                           training-set sigma^2 estimate (§3.2, §7.1).
  - ``bernoulli``       : Bernoulli NLL for binary adjacency (§3.1, §7.1).
  - ``poisson``         : Poisson NLL for count edges (§3.3, §7.1).
  - ``frac_bernoulli``  : Fractional Bernoulli cross-entropy for affinities
                           in [0, 1] (§3.4). Quasi-likelihood only.

The single user-facing entry point is ``score_sbm_predictions(S, S_hat, val_mask,
loss, extra_params=None)``, matching the interface stipulated in §13.4.
"""
from __future__ import annotations
import numpy as np

LOSS_FAMILIES = ("gaussian_mse", "gaussian_nll", "bernoulli", "poisson", "frac_bernoulli")


def _clip_prob(p, eps=1e-6):
    return np.minimum(1.0 - eps, np.maximum(eps, p))


def _validate_mask(S, val_mask):
    """Restrict val_mask to finite off-diagonal entries."""
    n = S.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(S)
    return val_mask & off_diag & finite


def score_sbm_predictions(S, S_hat, val_mask, loss, extra_params=None):
    """Compute per-edge loss on the validation entries.

    Parameters
    ----------
    S : (n, n) ndarray
        Observed similarity / adjacency. NaNs treated as missing.
    S_hat : (n, n) ndarray
        Predicted similarity. Must be finite everywhere on the validation mask.
    val_mask : (n, n) boolean ndarray
        True at validation entries (symmetric). Diagonal is always excluded.
    loss : str
        One of LOSS_FAMILIES.
    extra_params : dict or None
        - For ``gaussian_nll``: must contain ``sigma2`` (training-set variance).
        - For ``poisson``: may contain ``eps`` (default 1e-6) for log clipping.
        - For ``bernoulli``/``frac_bernoulli``: may contain ``eps`` (default 1e-6).

    Returns
    -------
    dict with keys:
        ``loss_mean`` : mean per-edge loss over the validation entries.
        ``loss_total``: sum of per-edge loss over the validation entries.
        ``n_scored``  : number of validation pairs scored (each unordered pair
                         counted once, so n_scored = (val_mask & off-diag &
                         finite).sum() / 2).
    """
    extra_params = extra_params or {}
    eps = float(extra_params.get("eps", 1e-6))

    mask = _validate_mask(S, val_mask)
    # Symmetric mask: each unordered pair appears twice in upper+lower.
    # Use upper-triangular only to avoid double counting.
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_mask = mask[iu]
    if not pair_mask.any():
        return {"loss_mean": float("nan"), "loss_total": 0.0, "n_scored": 0}

    s = S[iu][pair_mask]
    s_hat = S_hat[iu][pair_mask]

    if loss == "gaussian_mse":
        per = (s - s_hat) ** 2
    elif loss == "gaussian_nll":
        sigma2 = float(extra_params.get("sigma2", np.nan))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return {"loss_mean": float("nan"), "loss_total": float("nan"),
                    "n_scored": int(pair_mask.sum())}
        per = 0.5 * np.log(sigma2) + (s - s_hat) ** 2 / (2.0 * sigma2)
    elif loss == "bernoulli":
        p = _clip_prob(s_hat, eps)
        # s assumed in {0, 1}; we still let it be fractional and use cross-entropy.
        per = -(s * np.log(p) + (1.0 - s) * np.log(1.0 - p))
    elif loss == "frac_bernoulli":
        p = _clip_prob(s_hat, eps)
        s_clip = _clip_prob(s, eps)  # purely a numerical convenience
        per = -(s_clip * np.log(p) + (1.0 - s_clip) * np.log(1.0 - p))
    elif loss == "poisson":
        lam = np.maximum(s_hat, eps)
        per = lam - s * np.log(lam)
    else:
        raise ValueError(f"unknown loss {loss!r}, expected one of {LOSS_FAMILIES}")

    return {
        "loss_mean": float(np.mean(per)),
        "loss_total": float(np.sum(per)),
        "n_scored": int(pair_mask.sum()),
    }


def training_sigma2(S, S_hat, train_mask):
    """Training-set Gaussian variance estimate sigma_hat^2 used by gaussian_nll.

    Uses each unordered pair once (upper triangle), only finite entries, only
    off-diagonal entries.
    """
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    mask = train_mask[iu] & np.isfinite(S[iu]) & np.isfinite(S_hat[iu])
    if not mask.any():
        return float("nan")
    res = S[iu][mask] - S_hat[iu][mask]
    return float(np.mean(res ** 2))
