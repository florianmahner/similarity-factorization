#!/usr/bin/env python3
"""
Residualize species-specific CCA spaces against cross-species C (use all C comps).

For M (monkey) and H (human):
- Fit multioutput Ridge (alpha via CV) to predict Y from C
- Residuals: R = Y - C W_hat (computed on full data with chosen alpha)
- NMF on residuals (rank=20) after column-wise min-shift (no ReLU)

Saves: data/results/residualized/ccaw_residual.pkl
Visuals handled by viz/CCA/viz_cca_residuals.py
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys
import pickle
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import NMF

_here = os.path.dirname(__file__)
for _u in (1,2,3,4):
    _cand = os.path.abspath(os.path.join(_here, *(['..']*_u)))
    if os.path.exists(os.path.join(_cand, 'config')):
        sys.path.append(_cand) if _cand not in sys.path else None
        break

from config.paths import config
from model.feature.core.feature_io import load_cca_components
from model.feature.core.ridge_utils import get_ridge_params


def _energy_ratio(R, Y):
    num = float((R*R).sum())
    den = float((Y*Y).sum()) + 1e-12
    return num/den

def _fit_alpha_cv(C, Y, kfolds, alphas):
    xs = StandardScaler().fit(C)
    ys = StandardScaler().fit(Y)
    Xs = xs.transform(C)
    Ys = ys.transform(Y)
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    a_sel = []

    for tr, te in kf.split(Xs):
        rc = RidgeCV(alphas=alphas, cv=3).fit(Xs[tr], Ys[tr])
        a_sel.append(float(rc.alpha_))
    # choose most frequent alpha (mode); fallback to first if tie
    uniq, cnt = np.unique(np.array(a_sel), return_counts=True)
    a = float(uniq[np.argmax(cnt)])
    return a

def _residual_full(C, Y, alpha):
    xs = StandardScaler().fit(C)
    ys = StandardScaler().fit(Y)
    Xs = xs.transform(C)
    Ys = ys.transform(Y)
    mdl = Ridge(alpha=float(alpha)).fit(Xs, Ys)
    Yhat_s = mdl.predict(Xs)
    R_raw = (Ys - Yhat_s) * ys.scale_  # back to original units
    return R_raw.astype(np.float32)

def _cross_pred_r2(Xtr, Ytr, Xte, Yte, alphas):
    # X->Y multioutput ridge; return mean R^2 across outputs on test
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xtr = xs.transform(Xtr); Xte = xs.transform(Xte)
    Ytr = ys.transform(Ytr); Yte = ys.transform(Yte)
    rc = RidgeCV(alphas=alphas, cv=3).fit(Xtr, Ytr)
    Yp = rc.predict(Xte)
    num = ((Yte - Yp)**2).sum(axis=0)
    den = ((Yte - Yte.mean(axis=0))**2).sum(axis=0) + 1e-12
    r2 = 1.0 - (num/den)
    return float(np.mean(r2))

def _cv_diagnostics(C, M, H, kfolds, alphas, n_jobs=1):
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    idx = list(kf.split(C))

    def _one(tr, te):
        # Standardize per fold for each target
        xsC = StandardScaler().fit(C[tr])
        Ctr = xsC.transform(C[tr]); Cte = xsC.transform(C[te])

        # M ~ C
        ysM = StandardScaler().fit(M[tr])
        Mtr = ysM.transform(M[tr]); Mte = ysM.transform(M[te])
        rcM = RidgeCV(alphas=alphas, cv=3).fit(Ctr, Mtr)
        Rm_tr = Mtr - rcM.predict(Ctr)
        Rm_te = Mte - rcM.predict(Cte)
        em = float((Rm_te*Rm_te).sum() / ((Mte*Mte).sum() + 1e-12))

        # H ~ C
        ysH = StandardScaler().fit(H[tr])
        Htr = ysH.transform(H[tr]); Hte = ysH.transform(H[te])
        rcH = RidgeCV(alphas=alphas, cv=3).fit(Ctr, Htr)
        Rh_tr = Htr - rcH.predict(Ctr)
        Rh_te = Hte - rcH.predict(Cte)
        eh = float((Rh_te*Rh_te).sum() / ((Hte*Hte).sum() + 1e-12))

        # Cross-predictability on folds
        r2_m2h = _cross_pred_r2(Rm_tr, Htr, Rm_te, Hte, alphas)
        r2_h2m = _cross_pred_r2(Rh_tr, Mtr, Rh_te, Mte, alphas)

        return {
            'alpha_m': float(rcM.alpha_), 'alpha_h': float(rcH.alpha_),
            'energy_m': em, 'energy_h': eh,
            'pred_m2h_r2': r2_m2h, 'pred_h2m_r2': r2_h2m,
        }

    outs = (Parallel(n_jobs=int(n_jobs))(delayed(_one)(tr, te) for tr, te in idx)
            if int(n_jobs) != 1 else [
                _one(tr, te) for tr, te in idx
            ])
    return outs

def _nmf_minshift(R, rank=20, max_iter=100000, seed=42):
    X = R - R.min(axis=0, keepdims=True)
    X[X < 0] = 0
    nmf = NMF(n_components=int(rank), init='nndsvda', random_state=seed, max_iter=int(max_iter))
    W = nmf.fit_transform(X)
    return W.astype(np.float32)

def main():
    # Load CCA components (N x d)
    C, stims = load_cca_components('all')
    M, _ = load_cca_components('monkey')
    H, _ = load_cca_components('human')

    N = C.shape[0]
    assert M.shape[0] == N and H.shape[0] == N

    alphas, _ = get_ridge_params()
    kfolds = int(config.hyperparameters.get('crossview', {}).get('n_folds', 5))
    n_jobs = int(config.analysis.get('n_jobs', 1))

    # CV diagnostics across folds
    folds = _cv_diagnostics(C, M, H, kfolds, alphas, n_jobs=n_jobs)

    # Choose final alpha (mode across folds)
    am = _fit_alpha_cv(C, M, kfolds, alphas)
    ah = _fit_alpha_cv(C, H, kfolds, alphas)

    # Final residuals on full data (back in original units)
    Rm = _residual_full(C, M, am)
    Rh = _residual_full(C, H, ah)
    e_m = _energy_ratio(Rm, M)
    e_h = _energy_ratio(Rh, H)

    # NMF on residuals (rank=20, min-shift columns)
    rank = 20
    max_iter = 10000
    Wm = _nmf_minshift(Rm, rank=rank, max_iter=max_iter)
    Wh = _nmf_minshift(Rh, rank=rank, max_iter=max_iter)

    out = {
        'stims': stims,
        'params': {
            'n_jobs': n_jobs,
            'n_folds': kfolds,
            'alphas_grid': [float(a) for a in alphas],
            'alpha_m': float(am), 'alpha_h': float(ah),
            'energy': {'monkey': float(e_m), 'human': float(e_h)},
            'folds': folds,
        },
        'shared': {
            'scores': C.astype(np.float32, copy=False),
            'n_components': int(C.shape[1]),
        },
        'monkey': {
            'resid': Rm,
            'nmf': {'W': Wm, 'rank': rank},
        },
        'human': {
            'resid': Rh,
            'nmf': {'W': Wh, 'rank': rank},
        },
    }

    out_dir = Path(config.results_dir) / 'residualized'
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / 'ccaw_residual.pkl'
    with open(fp, 'wb') as f:
        pickle.dump(out, f)
    print(f"[CCA_residuals] wrote {fp}")


if __name__ == '__main__':
    main()
