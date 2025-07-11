import os
import pickle
import copy
import time
from collections import defaultdict

import pyximport

pyximport.install()

import numpy as np
import itertools

import pandas as pd

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
from sklearn.utils import check_random_state, check_array

from sklearn.model_selection import KFold, StratifiedKFold

import warnings
from tqdm import tqdm, trange

# from src.bsum_cython import update_bsum
from models.bsum import update_w as update_bsum

# from kneed import KneeLocator
from matplotlib import pyplot
import seaborn as sns


def _update_coordinate_descent(M, W0, max_iter):
    W = update_bsum(M, W0, max_iter=max_iter, verbose=0)
    return W


class symNMF(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_components=2,  # number of factors
        *,
        rho=3.0,  # alternative minimziation parameters
        M_lowerbd=(True, -1.0),
        M_upperbd=(True, 1.0),
        bsum_iter=50,
        min_iter=10,
        max_iter=200,
        tol=1e-4,
        random_state=None,
        verbose=False,
    ):

        self.n_components = n_components
        self.rho = rho

        self.M_lowerbd = M_lowerbd
        self.M_upperbd = M_upperbd

        self.bsum_iter = bsum_iter
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol

        self.random_state = random_state
        self.verbose = verbose

        self.iteration = 0

        if self.M_lowerbd[0] == True:
            self.Mmin = self.M_lowerbd[1]
        if self.M_upperbd[0] == True:
            self.Mmax = self.M_upperbd[1]

        self.rng = check_random_state(self.random_state)

    def _vprint(self, str):
        if self.verbose == True:
            print(str)

    def initialize(self, M, nan_mask, W_initial=None):

        assert np.sum(np.isnan(M)) == 0

        if W_initial is not None:
            assert W_initial.shape[0] == M.shape[0]
            assert W_initial.shape[1] == self.n_components
            W = W_initial
        else:
            eigvals, eigvecs = np.linalg.eigh(M)
            idx = np.argsort(np.abs(eigvals))[::-1][: self.n_components]
            eigenvalues_sorted = eigvals[idx]
            eigenvectors_sorted = eigvecs[:, idx]
            W = eigenvectors_sorted @ np.diag(np.sqrt(np.abs(eigenvalues_sorted)))

        # initialize W satisfying the bounded constraints
        W[W < 0] = 0

        # initialize Z satisfying the bounded constraints
        # Z = W @ (W.T)
        # if self.M_lowerbd[0] == True:
        #     Z[Z < self.Mmin] = self.Mmin
        # if self.M_upperbd[0] == True:
        #     Z[Z > self.Mmax] = self.Mmax

        Z = copy.deepcopy(M)

        aZ = np.zeros_like(M)

        self.M = M
        self.nan_mask = nan_mask
        self.W = W
        self.Z = Z
        self.aZ = aZ

    def _update_Z(self, R):

        # update Z element-wise via solving
        # min || mask * (M - X) ||^2_F + rho * || X - R ||^2_F
        self.Z = (self.M * self.nan_mask + self.rho * R) / (
            self.rho + self.nan_mask * 1.0
        )

        # project to the target interval
        if self.M_lowerbd[0] == True:
            self.Z[self.Z < self.Mmin] = self.Mmin
        if self.M_upperbd[0] == True:
            self.Z[self.Z > self.Mmax] = self.Mmax

    def _update_W(self, B, W0):

        self.W = update_bsum(B, W0, max_iter=self.bsum_iter, verbose=self.verbose)
        # self.W = _update_coordinate_descent(B, W0, max_iter=self.bsum_iter)

    def _obj_func(self):
        return 0.5 * np.sum(self.nan_mask * (self.M - self.W @ (self.W.T)) ** 2)

    def _missmatch_loss(self, M, nan_mask, W, extra_mask):
        # normalized reconstruction error
        return np.sum(extra_mask * nan_mask * (M - W @ (W.T)) ** 2) / np.sum(
            extra_mask * nan_mask
        )

    # def fit(self, matrix_class):
    #     _ = self.fit_transform(matrix_class)

    def fit_transform(self, M, nan_mask, W_initial=None):

        # if self.n_components is None:
        #     _ = self.detect_dimension(M)
        # else:
        self.initialize(M, nan_mask, W_initial=W_initial)
        loss_history, bsum_loss_history = self._fit_transform()
        return self.W, loss_history, bsum_loss_history

    def _fit_transform(self):

        print("SNMF", self.W.min(), self.W.max())

        # initial distance value
        loss_history = []
        bsum_loss_history = []

        # tqdm setting
        tqdm_iterator = trange(
            self.max_iter, desc="symNMF", leave=True, disable=not self.verbose
        )

        self.params = defaultdict(list)

        # Main iteration
        for i in tqdm_iterator:

            # Z = v = W @ W.T = x at beginning of iteration
            #
            # subproblem W

            abc = self.Z + self.aZ / self.rho

            self._update_W(self.Z + self.aZ / self.rho, self.W)

            self.params["W"].append(self.W.min())
            self.params["W_sum"].append(self.W.sum())

            self.params["W_max"].append(self.W.max())

            R = self.W @ (self.W.T)
            bsum_loss_history.append(0.5 * np.sum(self.nan_mask * (self.M - R) ** 2))

            # subproblem Z
            self._update_Z(R - self.aZ / self.rho)

            # project Z to symmetric matrix space
            self.Z = 0.5 * (self.Z + self.Z.T)

            self.params["V"].append(self.Z.min())
            self.params["V_sum"].append(self.Z.sum())

            # auxiliary varibles update
            self.aZ += self.rho * (self.Z - R)

            self.params["lam"].append(self.aZ.min())
            self.params["lam_sum"].append(self.aZ.sum())

            # Iteration info
            loss_history.append(self._obj_func())

            if i > 0:
                err_ratio = np.abs(loss_history[-1] - loss_history[-2]) / np.abs(
                    loss_history[-1]
                )
            else:
                err_ratio = np.nan
            message = f"loss={loss_history[-1]:0.3e}, tol={err_ratio:0.3e}, "
            tqdm_iterator.set_description(message)
            tqdm_iterator.refresh()

            # Check convergence
            # if i > self.min_iter:
            #     converged = True
            #     if err_ratio < self.tol:
            #         self._vprint(
            #             "Algorithm converged with relative error < {}.".format(self.tol)
            #         )
            #         return loss_history, bsum_loss_history
            #     else:
            #         converged = False

        return loss_history, bsum_loss_history
