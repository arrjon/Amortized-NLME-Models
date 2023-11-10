#!/usr/bin/env python
# coding: utf-8

from typing import Optional

import numpy as np


class ObjectiveFunctionEmpiricalBayes:
    def __init__(self,
                 data: np.ndarray,
                 pop_mean: np.ndarray,
                 pop_cov: np.ndarray,
                 batch_simulator: Optional[callable] = None,
                 inv_error_cov: Optional[np.ndarray] = None):
        self.data = data
        self.pop_mean = pop_mean
        self.pop_cov = pop_cov
        self.inv_error_cov = inv_error_cov
        if inv_error_cov is not None:
            log_det_inv_error_cov = self._log_det(self.inv_error_cov)
            self.precompute_mvn_likelihood = (-inv_error_cov.shape[0] / 2 * np.log(2 * np.pi)
                                              + 1 / 2 * log_det_inv_error_cov)
        self.batch_simulator = batch_simulator

        # precompute some values
        self.inv_pop_cov = np.linalg.inv(pop_cov)
        self.precompute_mvn = - self.pop_mean.size / 2 * np.log(2 * np.pi) - 1 / 2 * self._log_det(self.pop_cov)

    def update_data(self, data: np.ndarray):
        self.data = data
        return

    def __call__(self, individual_params: np.ndarray, sim_data: Optional[np.ndarray] = None,
                 inv_error_cov: Optional[np.ndarray] = None,) -> float:

        if sim_data is None:
            # batch simulator expects array of form (sim=1, params)
            sim_data = self.batch_simulator(individual_params[np.newaxis, :])

        # compute multivariate normal logpdf
        # \log p(x) =-\frac{k}{2}\log(2\pi) -\frac12 \log\vert\Sigma\vert-{\frac{1}{2}}(x-\mu)^T\Sigma^{-1}(x-\mu)
        dif = individual_params - self.pop_mean
        temp = dif.T.dot(self.inv_pop_cov).dot(dif)
        conditional = self.precompute_mvn - 1 / 2 * temp

        # compute multivariate normal logpdf
        # \log p(x) =-\frac{k}{2}\log(2\pi) -\frac12 \log\vert\Sigma\vert-{\frac{1}{2}}(x-\mu)^T\Sigma^{-1}(x-\mu)
        dif = self.data - sim_data
        if inv_error_cov is None:
            # use precomputed values since we always use the same error covariance (same number of measurements)
            inv_error_cov = self.inv_error_cov
            precompute_mvn_likelihood = self.precompute_mvn_likelihood
        else:
            log_det_inv_error_cov = self._log_det(inv_error_cov)
            precompute_mvn_likelihood = -sim_data.shape[0] / 2 * np.log(2 * np.pi) + 1 / 2 * log_det_inv_error_cov
        temp = dif.T.dot(inv_error_cov).dot(dif)
        log_likelihood = precompute_mvn_likelihood - 1 / 2 * temp

        return -(log_likelihood + conditional)

    @staticmethod
    def _log_det(matrix: np.ndarray) -> np.ndarray:
        """compute log of determinant of matrix"""
        eig_values = np.linalg.eig(matrix)[0]
        # determinant = np.prod(eig_values)
        log_det = np.sum(np.log(eig_values))
        return log_det

