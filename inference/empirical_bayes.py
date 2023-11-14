#!/usr/bin/env python
# coding: utf-8

from typing import Optional
import numpy as np


class ObjectiveFunctionEmpiricalBayes:
    def __init__(self,
                 data: np.ndarray,
                 pop_mean: np.ndarray,
                 pop_cov: np.ndarray,
                 inv_error_cov: np.ndarray,
                 batch_simulator: Optional[callable] = None,
                 ):
        self.data = data
        self.pop_mean = pop_mean
        self.pop_cov = pop_cov
        self.inv_error_cov = inv_error_cov
        log_det_inv_error_cov = np.linalg.slogdet(self.inv_error_cov).logabsdet
        self.precompute_mvn_likelihood = -inv_error_cov.shape[0] * 0.5 * np.log(2 * np.pi) + 0.5 * log_det_inv_error_cov
        self.batch_simulator = batch_simulator

        # precompute some values
        self.inv_pop_cov = np.linalg.inv(pop_cov)
        self.precompute_mvn = -0.5 * (self.pop_mean.size * np.log(2 * np.pi) +
                                      np.linalg.slogdet(self.pop_cov).logabsdet)

    def update_data(self, data: np.ndarray):
        self.data = data
        return

    def __call__(self, individual_params: np.ndarray, sim_data: Optional[np.ndarray] = None) -> float:

        if sim_data is None:
            # simulate data from the simulator if not provided
            sim_data = self.batch_simulator(individual_params)

        # compute multivariate normal logpdf
        # \log p(x) =-\frac{k}{2}\log(2\pi) -\frac12 \log\vert\Sigma\vert-{\frac{1}{2}}(x-\mu)^T\Sigma^{-1}(x-\mu)
        dif = individual_params - self.pop_mean
        temp = dif.T.dot(self.inv_pop_cov).dot(dif)
        conditional = self.precompute_mvn - 0.5 * temp

        # compute multivariate normal logpdf
        # \log p(x) =-\frac{k}{2}\log(2\pi) -\frac12 \log\vert\Sigma\vert-{\frac{1}{2}}(x-\mu)^T\Sigma^{-1}(x-\mu)
        dif = self.data - sim_data
        temp = dif.T.dot(self.inv_error_cov).dot(dif)
        log_likelihood = self.precompute_mvn_likelihood - 0.5 * temp

        return -(log_likelihood + conditional)
