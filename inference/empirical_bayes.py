#!/usr/bin/env python
# coding: utf-8

from typing import Optional
import numpy as np

from inference.likelihoods import log_likelihood_additive_noise, log_likelihood_multiplicative_noise


class ObjectiveFunctionEmpiricalBayes:
    def __init__(self,
                 data: np.ndarray,
                 pop_mean: np.ndarray,
                 pop_cov: np.ndarray,
                 sigmas: np.ndarray,
                 batch_simulator: Optional[callable] = None,
                 noise_type: str = "multiplicative",
                 ignore_conditional: bool = False  # returns only a likelihood
                 ):
        self.data = data
        self.pop_mean = pop_mean
        self.pop_cov = pop_cov
        self.sigmas = sigmas
        self.batch_simulator = batch_simulator
        self.ignore_conditional = ignore_conditional

        # precompute some values
        self.inv_pop_cov = np.linalg.inv(pop_cov)
        _, logabsdet = np.linalg.slogdet(self.pop_cov)
        self.precompute_mvn = -0.5 * (self.pop_mean.size * np.log(2 * np.pi) + logabsdet)
        self.noise_type = noise_type

    def __call__(self, individual_params: np.ndarray, sim_data: Optional[np.ndarray] = None) -> float:

        if sim_data is None:
            # simulate data from the simulator if not provided
            sim_data = self.batch_simulator(individual_params)

        if self.ignore_conditional:
            # then only the likelihood is returned
            conditional = 0
        else:
            # compute multivariate normal logpdf
            # \log p(x) =-\frac{k}{2}\log(2\pi) -\frac12 \log\vert\Sigma\vert-{\frac{1}{2}}(x-\mu)^T\Sigma^{-1}(x-\mu)
            dif = individual_params - self.pop_mean
            temp = dif.T.dot(self.inv_pop_cov).dot(dif)
            conditional = self.precompute_mvn - 0.5 * temp

        # compute multivariate normal logpdf
        if self.noise_type == "additive":
            log_likelihood = log_likelihood_additive_noise(measurements=self.data,
                                                           simulations=sim_data,
                                                           sigmas=self.sigmas)
        else:
            log_likelihood = log_likelihood_multiplicative_noise(log_measurements=self.data,
                                                                 log_simulations=sim_data,
                                                                 sigmas=self.sigmas)
        return -(log_likelihood + conditional)
