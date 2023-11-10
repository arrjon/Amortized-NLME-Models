#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import logsumexp
from scipy.linalg import ldl as ldl_decomposition
from numba import jit

from typing import Optional


@jit(nopython=True)
def apply_huber_loss(x: float, delta: float) -> float:
    """apply huber loss to x"""
    if np.abs(x) <= delta:
        return 0.5 * x ** 2
    else:
        return delta * (np.abs(x) - 0.5 * delta)


@jit(nopython=True)
def second_term_gaussian(log_phi: np.ndarray,  # parameter from amortizer
                         log_phi_cov: np.ndarray,  # parameter including covariates
                         beta: np.ndarray,
                         psi_inverse: np.ndarray,
                         prior_mean: Optional[np.ndarray] = None,
                         prior_cov_inverse: Optional[np.ndarray] = None,
                         huber_loss_delta: Optional[float] = None) -> float:
    """compute second term of objective function for gaussian likelihood"""
    dif = log_phi_cov - beta  # parameters with covariates are normal, prior is only on parameters

    # population density
    if huber_loss_delta is None:
        # compute normal gaussian density
        temp_psi = 0.5 * np.dot(np.dot(dif, psi_inverse), dif.T)
    else:
        cholesky_psi = np.linalg.cholesky(psi_inverse)
        dif_psi_norm = np.linalg.norm(np.dot(dif, cholesky_psi))
        temp_psi = apply_huber_loss(dif_psi_norm, delta=huber_loss_delta)

    # prior
    if prior_cov_inverse is None:
        # uniform prior
        temp_sigma = 0.
    else:
        # gaussian prior
        dif_2 = log_phi - prior_mean
        temp_sigma = 0.5 * np.dot(np.dot(dif_2, prior_cov_inverse), dif_2.T)

    # sum up
    second_term = temp_sigma - temp_psi
    return second_term


@jit(nopython=True)
def compute_log_sum(n_sim: int,
                    n_samples: int,
                    param_samples: np.ndarray,
                    param_samples_cov: np.ndarray,
                    beta: np.ndarray,
                    psi_inverse: np.ndarray,
                    prior_mean: Optional[np.ndarray] = None,
                    prior_cov_inverse: Optional[np.ndarray] = None,
                    huber_loss_delta: Optional[float] = None  # 1.5 times the median of the standard deviations
                    ) -> np.ndarray:
    """compute log-sum-exp of second term in objective function with numba"""
    expectation_approx = np.zeros((n_sim, n_samples))
    for sim_idx, (p_sample, p_cov_sample) in enumerate(zip(param_samples, param_samples_cov)):
        for param_idx, (log_phi, log_phi_cov) in enumerate(zip(p_sample, p_cov_sample)):
            # compute cell-specific contribution to expectation value
            expectation_approx[sim_idx, param_idx] = second_term_gaussian(log_phi=log_phi,
                                                                          log_phi_cov=log_phi_cov,
                                                                          beta=beta,
                                                                          psi_inverse=psi_inverse,
                                                                          prior_mean=prior_mean,
                                                                          prior_cov_inverse=prior_cov_inverse,
                                                                          huber_loss_delta=huber_loss_delta)
    return expectation_approx


# define objective class
class ObjectiveFunctionNLME:
    """
    objective function for non-linear mixed effects models with normal population distribution
    prior can be either gaussian or uniform
    """

    def __init__(self,
                 model_name: str,
                 prior_mean: np.ndarray,  # needed for gaussian prior
                 prior_std: Optional[np.ndarray] = None,  # needed for gaussian prior
                 param_samples: Optional[np.ndarray] = None,  # (n_sim, n_samples, n_posterior_params)
                 covariance_format: str = 'diag',
                 covariates: Optional[np.ndarray] = None,
                 covariate_mapping: Optional[callable] = None,
                 penalize_correlations: Optional[float] = None,
                 huber_loss_delta: Optional[float] = None,
                 prior_type: str = 'normal',
                 prior_bounds: Optional[np.ndarray] = None):

        self.model_name = model_name
        self.param_samples = param_samples if param_samples is not None else np.empty((1, 1, 1))
        self.covariance_format = covariance_format
        if covariance_format != 'diag' and covariance_format != 'cholesky':
            raise ValueError(f'covariance_format must be either "diag" or "cholesky", but is {covariance_format}')
        self.penalize_correlations = penalize_correlations
        self.huber_loss_delta = huber_loss_delta

        self.prior_type = prior_type
        self.prior_mean = prior_mean
        if prior_type == 'normal':
            self.prior_cov_inverse = np.diag(1. / prior_std ** 2)
            self.constant_prior_term = -self._log_sqrt_det(self.prior_cov_inverse)
        elif prior_type == 'uniform':
            # log( (b-a)^n )
            self.prior_cov_inverse = None
            self.constant_prior_term = np.sum(np.log(np.diff(prior_bounds, axis=1)))
            # gaussian population density constant now does not cancel out
            self.constant_prior_term += np.log((2*np.pi)**(-prior_mean.size/2))
        else:
            raise ValueError(f'prior_type must be either "gaussian" or "uniform", but is {prior_type}')

        # some constants
        self.n_sim, self.n_samples, n_posterior_params = param_samples.shape
        self.param_dim = self.prior_mean.size

        # prepare covariates
        self.covariates = covariates
        self.covariate_mapping = covariate_mapping  # maps samples parameters, covariates to distributional parameters
        if covariates is not None:
            assert covariate_mapping is not None, '"covariate_mapping" must be specified if covariates are given'
            self.n_covariates = covariates.shape[1]
        else:
            self.n_covariates = 0
            self.param_samples_cov = self.param_samples

        # prepare computation of loss
        self.log_n_samples = np.log(self.n_samples)

        # some placeholders
        self.beta = np.empty(self.param_dim)
        self.psi_inverse = np.empty((self.param_dim, self.param_dim))
        self.part_one = np.empty(1)

    def update_param_samples(self, param_samples: np.ndarray, covariates: Optional[np.ndarray] = None) -> None:
        """update parameter samples, everything else stays the same"""
        self.param_samples = param_samples
        self.n_sim, self.n_samples, n_posterior_params = param_samples.shape
        self.log_n_samples = np.log(self.n_samples)
        assert n_posterior_params == self.prior_mean.size, 'number of posterior parameters does not match prior'

        if covariates is not None:
            self.covariates = covariates
            assert self.covariates.shape[0] == self.n_sim, \
                'number of covariates does not match number of simulations'
            assert self.covariate_mapping is not None, '"covariate_mapping" must be specified if covariates are given'
        else:
            self.param_samples_cov = self.param_samples
        return

    def _helper_log_sum(self, beta: np.ndarray, psi_inverse: np.ndarray) -> np.ndarray:
        """wrapper function to compute log-sum-exp of second term in objective function with numba"""
        expectation_approx = compute_log_sum(
            n_sim=self.n_sim,
            n_samples=self.n_samples,
            param_samples=self.param_samples,
            param_samples_cov=self.param_samples_cov,
            beta=beta,
            psi_inverse=psi_inverse,
            prior_mean=self.prior_mean,
            prior_cov_inverse=self.prior_cov_inverse,
            huber_loss_delta=self.huber_loss_delta
        )

        # log-sum-exp of second term
        log_sum_exp = logsumexp(expectation_approx, axis=1)  # more stable

        # take sum again over cells/individuals
        expectation_log_sum = np.sum(log_sum_exp)
        return expectation_log_sum

    # define objective function to minimize parameters
    def __call__(self, vector_params: np.ndarray) -> float:
        # get parameter vectors
        beta = vector_params[:self.param_dim]
        psi_inverse_vector = vector_params[self.param_dim:]
        psi_inverse = self.get_inverse_covariance(psi_inverse_vector)

        # include covariates
        if self.covariates is not None:
            self.param_samples_cov = self.covariate_mapping(param_samples=self.param_samples,
                                                            covariates=self.covariates,
                                                            params=beta)

        # compute parts of loss
        # gaussian prior: constant_prior_term is _log_sqrt_det of prior covariance
        # uniform prior: constant_prior_term is log of (b-a)^n
        part_one = self.n_sim * (self._log_sqrt_det(psi_inverse) - self.log_n_samples + self.constant_prior_term)
        expectation_log_sum = self._helper_log_sum(beta, psi_inverse)

        # compute negative log-likelihood
        nll = -(part_one + expectation_log_sum)

        # add penalty for correlations
        if self.penalize_correlations is not None:
            nll += self.penalize_correlations * np.sum(np.abs(psi_inverse[self.param_dim * 2:]))
        return nll

    @staticmethod
    @jit(nopython=True)
    def _log_sqrt_det(matrix: np.ndarray) -> np.ndarray:
        """compute log of square root of determinant of matrix with numba"""
        eig_values = np.linalg.eig(matrix)[0]
        # determinant = np.prod(eig_values)
        log_s_det = np.sum(np.log(np.sqrt(eig_values)))
        return log_s_det

    def get_inverse_covariance(self, psi_inverse_vector: np.ndarray) -> np.ndarray:
        if self.covariance_format == 'diag':
            # psi = log of diagonal entries since entries must be positive
            psi_inverse = np.diag(np.exp(psi_inverse_vector))
        elif self.covariance_format == 'cholesky':
            # vector to triangular matrix
            psi_inverse_lower = np.zeros((self.param_dim, self.param_dim))
            psi_inverse_lower[np.diag_indices(self.param_dim)] = 1
            psi_inverse_lower[np.tril_indices(self.param_dim, k=-1)] = psi_inverse_vector[self.param_dim:]

            psi_inverse_diag = np.diag(np.exp(psi_inverse_vector[:self.param_dim]))
            psi_inverse = psi_inverse_lower.dot(psi_inverse_diag).dot(psi_inverse_lower.T)
        else:
            raise ValueError(f'covariance_format must be either "diag" or "cholesky", '
                             f'but is {self.covariance_format}')
        return psi_inverse

    def get_covariance(self, psi_inverse_vector: np.ndarray) -> np.ndarray:
        inverse_covariance = self.get_inverse_covariance(psi_inverse_vector)
        psi = np.linalg.inv(inverse_covariance)
        return psi

    def get_inverse_covariance_vector(self, psi: np.ndarray) -> np.ndarray:
        if self.covariance_format == 'diag':
            # psi = log of diagonal entries since entries must be positive
            psi_inv_vector = -np.log(psi.diagonal())
        elif self.covariance_format == 'cholesky':
            # triangular matrix to vector
            psi_inv = np.linalg.inv(psi)
            lu, d, perm = ldl_decomposition(psi_inv)
            psi_inv_lower = lu[perm, :][np.tril_indices(self.param_dim, k=-1)]
            psi_inv_vector = np.concatenate((np.log(d.diagonal()), psi_inv_lower))
        else:
            raise ValueError(f'covariance_format must be either "diag" or "cholesky", '
                             f'but is {self.covariance_format}')
        return psi_inv_vector
