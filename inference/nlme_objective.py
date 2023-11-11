#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import logsumexp
from scipy.linalg import ldl as ldl_decomposition
from numba import njit, prange

from typing import Optional


@njit()
def second_term_gaussian(log_phi: np.ndarray,  # individual parameter excluding covariates (for prior)
                         log_phi_cov: np.ndarray,  # individual parameter including covariates
                         beta: np.ndarray,
                         psi_inverse: np.ndarray,
                         prior_mean: np.ndarray,
                         prior_cov_inverse: Optional[np.ndarray] = None,
                         huber_loss_delta: Optional[float] = None,
                         cholesky_psi: Optional[np.ndarray] = None  # precomputed cholesky decomposition for huber loss
                         ) -> float:
    """compute second term of objective function for gaussian likelihood"""
    dif = log_phi_cov - beta

    # population density
    if huber_loss_delta is None:
        # compute normal gaussian density
        temp_psi = 0.5 * np.dot(np.dot(dif, psi_inverse), dif.T)
    else:
        # compute huber loss
        dif_psi_norm = np.linalg.norm(np.dot(dif, cholesky_psi))
        if np.abs(dif_psi_norm) <= huber_loss_delta:
            temp_psi = 0.5 * dif_psi_norm ** 2
        else:
            temp_psi = huber_loss_delta * (np.abs(dif_psi_norm) - 0.5 * huber_loss_delta)

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


@njit(parallel=True)
def compute_log_sum(n_sim: int,
                    n_samples: int,
                    param_samples: np.ndarray,
                    param_samples_cov: np.ndarray,  # param_samples_cov = param_samples if no covariates are given
                    beta: np.ndarray,  # beta is the mean of the population distribution
                    psi_inverse: np.ndarray,  # psi_inverse is the inverse covariance of the population distribution
                    prior_mean: np.ndarray = None,  # (only really needed for gaussian prior)
                    prior_cov_inverse: Optional[np.ndarray] = None,  # None if uniform prior
                    huber_loss_delta: Optional[float] = None  # 1.5 times the median of the standard deviations
                    ) -> np.ndarray:
    """compute log-sum-exp of second term in objective function with numba"""
    if huber_loss_delta is not None:
        # precompute cholesky decomposition of psi_inverse
        cholesky_psi = np.linalg.cholesky(psi_inverse)
    else:
        cholesky_psi = None

    # each evaluation of the objective function is independent of the others, so we can parallelize
    expectation_approx = np.zeros((n_sim, n_samples))
    for sim_idx in prange(n_sim):
        for sample_idx in prange(n_samples):
            # compute individual-specific contribution to expectation value
            expectation_approx[sim_idx, sample_idx] = second_term_gaussian(log_phi=param_samples[sim_idx, sample_idx],
                                                                           log_phi_cov=param_samples_cov[
                                                                               sim_idx, sample_idx],
                                                                           beta=beta,
                                                                           psi_inverse=psi_inverse,
                                                                           prior_mean=prior_mean,
                                                                           prior_cov_inverse=prior_cov_inverse,
                                                                           huber_loss_delta=huber_loss_delta,
                                                                           cholesky_psi=cholesky_psi)

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
                 correlation_penalty: Optional[float] = None,
                 huber_loss_delta: Optional[float] = None,
                 prior_type: str = 'normal',
                 prior_bounds: Optional[np.ndarray] = None):
        """

        :param model_name: name of model
        :param prior_mean: numpy array of prior means
        :param prior_std: numpy array of prior standard deviations (only needed for gaussian prior)
        :param param_samples: numpy array of parameter samples (can be updated later via update_param_samples)
        :param covariance_format: either 'diag' or 'cholesky'
        :param covariates: numpy array of covariates
        :param covariate_mapping: function that maps parameters, covariates to distributional parameters
        :param correlation_penalty: l1 penalty for correlations
        :param huber_loss_delta: delta for huber loss (e.g. 1.5 times the median of the standard deviations,
                        penalizes outliers more strongly than a normal distribution)
        :param prior_type: either 'normal' or 'uniform'
        :param prior_bounds: numpy array of uniform prior bounds (only needed for uniform prior)
        """

        self.model_name = model_name
        self.param_samples = param_samples if param_samples is not None else np.empty((1, 1, 1))
        self.covariance_format = covariance_format
        if covariance_format != 'diag' and covariance_format != 'cholesky':
            raise ValueError(f'covariance_format must be either "diag" or "cholesky", but is {covariance_format}')
        self.correlation_penalty = correlation_penalty
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
            # add constant to make approximations more comparable
            self.constant_prior_term += -prior_mean.size * 0.5 * np.log(2 * np.pi)
        else:
            raise ValueError(f'prior_type must be either "gaussian" or "uniform", but is {prior_type}')

        # some constants
        self.n_sim, self.n_samples, n_posterior_params = param_samples.shape
        # number of parameters for the population distribution (does not include covariates)
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

    def update_param_samples(self, param_samples: np.ndarray) -> None:
        """update parameter samples, everything else stays the same"""
        self.param_samples = param_samples
        self.n_sim, self.n_samples, n_posterior_params = param_samples.shape
        self.log_n_samples = np.log(self.n_samples)
        assert n_posterior_params == self.prior_mean.size, 'number of posterior parameters does not match prior'

        if self.covariates is not None:
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
            param_samples_cov=self.param_samples_cov,  # param_samples_cov = param_samples if no covariates are given
            beta=beta,  # beta is the mean of the population distribution
            psi_inverse=psi_inverse,  # psi_inverse is the inverse covariance of the population distribution
            prior_mean=self.prior_mean,  # (only really needed for gaussian prior)
            prior_cov_inverse=self.prior_cov_inverse,  # None if uniform prior
            huber_loss_delta=self.huber_loss_delta  # optional, only needed for huber loss
        )

        # log-sum-exp, computes the Monte Carlo approximation of the expectation value
        log_sum_exp = logsumexp(expectation_approx, axis=1)  # more stable

        # take sum again over cells/individuals
        expectation_log_sum = np.sum(log_sum_exp)
        return expectation_log_sum

    # define objective function to minimize parameters
    def __call__(self, vector_params: np.ndarray) -> float:
        # get parameter vectors
        # vector_params = (beta, psi_inverse_vector, covariates_params)
        beta = vector_params[:self.param_dim]
        psi_inverse_vector = vector_params[self.param_dim:-self.n_covariates]
        psi_inverse = self.get_inverse_covariance(psi_inverse_vector)
        covariates_params = vector_params[-self.n_covariates:]

        # include covariates
        if self.covariates is not None:
            # if no covariates are given, param_samples_cov = param_samples
            self.param_samples_cov = self.covariate_mapping(param_samples=self.param_samples,
                                                            covariates=self.covariates,
                                                            covariate_params=covariates_params)

        # compute parts of loss
        # gaussian prior: constant_prior_term is _log_sqrt_det of prior covariance
        # uniform prior: constant_prior_term is log of (b-a)^n and constant from gaussian population density
        part_one = self.n_sim * (self._log_sqrt_det(psi_inverse) - self.log_n_samples + self.constant_prior_term)
        expectation_log_sum = self._helper_log_sum(beta, psi_inverse)

        # compute negative log-likelihood
        nll = -(part_one + expectation_log_sum)

        # add l1 penalty for correlations
        if self.correlation_penalty is not None:
            # the first param_dim entries of psi_inverse_vector are the diagonal entries of psi_inverse
            nll += self.correlation_penalty * np.sum(np.abs(psi_inverse_vector[self.param_dim:]))
        return nll

    @staticmethod
    def _log_sqrt_det(matrix: np.ndarray) -> np.ndarray:
        """compute log of square root of determinant of matrix in a numerically stable way"""
        # determinant = np.prod(eig_values)
        # log(sqrt(determinant)) = 0.5 * log(determinant)
        # slogdet returns sign and log(abs(determinant)), which is more stable
        # but determinant is positive anyhow since matrix is positive definite
        log_s_det = 0.5 * np.linalg.slogdet(matrix).logabsdet
        return log_s_det

    def get_inverse_covariance(self, psi_inverse_vector: np.ndarray) -> np.ndarray:
        if self.covariance_format == 'diag':
            # psi = log of diagonal entries since entries must be positive
            psi_inverse = np.diag(np.exp(psi_inverse_vector))
        else:
            # matrix is 'cholesky'
            # vector to triangular matrix
            psi_inverse_lower = np.zeros((self.param_dim, self.param_dim))
            psi_inverse_lower[np.diag_indices(self.param_dim)] = 1
            psi_inverse_lower[np.tril_indices(self.param_dim, k=-1)] = psi_inverse_vector[self.param_dim:]

            psi_inverse_diag = np.diag(np.exp(psi_inverse_vector[:self.param_dim]))
            psi_inverse = psi_inverse_lower.dot(psi_inverse_diag).dot(psi_inverse_lower.T)
        return psi_inverse

    def get_covariance(self, psi_inverse_vector: np.ndarray) -> np.ndarray:
        inverse_covariance = self.get_inverse_covariance(psi_inverse_vector)
        psi = np.linalg.inv(inverse_covariance)
        return psi

    def get_inverse_covariance_vector(self, psi: np.ndarray) -> np.ndarray:
        if self.covariance_format == 'diag':
            # psi = log of diagonal entries since entries must be positive
            psi_inv_vector = -np.log(psi.diagonal())
        else:
            # 'cholesky'
            # triangular matrix to vector
            psi_inv = np.linalg.inv(psi)
            lu, d, perm = ldl_decomposition(psi_inv)
            psi_inv_lower = lu[perm, :][np.tril_indices(self.param_dim, k=-1)]
            psi_inv_vector = np.concatenate((np.log(d.diagonal()), psi_inv_lower))
        return psi_inv_vector
