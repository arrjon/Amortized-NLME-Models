#!/usr/bin/env python
# coding: utf-8

from typing import Union
import numpy as np
from numba import njit


@njit
def log_likelihood_multiplicative_noise(log_measurements: np.ndarray,
                                        log_simulations: np.ndarray,
                                        sigmas: Union[float, np.ndarray]) -> float:
    # compute the log-likelihood for multiplicative normal noise (log-normal distribution)
    dif_sum = np.sum(((log_measurements - log_simulations) / sigmas)**2)
    if isinstance(sigmas, float):
        # needed for njit, cannot sum over float
        log_det_sigma = np.log(sigmas**2)
    else:
        log_det_sigma = np.sum(np.log(sigmas**2))
    # log_measurement.size = n_measurements + n_observables, len(log_measurement) = n_measurements
    llh = (-0.5 * log_measurements.size * np.log(2 * np.pi) - 0.5 * len(log_measurements) * log_det_sigma
           - log_measurements.sum() - 0.5 * dif_sum)
    return llh


@njit
def log_likelihood_additive_noise(measurements: np.ndarray,
                                  simulations: np.ndarray,
                                  sigmas: Union[float, np.ndarray]) -> float:
    # compute the log-likelihood for additive normal noise, proportionality might be captured in sigma already
    # normal distribution
    dif_sum = np.sum(((measurements - simulations) / sigmas)**2)
    if isinstance(sigmas, float):
        # needed for njit, cannot sum over float
        log_det_sigma = np.log(sigmas ** 2)
    else:
        log_det_sigma = np.sum(np.log(sigmas ** 2))
    llh = -0.5 * measurements.size * np.log(2 * np.pi) - 0.5 * len(measurements) * log_det_sigma - 0.5 * dif_sum
    return llh
