from typing import Union

import numpy as np
import pandas as pd
from scipy.linalg import ldl as ldl_decomposition

from bayesflow.amortizers import AmortizedPosterior


def create_param_names_opt(bf_amortizer: AmortizedPosterior,
                           param_names: list,
                           n_covariates: int = 0,
                           multi_experiment: bool = False,
                           ):
    if not multi_experiment:
        dim = bf_amortizer.latent_dim
    else:
        dim = bf_amortizer.latent_dim + 1
    dim += n_covariates

    # create parameter names for optimization problem
    param_names_opt = []
    for i, name in enumerate(param_names):
        if i >= dim:
            continue  # only change population params
        param_names_opt.append(name)

    for i, name in enumerate(param_names):
        if i < dim or i >= 2 * dim:
            continue  # only change variance params
        param_names_opt.append('$\log$ (' + name + ')^{-1}')

    for i, name in enumerate(param_names):
        if i < 2 * dim:
            continue  # only change correlation params
        param_names_opt.append(name)
    return param_names_opt


def compute_error_estimate(results_transformed: np.ndarray,
                           true_pop_parameters: np.ndarray,
                           relative_error: bool = False,
                           epsilon: float = 0.0001,
                           small_model: bool = False) -> np.ndarray:
    if relative_error:
        rel_error = np.mean((results_transformed - true_pop_parameters) ** 2 / (np.abs(true_pop_parameters) + epsilon),
                            axis=1)
    else:
        rel_error = np.mean((results_transformed - true_pop_parameters) ** 2, axis=1)
    # handle the bimodal distributions, both modes are equally acceptable
    if small_model:
        other_mode = true_pop_parameters.copy()
        other_mode[[0, 1, 6, 7]] = other_mode[[1, 0, 7, 6]]
        if relative_error:
            rel_error_2 = np.mean((results_transformed - other_mode) ** 2 / (np.abs(other_mode) + epsilon),
                                  axis=1)
        else:
            rel_error_2 = np.mean((results_transformed - other_mode) ** 2, axis=1)
        rel_error = np.minimum(rel_error, rel_error_2)
    return rel_error

