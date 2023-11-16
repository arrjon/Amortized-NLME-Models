#!/usr/bin/env python
# coding: utf-8

import itertools
from typing import Optional, Union
import numpy as np
import pandas as pd
from bayesflow.amortizers import AmortizedPosterior
from heatmap import corrplot

from inference.base_nlme_model import NlmeBaseAmortizer


def create_boundaries_from_prior(
        prior_mean: np.ndarray,
        prior_std: np.ndarray,
        prior_type: str,
        prior_bounds: Optional[np.ndarray] = None,
        covariates_bounds: Optional[np.ndarray] = None,
        boundary_width_from_prior: float = 3,
        covariance_format: str = 'diag') -> np.ndarray:
    """Create boundaries for optimization problem from prior mean and standard deviation."""
    if prior_type == 'uniform':
        assert prior_bounds is not None, 'prior_bounds must be given for uniform prior'
        lb_means = prior_bounds[:, 0]
        ub_means = prior_bounds[:, 1]
    else:
        # gaussian prior # TODO: how should bound change with covariates?
        lb_means = prior_mean - boundary_width_from_prior * prior_std
        ub_means = prior_mean + boundary_width_from_prior * prior_std

    # add boundaries for variance parameters (log-inverse-transformed)
    max_std = np.max(boundary_width_from_prior * prior_std)
    mean_std = np.mean(prior_std)
    lb_var = np.log(1. / mean_std) - max_std
    ub_var = np.log(1. / mean_std) + max_std

    # concatenate mean and variance boundaries
    lower_bound = np.concatenate((lb_means, np.ones_like(lb_means)*lb_var))
    upper_bound = np.concatenate((ub_means, np.ones_like(ub_means)*ub_var))

    if covariance_format == 'cholesky':
        # add lower triangular part of covariance matrix
        n_corr = prior_mean.size * (prior_mean.size - 1) // 2  # number of correlation parameters
        lower_bound = np.concatenate((lower_bound, -boundary_width_from_prior * np.ones(n_corr)))
        upper_bound = np.concatenate((upper_bound, boundary_width_from_prior * np.ones(n_corr)))

    if covariates_bounds is not None:
        # add boundaries for covariates
        lower_bound = np.concatenate((lower_bound, covariates_bounds[:, 0]))
        upper_bound = np.concatenate((upper_bound, covariates_bounds[:, 1]))

    assert (lower_bound - upper_bound < 0).all(), 'lower bound must be smaller than upper bound'
    return np.stack((lower_bound, upper_bound))


def create_param_names_opt(dim: int,
                           param_names: list,
                           ):
    # create parameter names for optimization problem
    param_names_opt = []
    for i, name in enumerate(param_names):
        if i < dim:
            param_names_opt.append(name)
        elif i < 2 * dim:
            # only change variance params
            param_names_opt.append('$\log$ (' + name + ')^{-1}')
        elif i < 2 * dim + dim * (dim - 1) // 2:
            # only change correlation params
            param_names_opt.append('inv_' + name)
        else:
            param_names_opt.append(name)
    return param_names_opt


def create_mixed_effect_model_param_names(param_names: list,
                                          cov_type: str,
                                          covariates_names: Optional[list] = None
                                          ) -> list:
    """create parameter names for mixed effect model (mean, variance, correlation)"""
    pop_param_names = ['pop-' + name for name in param_names]
    var_param_names = ['var-' + name for name in param_names]
    mixed_effect_params_names = pop_param_names + var_param_names

    if cov_type == 'cholesky' and len(mixed_effect_params_names) == len(param_names) * 2:
        # add parameter names in exact the same order as in the covariance matrix
        mixed_effect_params_names += create_correlation_names(param_names)
    if covariates_names is not None:
        mixed_effect_params_names += covariates_names
    return mixed_effect_params_names


def create_correlation_names(param_names: list) -> list:
    """make all possible combinations of parameter names in exact the same order as in the covariance matrix"""
    combinations = list(itertools.combinations(param_names, 2))
    # create upper triangular matrix of parameter names
    psi_inverse_upper = np.chararray((len(param_names), len(param_names)), itemsize=100, unicode=True)
    psi_inverse_upper[np.diag_indices(len(param_names))] = "1"
    psi_inverse_upper[np.triu_indices(len(param_names), k=1)] = [f"corr_{x}_{y}" for x, y in combinations]
    # extract lower triangular matrix of parameter names, so that they are ordered as in the covariance matrix
    corr_names = list(psi_inverse_upper.T[np.tril_indices(len(param_names), k=-1)])
    return corr_names


def get_high_correlation_pairs(corr_df: pd.DataFrame,
                               mixed_effect_params_names: list,
                               threshold: float) -> list:
    """Find parameter pairs with correlation above the threshold"""
    abs_corr_matrix = corr_df.abs()
    high_corr_pairs = abs_corr_matrix[abs_corr_matrix > threshold].unstack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs < 1].sort_values(ascending=False)

    # find indices corresponding to pairs with high correlations
    high_corr_pairs_index = []
    # high_corr_pairs_names = []
    for x, y in high_corr_pairs.index:
        name = f'corr_{x}_{y}'
        high_corr_pairs_index += [p_i for p_i, p_name in enumerate(mixed_effect_params_names) if name == p_name]
        # high_corr_pairs_names += [p_name for p_i, p_name in enumerate(mixed_effect_params_names) if name == p_name]
    return high_corr_pairs_index


def analyse_correlation_in_posterior(model: NlmeBaseAmortizer,
                                     mixed_effect_params_names: list,
                                     obs_data: Union[list[np.ndarray], np.ndarray],
                                     threshold_corr: float = 0.) -> list:
    """Analyse correlation in posterior samples by computing the correlation matrix and plotting it as heatmap.
    Returns indices of parameter pairs with correlation above the threshold.
    """
    param_samples = model.draw_posterior_samples(data=obs_data, n_samples=100)
    param_median = np.median(param_samples, axis=1)
    median_df = pd.DataFrame(param_median, columns=model.param_names)
    corr_df = median_df.corr()

    corrplot(corr_df, size_scale=300)
    high_corr_pairs_index = get_high_correlation_pairs(corr_df=corr_df,
                                                       mixed_effect_params_names=mixed_effect_params_names,
                                                       threshold=threshold_corr)
    return high_corr_pairs_index


def create_fixed_params(fix_names: list, fixed_values: list,
                        params_list: list,
                        fix_low_correlation: bool = False,
                        high_correlation_pairs: Optional[list] = None) -> (np.ndarray, np.ndarray):
    """Create list of fixed parameters (names given) and their values.
    If desired, also fix all correlation parameters to zero, except for the ones in high_correlation_pairs
    (assumes that the correlation parameters start with the name 'corr_').
    """
    assert len(fix_names) == len(fixed_values), 'fix_names and fix_values must have the same length'

    fixed_indices = []
    fixed_values_out = []
    for n_i, name in enumerate(fix_names):
        fixed_indices.append(params_list.index(name))
        if name.startswith('var-'):
            # variance parameters are log-inverse-transformed
            if fixed_values[n_i] == 0:
                # if variance is zero, set to lower bound (or upper bound of log-inverse-transformed)
                temp_val = np.Inf
            else:
                temp_val = np.log(1 / fixed_values[n_i])
        else:
            temp_val = fixed_values[n_i]
        fixed_values_out.append(temp_val)

    if fix_low_correlation:
        assert high_correlation_pairs is not None, 'high_correlation_pairs must be given'
        for n_i, name in enumerate(params_list):
            if name.startswith('corr_') and n_i not in high_correlation_pairs:
                fixed_indices.append(n_i)
                fixed_values_out.append(0)

    # make sure that fixed indices are unique
    fixed_indices, unique_indices = np.unique(np.array(fixed_indices), return_index=True)
    fixed_values_out = np.array(fixed_values_out)[unique_indices]
    return fixed_indices, fixed_values_out


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

