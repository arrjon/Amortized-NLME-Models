#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from typing import Union, Optional
from functools import partial

# optimization
from pypesto import Result, Objective, Problem, FD, HistoryOptions, optimize, engine

from inference.nlme_objective import ObjectiveFunctionNLME
from inference.helper_functions import create_param_names_opt

from bayesflow.amortizers import AmortizedPosterior


def create_boundaries_from_prior(
        prior_mean: np.ndarray,
        prior_std: np.ndarray,
        n_covariates: int = 0,
        covariates_bound: tuple = (-1, 1),
        boundary_width_from_prior: float = 2.58,
        covariance_format: str = 'diag'
) -> (np.ndarray, np.ndarray):
    param_dim = prior_mean.size
    # define bounds for optimization

    lb_means = prior_mean - boundary_width_from_prior * prior_std
    ub_means = prior_mean + boundary_width_from_prior * prior_std

    lb_var = np.log(1. / prior_std) - boundary_width_from_prior * prior_std
    ub_var = np.log(1. / prior_std) + boundary_width_from_prior * prior_std

    if n_covariates > 0:
        param_dim += n_covariates
        cov_mean_lb = np.ones(n_covariates) * covariates_bound[0]
        cov_mean_ub = np.ones(n_covariates) * covariates_bound[1]
        cov_var_lb = np.ones(n_covariates) * 0.01
        cov_var_ub = np.ones(n_covariates) * 5
        lower_bound = np.concatenate((lb_means, cov_mean_lb, lb_var, np.log(1. / cov_var_ub)))
        upper_bound = np.concatenate((ub_means, cov_mean_ub, ub_var, np.log(1. / cov_var_lb)))

    else:
        lower_bound = np.concatenate((lb_means, lb_var))
        upper_bound = np.concatenate((ub_means,ub_var))

    if covariance_format == 'cholesky':
        # add lower triangular part of covariance matrix
        n_corr = param_dim * (param_dim - 1) // 2  # number of correlation parameters
        lower_bound = np.concatenate((lower_bound, -boundary_width_from_prior * np.ones(n_corr)))
        upper_bound = np.concatenate((upper_bound, boundary_width_from_prior * np.ones(n_corr)))

    assert (lower_bound - upper_bound < 0).all(), 'lower bound must be smaller than upper bound'
    return lower_bound, upper_bound


def run_population_optimization(
        bf_amortizer: AmortizedPosterior,
        data: [np.ndarray, list],
        param_names: list,
        objective_function: Union[ObjectiveFunctionNLME, list[ObjectiveFunctionNLME]],
        sample_posterior: callable,
        n_multi_starts: int = 1,
        n_samples_opt: int = 100,
        lb: np.array = None,
        ub: np.array = None,
        x_fixed_indices: np.ndarray = None,
        x_fixed_vals: np.ndarray = None,
        file_name: str = None,
        verbose: bool = False,
        trace_record: bool = False,
        multi_experiment: bool = False,
        pesto_multi_processes: int = 1,
        pesto_optimizer: optimize.Optimizer = optimize.ScipyOptimizer(),
        result: Optional[Result] = None
):
    # set up pyPesto
    param_names_opt = create_param_names_opt(bf_amortizer=bf_amortizer,
                                             param_names=param_names,
                                             n_covariates=objective_function.n_covariates,
                                             multi_experiment=multi_experiment)

    # bounds
    if lb is None:
        lb = -np.ones(len(param_names_opt)) * np.Inf
    if ub is None:
        ub = np.ones(len(param_names_opt)) * np.Inf

    # save optimizer trace
    history_options = HistoryOptions(trace_record=trace_record)

    if pesto_multi_processes > 1:
        # use multiprocessing, but start serval times with new objective function
        n_runs = n_multi_starts // pesto_multi_processes
        if n_runs == 0:
            # if more processes then starts are given, all use the same objective function
            n_runs = 1
        elif n_runs * pesto_multi_processes < n_multi_starts:
            # if not all starts are used, add one more run
            n_runs += 1
        n_starts_per_run = pesto_multi_processes
        pesto_engine = engine.MultiProcessEngine(pesto_multi_processes)
    else:
        n_runs = n_multi_starts
        n_starts_per_run = 1
        pesto_engine = engine.SingleCoreEngine()

    if pesto_multi_processes >= n_multi_starts:
        print("Warning: pesto_multi_processes >= n_multi_starts. All starts use the same sample from the posterior. "
              "This is not recommended and you should increase 'n_multi_starts'.")

    for run_idx in tqdm(range(n_runs), disable=not verbose, desc='Multi-start optimization'):
        # run optimization for each starting point with different objective functions (due to sampling)
        # if pesto_multi_processes > 1, same objective function is used for all starting points

        # create objective function with samples
        objective_function = create_objective_with_samples(data=data,
                                                           objective_function=objective_function,
                                                           sample_posterior=sample_posterior,
                                                           n_samples_opt=n_samples_opt,
                                                           multi_experiment=multi_experiment)
        # wrap finite difference around objective function
        if multi_experiment:
            # extract indices of param names containing eGFP
            param_idx_egfp = [i_n for i_n, name in enumerate(param_names_opt) if 'd2eGFP' not in name]
            param_idx_d2egfp = [i_n for i_n, name in enumerate(param_names_opt) if
                                'd2eGFP' in name or 'eGFP' not in name]
            pesto_objective = FD(obj=Objective(fun=partial(obj_fun_multi_helper,
                                                           param_idx_egfp=param_idx_egfp,
                                                           param_idx_d2egfp=param_idx_d2egfp,
                                                           obj_fun_amortized=objective_function[0],
                                                           obj_fun_amortized_2=objective_function[1]),
                                               x_names=param_names_opt))
        else:
            pesto_objective = FD(obj=Objective(fun=objective_function, x_names=param_names_opt))

        # create pypesto problem
        pesto_problem = Problem(objective=pesto_objective,
                                lb=lb, ub=ub,
                                x_names=param_names_opt,
                                x_scales=['log'] * len(param_names_opt),
                                x_fixed_indices=x_fixed_indices,
                                x_fixed_vals=x_fixed_vals)
        if verbose and run_idx == 0:
            pesto_problem.print_parameter_summary()

        # Run optimizations for different starting values
        result = optimize.minimize(
            problem=pesto_problem,
            result=result,
            optimizer=pesto_optimizer,
            ids=[f"batch_{run_idx}_{i}" for i in range(n_starts_per_run)],
            n_starts=n_starts_per_run,
            engine=pesto_engine,
            history_options=history_options,
            progress_bar=False if pesto_multi_processes <= 1 else verbose,
            filename=file_name if file_name is not None else None,
            overwrite=True if file_name is not None else False)

    # make final objective value comparable across samples
    objective_function = create_objective_with_samples(data=data,
                                                       objective_function=objective_function,
                                                       sample_posterior=sample_posterior,
                                                       n_samples_opt=n_samples_opt * 100,
                                                       # always use more samples for final evaluation
                                                       # MC integration error reduces with sqrt(1/n_samples)
                                                       multi_experiment=multi_experiment)

    # wrap finite difference around objective function
    if multi_experiment:
        # extract indices of param names containing eGFP
        param_idx_egfp = [i_n for i_n, name in enumerate(param_names_opt) if 'd2eGFP' not in name]
        param_idx_d2egfp = [i_n for i_n, name in enumerate(param_names_opt) if
                            'd2eGFP' in name or 'eGFP' not in name]
        pesto_objective = FD(obj=Objective(fun=partial(obj_fun_multi_helper,
                                                       param_idx_egfp=param_idx_egfp,
                                                       param_idx_d2egfp=param_idx_d2egfp,
                                                       obj_fun_amortized=objective_function[0],
                                                       obj_fun_amortized_2=objective_function[1]),
                                           x_names=param_names_opt))
    else:
        pesto_objective = FD(obj=Objective(fun=objective_function, x_names=param_names_opt))
    pesto_problem = Problem(objective=pesto_objective,
                            lb=lb, ub=ub,
                            x_names=param_names_opt,
                            x_scales=['log'] * len(param_names_opt),
                            x_fixed_indices=x_fixed_indices,
                            x_fixed_vals=x_fixed_vals)
    setattr(result, "problem", pesto_problem)

    result_list = result.optimize_result.list.copy()
    for res in result_list:
        res['fval'] = result.problem.objective(res['x'][result.problem.x_free_indices])
    setattr(result.optimize_result, "list", result_list)
    result.optimize_result.sort()
    return result


def create_objective_with_samples(data: Union[np.ndarray, list],
                                  objective_function: Union[ObjectiveFunctionNLME, list[ObjectiveFunctionNLME]],
                                  sample_posterior: callable,
                                  n_samples_opt: int = 100,
                                  multi_experiment: bool = False,
                                  ) -> Union[ObjectiveFunctionNLME, list[ObjectiveFunctionNLME]]:
    # create objective function with samples
    # sample parameters from amortizer given observed data
    if not multi_experiment:
        param_samples = sample_posterior(data=data, n_samples=n_samples_opt)
        # update objective function with samples
        objective_function.update_param_samples(param_samples=param_samples)
    else:
        # list of samples per experiment, needs adapted objective function
        data_eGFP, data_d2eGFP = data
        param_samples_eGFP = sample_posterior(data=data_eGFP,
                                              n_samples=n_samples_opt)
        param_samples_d2eGFP = sample_posterior(data=data_d2eGFP,
                                                n_samples=n_samples_opt)
        objective_function[0].update_param_samples(param_samples=param_samples_eGFP)
        objective_function[1].update_param_samples(param_samples=param_samples_d2eGFP)
    return objective_function


def obj_fun_multi_helper(params: np.ndarray,
                         param_idx_egfp: list,
                         param_idx_d2egfp: list,
                         obj_fun_amortized: ObjectiveFunctionNLME,
                         obj_fun_amortized_2: ObjectiveFunctionNLME):
    """
    Helper function to evaluate objective function for a multiple experiments setup.
    """
    # compute objective function
    return obj_fun_amortized(params[param_idx_egfp]) + obj_fun_amortized_2(params[param_idx_d2egfp])
