#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Optional
from functools import partial

# optimization
from pypesto import Result, Objective, Problem, FD, HistoryOptions, optimize, engine

from inference.nlme_objective import ObjectiveFunctionNLME
from inference.helper_functions import create_param_names_opt, create_boundaries_from_prior
from inference.base_nlme_model import NlmeBaseAmortizer


def run_population_optimization(
        individual_model: NlmeBaseAmortizer,
        data: Union[np.ndarray, list[np.ndarray]],
        param_names: list,
        objective_function: Union[ObjectiveFunctionNLME, list[ObjectiveFunctionNLME]],
        n_multi_starts: int = 1,
        n_samples_opt: int = 100,
        param_bounds: Optional[np.array] = None,
        covariates_bounds: Optional[np.ndarray] = None,
        x_fixed_indices: Optional[np.ndarray] = None,
        x_fixed_vals: Optional[np.ndarray] = None,
        file_name: Optional[str] = None,
        verbose: bool = False,
        trace_record: bool = False,
        multi_experiment: bool = False,
        pesto_multi_processes: int = 1,
        pesto_optimizer: optimize.Optimizer = optimize.ScipyOptimizer(),
        result: Optional[Result] = None
):
    # set up pyPesto
    param_names_opt = create_param_names_opt(bf_amortizer=individual_model.amortizer,
                                             param_names=param_names,
                                             multi_experiment=multi_experiment)
    # create bounds if none are given
    if param_bounds is None:
        # automatically create boundaries
        param_bounds = create_boundaries_from_prior(
            prior_mean=individual_model.prior_mean,
            prior_std=individual_model.prior_std,
            prior_type=individual_model.prior_type,
            prior_bounds=individual_model.prior_bounds if hasattr(individual_model, 'prior_bounds') else None,
            covariates_bounds=covariates_bounds,  # only used if covariates are used
            covariance_format=objective_function.covariance_format)
    if x_fixed_indices is not None and x_fixed_vals is not None:
        too_small_idx = x_fixed_vals < param_bounds[0, x_fixed_indices]
        x_fixed_vals[too_small_idx] = param_bounds[0, x_fixed_indices][too_small_idx]
        too_large_idx = x_fixed_vals > param_bounds[1, x_fixed_indices]
        x_fixed_vals[too_large_idx] = param_bounds[1, x_fixed_indices][too_large_idx]

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
                                                           sample_posterior=individual_model.draw_posterior_samples,
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
                                lb=param_bounds[0, :], ub=param_bounds[1, :],
                                x_names=param_names_opt,
                                x_scales=['log'] * len(param_names_opt),
                                x_fixed_indices=x_fixed_indices,
                                x_fixed_vals=x_fixed_vals)
        if verbose and run_idx == 0:
            pesto_problem.print_parameter_summary()
            df = pd.DataFrame(pesto_problem.x_fixed_vals,
                              index=np.array(param_names_opt)[pesto_problem.x_fixed_indices],
                              columns=['fixed value'])
            print(df)

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
                                                       sample_posterior=individual_model.draw_posterior_samples,
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
                            lb=param_bounds[0, :], ub=param_bounds[1, :],
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
    # create objective function from samples
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
