#!/usr/bin/env python
# coding: utf-8

from typing import Union, Optional
import numpy as np
import pandas as pd
from pypesto import Result, Objective, Problem, FD, HistoryOptions, optimize, engine
from tqdm import tqdm

from inference.base_nlme_model import NlmeBaseAmortizer
from inference.helper_functions import create_param_names_opt, create_boundaries_from_prior
from inference.nlme_objective import ObjectiveFunctionNLME


def run_population_optimization(
        individual_model: NlmeBaseAmortizer,
        data: Union[np.ndarray, list[np.ndarray]],
        param_names: list,
        cov_type: str = 'diag',
        n_multi_starts: int = 1,
        n_samples_opt: int = 100,
        param_bounds: Optional[np.array] = None,
        covariates: Optional[np.ndarray] = None,
        covariate_mapping: Optional[callable] = None,
        n_covariates_params: int = 0,
        covariates_bounds: Optional[np.ndarray] = None,
        x_fixed_indices: Optional[np.ndarray] = None,
        x_fixed_vals: Optional[np.ndarray] = None,
        file_name: Optional[str] = None,
        verbose: bool = False,
        trace_record: bool = False,
        pesto_multi_processes: int = 1,
        pesto_optimizer: optimize.Optimizer = optimize.ScipyOptimizer(),
        result: Optional[Result] = None
) -> (Result, ObjectiveFunctionNLME):
    # create objective function
    obj_fun_amortized = ObjectiveFunctionNLME(model_name=individual_model.name,
                                              prior_mean=individual_model.prior_mean,
                                              prior_std=individual_model.prior_std,
                                              covariance_format=cov_type,
                                              covariates=covariates,
                                              covariate_mapping=covariate_mapping,
                                              n_covariates_params=n_covariates_params,
                                              prior_type=individual_model.prior_type,
                                              # for uniform prior
                                              prior_bounds=individual_model.prior_bounds if hasattr(individual_model,
                                                                                                    'prior_bounds') else None,
                                              )
    # set up pyPesto
    param_names_opt = create_param_names_opt(dim=individual_model.amortizer.latent_dim,
                                             cov_type=cov_type,
                                             param_names=param_names)
    # create bounds if none are given
    if param_bounds is None:
        # automatically create boundaries
        param_bounds = create_boundaries_from_prior(
            prior_mean=individual_model.prior_mean,
            prior_std=individual_model.prior_std,
            prior_type=individual_model.prior_type,
            prior_bounds=individual_model.prior_bounds if hasattr(individual_model, 'prior_bounds') else None,
            covariates_bounds=covariates_bounds,  # only used if covariates are used
            covariance_format=cov_type)

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
        param_samples = individual_model.draw_posterior_samples(data=data, n_samples=n_samples_opt)
        # update objective function with samples
        obj_fun_amortized.update_param_samples(param_samples=param_samples)

        pesto_objective = FD(obj=Objective(fun=obj_fun_amortized, x_names=param_names_opt))

        # create pypesto problem
        pesto_problem = Problem(objective=pesto_objective,
                                lb=param_bounds[0, :], ub=param_bounds[1, :],
                                x_names=param_names_opt,
                                x_scales=['log'] * len(param_names_opt),
                                x_fixed_indices=x_fixed_indices,
                                x_fixed_vals=x_fixed_vals)
        if verbose and run_idx == 0:
            pesto_problem.print_parameter_summary()
            df_fixed = pd.DataFrame(pesto_problem.x_fixed_vals,
                                    index=np.array(param_names_opt)[pesto_problem.x_fixed_indices],
                                    columns=['fixed value'])
            print(df_fixed)

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
    # always use more samples for final evaluation
    # MC integration error reduces with sqrt(1/n_samples)
    param_samples = individual_model.draw_posterior_samples(data=data, n_samples=n_samples_opt * 10)
    # update objective function with samples
    obj_fun_amortized.update_param_samples(param_samples=param_samples)

    result_list = result.optimize_result.list.copy()
    for res in result_list:
        res['fval'] = obj_fun_amortized(res['x'])
    setattr(result.optimize_result, "list", result_list)
    result.optimize_result.sort()

    # update objective function with fewer samples (for faster evaluation, e.g. profiling)
    param_samples = individual_model.draw_posterior_samples(data=data, n_samples=n_samples_opt)
    obj_fun_amortized.update_param_samples(param_samples=param_samples)
    return result, obj_fun_amortized
