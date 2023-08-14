import numpy as np
from scipy.linalg import ldl as ldl_decomposition
from tqdm import tqdm
from typing import Union, Optional
from functools import partial

# optimization
from pypesto import Result, Objective, Problem, FD, HistoryOptions, optimize, startpoint, engine
from sklearn.covariance import GraphicalLassoCV

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


class PosteriorStartpoints(startpoint.FunctionStartpoints):
    """Generate starting parameters by estimating the modes of the entity-specific posterior distributions."""

    def __init__(
            self,
            param_samples: Union[np.ndarray, list[np.ndarray]],
            pesto_problem: Problem,
            use_guesses: bool = True,
            check_fval: bool = False,
            check_grad: bool = False,
            noise_on_param: int = 0,
            covariance_format: str = 'diag',
            param_names_opt: Optional[list[str]] = None,
            multi_experiment: bool = False,
            n_covariates: int = 0,
    ):
        self.x_fixed_indices = pesto_problem.x_fixed_indices
        self.noise_on_param = noise_on_param
        self.covariance_format = covariance_format
        if covariance_format != 'diag' and covariance_format != 'cholesky':
            raise ValueError(f'covariance_format must be either "diag" or "cholesky", but is {covariance_format}')
        if not multi_experiment:
            # generally applicable
            self.param_samples = param_samples
            self.n_covariates = n_covariates
        else:
            # particular for this two sample problem
            self.param_samples_eGFP = param_samples[0]
            self.param_samples_d2eGFP = param_samples[1]
            self.pesto_problem = pesto_problem
            self.param_names_opt = param_names_opt
            self.x_fixed_indices = []  # so we can use the same function for both experiments
            self.x_fixed_indices_full = pesto_problem.x_fixed_indices
            self.dim = pesto_problem.dim_full

        super().__init__(
            function=self.get_posterior_starting_values if not multi_experiment
            else self.get_posterior_starting_values_multi,
            use_guesses=use_guesses,
            check_fval=check_fval,
            check_grad=check_grad,
        )

    def get_posterior_starting_values(self,
                                      n_starts: int,
                                      lb: Optional[np.ndarray],
                                      ub: Optional[np.ndarray]) -> np.ndarray:
        """
        Generate starting parameters for multi-start optimization by sampling from the posterior distribution.
        """
        # compute median of density for each parameter and for each individual
        param_median_individual = np.zeros((self.param_samples.shape[0], self.param_samples.shape[2]))
        for individual_idx, ind_param in enumerate(self.param_samples):
            param_median_individual[individual_idx] = np.median(ind_param, axis=0)

        # estimate mean and covariance (log of inverse)
        # finds sparse inverse covariance (enforcing sparsity on correlations)
        cov_model = GraphicalLassoCV()
        cov_model.fit(param_median_individual)
        param_estimated_mean = cov_model.location_  # should be same as np.mean
        param_estimated_mean = np.concatenate((param_estimated_mean, np.zeros(self.n_covariates)))

        if self.covariance_format == 'diag':
            param_estimated_cov_inv = np.log(cov_model.get_precision().diagonal())
            param_estimated_cov_inv = np.concatenate((param_estimated_cov_inv,
                                                      np.log(100. / np.ones(self.n_covariates))))
        else:
            lu, d, perm = ldl_decomposition(cov_model.get_precision())
            psi_inv_lower = lu[perm, :][np.tril_indices(param_estimated_mean.size, k=-1)]
            param_estimated_cov_inv = np.concatenate((np.log(d.diagonal()), psi_inv_lower))

            if self.n_covariates > 0:
                raise NotImplementedError

        # get starting parameters which are not fixed parameters
        start_idv = np.concatenate((param_estimated_mean, param_estimated_cov_inv))
        starting_params = np.array([x0 for x_idx, x0 in enumerate(start_idv)
                                    if x_idx not in self.x_fixed_indices])

        # create starting parameters and add noise
        starting_params = np.repeat(starting_params[np.newaxis], repeats=n_starts, axis=0)
        starting_params += np.random.normal(0, self.noise_on_param, size=starting_params.shape)

        # apply bounds
        if lb is not None:
            starting_params = np.maximum(starting_params, lb)
        if ub is not None:
            starting_params = np.minimum(starting_params, ub)
        return starting_params

    def get_posterior_starting_values_multi(self,
                                            n_starts: int,
                                            lb: np.ndarray,
                                            ub: np.ndarray):
        """
        Generate starting parameters for multi-start optimization by sampling from the posterior distribution.
        """
        # compute median of density for each parameter and for each individual
        noise_on_param = self.noise_on_param
        self.noise_on_param = 0

        # compute starting values for eGFP
        self.param_samples = self.param_samples_eGFP
        start_idv_eGFP = self.get_posterior_starting_values(1, None, None).flatten()
        # compute starting values for d2eGFP
        self.param_samples = self.param_samples_d2eGFP
        start_idv_d2eGFP = self.get_posterior_starting_values(1, None, None).flatten()

        # combine eGFP and d2eGFP starting parameters depending on the model
        param_idx_egfp = [i_n for i_n, name in enumerate(self.param_names_opt) if 'd2eGFP' not in name]
        param_idx_d2egfp = [i_n for i_n, name in enumerate(self.param_names_opt) if ('d2eGFP' in name
                                                                                     or 'eGFP' not in name)
                            and not 'corr_$\\gamma_{eGFP}$_$\\gamma_{d2eGFP}$' == name]
        param_idx_both = [i_n for i_n, name in enumerate(self.param_names_opt) if 'eGFP' not in name]
        start_idv = np.zeros(self.dim)
        start_idv[param_idx_egfp] = start_idv_eGFP
        start_idv[param_idx_d2egfp] += start_idv_d2eGFP
        start_idv[param_idx_both] *= 0.5

        # before we have all parameters in one vector, we need to know which parameters are fixed
        include_indices = [x_idx for x_idx in range(start_idv.size)
                           if x_idx not in self.x_fixed_indices_full]
        starting_params = start_idv[include_indices]

        # add some noise on estimates to get more variability
        starting_params = np.repeat(starting_params[np.newaxis], repeats=n_starts, axis=0)
        starting_params += np.random.normal(0, noise_on_param, size=starting_params.shape)

        # apply bounds
        if lb is not None:
            starting_params = np.maximum(starting_params, lb)
        if ub is not None:
            starting_params = np.minimum(starting_params, ub)
        return starting_params


def run_population_optimization(
        bf_amortizer: AmortizedPosterior,
        data: [np.ndarray, list],
        param_names: list,
        objective_function: Union[ObjectiveFunctionNLME, list[ObjectiveFunctionNLME]],
        sample_posterior: callable,
        n_multi_starts: int = 1,
        noise_on_start_param: int = 0,
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
        pesto_optimizer: optimize.Optimizer = optimize.ScipyOptimizer(),  # FidesOptimizer
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

        # build starting function
        if multi_experiment:
            param_samples = [objective_function[0].param_samples, objective_function[1].param_samples]
            covariance_format = objective_function[0].covariance_format
            pesto_startpoint = PosteriorStartpoints(param_samples=param_samples,
                                                    pesto_problem=pesto_problem,
                                                    noise_on_param=noise_on_start_param,
                                                    covariance_format=covariance_format,
                                                    param_names_opt=param_names_opt,
                                                    multi_experiment=multi_experiment)
        else:
            param_samples = objective_function.param_samples
            covariance_format = objective_function.covariance_format
            pesto_startpoint = PosteriorStartpoints(param_samples=param_samples,
                                                    pesto_problem=pesto_problem,
                                                    noise_on_param=noise_on_start_param,
                                                    covariance_format=covariance_format,
                                                    param_names_opt=param_names_opt,
                                                    n_covariates=objective_function.n_covariates,
                                                    multi_experiment=multi_experiment)

        # Run optimizations for different starting values
        result = optimize.minimize(
            problem=pesto_problem,
            result=result,
            optimizer=pesto_optimizer,
            startpoint_method=pesto_startpoint,
            n_starts=n_starts_per_run,
            engine=pesto_engine,
            history_options=history_options,
            progress_bar=False if pesto_multi_processes <= 1 else verbose,
            filename=file_name if file_name is not None else None,
            overwrite=True if file_name is not None else False)
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
