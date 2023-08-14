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


def transform_pesto_results(result_list: list, n_pop_params: int, cov_type='diag'):
    # transform results to log-normal population parameters
    result_list_transformed = np.zeros((len(result_list), result_list[0].size))
    for l_idx, res_l in enumerate(result_list):
        if cov_type == 'diag':
            beta = res_l[:n_pop_params]
            psi_diag = np.exp(-res_l[n_pop_params:])
            result_list_transformed[l_idx] = np.concatenate((beta, psi_diag))
        elif cov_type == 'cholesky':
            beta = res_l[:n_pop_params]

            # get psi inverse from lower triangular part
            psi_inverse_lower = np.zeros((beta.size, beta.size))
            psi_inverse_lower[np.diag_indices(beta.size)] = 1
            psi_inverse_lower[np.tril_indices(beta.size, k=-1)] = res_l[beta.size * 2:]

            psi_inverse_diag = np.diag(np.exp(res_l[beta.size:beta.size * 2]))
            psi_inverse = psi_inverse_lower.dot(psi_inverse_diag).dot(psi_inverse_lower.T)

            # invert psi
            psi = np.linalg.inv(psi_inverse)

            # transform back to vector
            lu, d, perm = ldl_decomposition(psi)
            psi_lower = lu[perm, :][np.tril_indices(n_pop_params, k=-1)]
            result_list_transformed[l_idx] = np.concatenate((beta, d.diagonal(), psi_lower))
    return result_list_transformed


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


def load_single_cell_data(file_name: str, number_data_points: int = None, real_data: bool = False) -> np.ndarray:
    # excel=False -> csv format

    if real_data:
        # real data
        data = pd.read_excel(f'data/froehlich_eGFP/{file_name}.xlsx', index_col=0, header=None)
    else:
        # synthetic data
        data = pd.read_csv(f'data/synthetic/{file_name}.csv', index_col=0, header=0)

    # convert to right format
    data.index = data.index / 60 / 60  # convert to hours
    data.drop(index=data.index[data.index > 30], inplace=True)

    # format for BayesFlow
    n_real_cells = data.columns.shape[0]
    n_time_points = data.index.shape[0]
    data = np.log(data.values.T).reshape(n_real_cells, n_time_points, 1)

    if number_data_points is not None:
        return data[:number_data_points]

    return data


def load_multi_experiment_data(data_points: int = 50,
                               load_eGFP: bool = True,
                               load_d2eGFP: bool = True) -> Union[np.ndarray, list[np.ndarray]]:
    names_eGFP = ['20160427_mean_eGFP', '20160513_mean_eGFP', '20160603_mean_eGFP']
    names_d2eGFP = ['20160427_mean_d2eGFP', '20160513_mean_d2eGFP', '20160603_mean_d2eGFP']

    if not load_eGFP and not load_d2eGFP:
        raise ValueError("At least one of the two options ('load_eGFP', 'load_d2eGFP') has to be True.")
    data_eGFP = None  # only to silence warning

    # load data
    if load_eGFP:
        data_list_eGFP = []
        for name in names_eGFP:
            data_list_eGFP.append(load_single_cell_data(file_name=name,
                                                        number_data_points=data_points,
                                                        real_data=True))
        # concatenate data: shape for BayesFlow
        data_eGFP = np.concatenate(data_list_eGFP, axis=0)
        if not load_d2eGFP:
            return data_eGFP

    data_list_d2eGFP = []
    for name in names_d2eGFP:
        data_list_d2eGFP.append(load_single_cell_data(file_name=name,
                                                      number_data_points=data_points,
                                                      real_data=True))
    # concatenate data: shape for BayesFlow
    data_d2eGFP = np.concatenate(data_list_d2eGFP, axis=0)
    if not load_eGFP:
        return data_d2eGFP

    return [data_eGFP, data_d2eGFP]


def custom_presimulation_loader(file_path: str, n_obs: int = 180, batch_size: int = 128) -> list[dict]:
    """
    Custom loader for the presimulation data (large föhlich model directly generated in julia.
    outdated
    """
    # Read the prior draws and the simulated data
    prior = np.loadtxt(f'{file_path}/prior_fro_large.csv', delimiter=",", skiprows=1)
    simulations = np.loadtxt(f'{file_path}/simulations_fro_large.csv', delimiter=",", skiprows=1).T
    simulations = simulations.reshape(prior.shape[0], n_obs, 1)

    # Calculate the number of iterations per epoch
    iterations_per_epoch = simulations.shape[0] // batch_size

    # Create a list of dictionaries, each dictionary contains the prior draws and the simulated data
    epoch_data = []
    for it in range(iterations_per_epoch):
        simulation_dict = {
            'prior_draws': prior[it * batch_size:(it + 1) * batch_size],
            'sim_data': simulations[it * batch_size:(it + 1) * batch_size]
        }
        epoch_data.append(simulation_dict)
    # Add the last batch if it is not full
    if iterations_per_epoch * batch_size != simulations.shape[0]:
        simulation_dict = {
            'prior_draws': prior[iterations_per_epoch * batch_size:],
            'sim_data': simulations[iterations_per_epoch * batch_size:]
        }
        epoch_data.append(simulation_dict)
    return epoch_data
