#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the simple Fröhlich NLME Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

from typing import Optional, Union
from functools import partial
import itertools

from inference.base_nlme_model import NlmeBaseAmortizer
from bayesflow.simulation import Simulator


#@jit(nopython=True)
def ode_analytical_sol(t: np.ndarray, delta: float, gamma: float, k_m0_scale: float, t_0: float) -> np.ndarray:
    """
    Solves a first order ordinary differential equation (ODE) with the given parameters.
    The ODE is a model for a system with two state variables, m and p.

    Parameters:
    t (np.ndarray): Time points for which the solution should be evaluated.
    delta (float): Parameter of the ODE.
    gamma (float): Parameter of the ODE.
    k_m0_scale (float): Parameter of the ODE.
    t_0 (float): Initial time point.

    Returns:
    np.ndarray: Solution of the ODE at the provided time points. The array has shape (2, len(t))
        with the first row being the value of m at each time point, and the second row being the
        value of p at each time point.
    """
    m = np.exp(-delta * (t - t_0))
    m[t - t_0 < 0] = 0
    p = k_m0_scale / (delta - gamma) * (np.exp(-gamma * (t - t_0)) - m)
    p[t - t_0 < 0] = 0
    return np.row_stack((m, p))


def measurement(y: np.ndarray, offset: float) -> np.ndarray:
    """
    Applies a measurement function to a given variable.

    Parameters:
    p (np.ndarray): The variable to which the measurement function should be applied.
    offset (float): Offset for the measurement function.

    Returns:
    np.ndarray: The result of applying the measurement function to the input variable.
    """
    p = y[1]
    return np.log(p + offset)


def add_noise(y: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    Adds Gaussian noise to given trajectories.

    Parameters:
    y (np.ndarray): The trajectories to which noise should be added.
    sigmas (np.ndarray): Standard deviation of the Gaussian noise.

    Shapes: Sigmas (n_sim) and y (n_sim,n_obs)

    Returns:
    np.ndarray: The noisy trajectories.
    """
    noise = np.random.normal(loc=0, scale=1, size=y.shape)
    return y + sigmas[:, np.newaxis] * noise


def batch_simulator(param_batch: np.ndarray,
                    n_obs: int,
                    with_noise: bool = True,
                    exp_func: str = 'exp') -> np.ndarray:
    """
    Simulate ODE model

    param_samples: np.ndarray - (#simulations, #parameters) or (#parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)
    exp_func: str - if 'exp' then exp(log_param_samples) is used as parameter samples, otherwise
                    if exp_func == 'power10' then 10**log_param_samples is used as parameter samples

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1) or (#observations, 1)
    """

    # simulate batch
    if len(param_batch.shape) == 1:  # so not (batch_size, params)
        # just a single parameter set
        param_batch = param_batch[np.newaxis, :]
    n_sim = param_batch.shape[0]
    sim_data = np.zeros((n_sim, n_obs), dtype=np.float32)

    if exp_func == 'exp':
        exp_param_batch = np.exp(param_batch)
    elif exp_func == 'power10':
        exp_param_batch = np.power(10, param_batch)
    else:
        raise ValueError('exp_func must be "exp" or "power10"')

    t_points = np.linspace(start=1 / 6, stop=30, num=n_obs, endpoint=True)
    # iterate over batch
    for i, exp_param in enumerate(exp_param_batch):
        # simulate all observations together
        delta, gamma, k_m0_scale, t_0, offset, _ = exp_param
        sol_ana = ode_analytical_sol(t=t_points,
                                     delta=delta,
                                     gamma=gamma,
                                     k_m0_scale=k_m0_scale,
                                     t_0=t_0)
        sim_data[i, :] = measurement(y=sol_ana, offset=offset)

    # add noise for each cell
    if with_noise:
        sim_data = add_noise(sim_data, sigmas=exp_param_batch[:, -1])

    if n_sim == 1:
        # return only one simulation
        return sim_data[0]
    return sim_data[:, np.newaxis]  # add dimension for the channel (n_sim, n_obs, 1)


class FroehlichModelSimple(NlmeBaseAmortizer):
    def __init__(self, name: str = 'SimpleFroehlichModel', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\delta$', '$\gamma$', '$k m_0$-scale', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-3, -3, 5, 0, 0, -1])
        prior_cov = np.diag([5, 5, 11, 2, 6, 2])
        self.prior_type = 'normal'

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         n_obs=180)

        print(f'Using the model {name}')

    def load_amortizer_configuration(self, model_idx: int = -1, load_best: bool = False) -> Optional[str]:

        # load best
        if load_best:
            model_idx = -1

        bidirectional_LSTM = [False, True]
        n_coupling_layers = [6, 7]
        n_dense_layers_in_coupling = [2, 3]
        coupling_design = ['spline', 'affine']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design))

        if model_idx >= len(combinations):
            return None

        self.n_epochs = 500
        (self.bidirectional_LSTM,
         self.n_coupling_layers,
         self.n_dense_layers_in_coupling,
         self.coupling_design) = combinations[model_idx]

        model_name = f'amortizer-small-fro' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def build_simulator(self,
                        with_noise: bool = True,
                        exp_func: str = 'exp') -> Simulator:
        # build simulator
        simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                          n_obs=self.n_obs,
                                                          with_noise=with_noise,
                                                          exp_func=exp_func))
        return simulator

    @staticmethod
    def load_data(n_data: int,
                  load_eGFP: bool = True, load_d2eGFP: bool = False) -> (np.ndarray, Optional[np.ndarray]):
        if not load_eGFP and not load_d2eGFP:
            obs_data = load_single_cell_data('data_random_cells', n_data)
            true_pop_parameters = pd.read_csv(f'data/synthetic/sample_pop_parameters.csv',
                                              index_col=0, header=0).loc[f'{n_data}'].values
        else:
            obs_data = load_multi_experiment_data(n_data, load_eGFP=load_eGFP, load_d2eGFP=load_d2eGFP)
            true_pop_parameters = None
        return obs_data, true_pop_parameters

    def print_and_plot_example(self, params: Optional[np.ndarray] = None) -> None:
        if params is None:
            params = self.prior(1)['prior_draws']

        # give parameters names and display them
        sample_y = batch_simulator(param_batch=params, n_obs=180, with_noise=False)
        sample_y_noisy = batch_simulator(param_batch=params, n_obs=180, with_noise=True)
        t_points = np.linspace(start=1 / 6, stop=30, num=180, endpoint=True)

        # Plot
        plt.plot(t_points, sample_y, 'b', label='model simulation')
        plt.plot(t_points, sample_y_noisy, 'g', label='simulation with noise')
        plt.xlabel('$t\, [h]$')
        plt.ylabel('fluorescence intensity [a.u.]')
        plt.title('Simulations')
        plt.legend()

        plt.show()
        return


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
