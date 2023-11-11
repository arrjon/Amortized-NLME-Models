#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the SDE Fröhlich NLME Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

from typing import Optional
from functools import partial
from datetime import datetime
import itertools

from inference.base_nlme_model import NlmeBaseAmortizer
from models.froehlich_model_simple import load_single_cell_data, load_multi_experiment_data, add_noise

from bayesflow.simulation import Simulator


@jit(nopython=True)
def drift_term(x: np.ndarray, delta: float, gamma: float, k: float) -> np.ndarray:
    """
    Computes the drift term of the SDE.

    Parameters:
    x (np.ndarray): 2-dimensional state of the system at time t.
    delta (float): Parameter of the SDE.
    gamma (float): Parameter of the SDE.
    k (float): Parameter of the SDE.

    Returns:
    np.ndarray: Drift term of the SDE at time t.
    """
    m = -delta * x[0]
    p = k * x[0] - gamma * x[1]
    return np.array([m, p])


@jit(nopython=True)
def diffusion_term(x: np.ndarray, delta: float, gamma: float, k: float) -> np.ndarray:
    """
    Computes the diffusion term of the SDE.

    Parameters:
    x (np.ndarray): 2-dimensional state of the system at time t.
    delta (float): Parameter of the SDE.
    gamma (float): Parameter of the SDE.
    k (float): Parameter of the SDE.

    Returns:
    np.ndarray: Diffusion term of the SDE at time t.
    """
    m = np.sqrt(delta * x[0])
    p = np.sqrt(k * x[0] + gamma * x[1])
    return np.array([m, p])


@jit(nopython=True)
def measurement(x: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """
    Applies a measurement function to a given variable.

    Parameters:
    x (np.ndarray): Measurements of the variable to which the measurement function should be applied.
    scale (float): Scale for the measurement function.
    offset (float): Offset for the measurement function.

    Returns:
    np.ndarray: The result of applying the measurement function to the input variable.
    """
    return np.log(scale * x[:, 1] + offset)


@jit(nopython=True)
def euler_maruyama(t0: float, m0: float, delta: float, gamma: float, k: float,
                   step_size: float = 0.01) -> np.ndarray:
    """
    Simulates the SDE using the Euler-Maruyama method.

    Parameters:
    t0 (float): Initial time point.
    m0 (float): Initial value of the system.
    delta (float): Parameter of the SDE.
    gamma (float): Parameter of the SDE.
    k (float): Parameter of the SDE.
    step_size (float, optional): Step size for the Euler-Maruyama method.

    Returns:
    tuple(np.ndarray, np.ndarray): The simulated trajectory and the corresponding time points.
    """
    # if t0 > 30, return only the measurement at t0
    if t0 > 30:
        t_points = np.array([t0])
        x0 = np.array([[m0, 0]])
        return np.column_stack((t_points, x0))

    # precompute time points (including t0 and t_end=30)
    t_points = np.arange(t0, 30 + step_size, step_size)
    n_points = t_points.size

    # initialize array
    x = np.zeros((n_points, 2))
    x[0] = [m0, 0]

    # precompute random numbers for the diffusion term (Brownian motion)
    bm = np.random.normal(loc=0, scale=np.sqrt(step_size), size=(n_points - 1, 2))

    # simulate one step at a time (first step already done)
    for t_idx in range(n_points - 1):
        drift = step_size * drift_term(x[t_idx], delta, gamma, k)
        diffusion = bm[t_idx] * diffusion_term(x[t_idx], delta, gamma, k)
        # add up and set negative values to zero
        x[t_idx + 1] = np.maximum(x[t_idx] + drift + diffusion, 0)
    return np.column_stack((t_points, x))


def batch_simulator(param_batch: np.ndarray, n_obs: int, with_noise: bool = True) -> np.ndarray:
    """
    Simulate ODE model

    param_batch: np.ndarray - (#simulations, #parameters) or (#parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1) or (#observations, 1)
    """
    # simulate batch
    if len(param_batch.shape) == 1:  # so not (batch_size, params)
        # just a single parameter set
        param_batch = param_batch[np.newaxis, :]
    n_sim = param_batch.shape[0]
    sim_data = np.zeros((n_sim, n_obs), dtype=np.float32)

    t_points = np.linspace(start=1 / 6, stop=30, num=n_obs, endpoint=True)
    # iterate over batch
    for i, p_sample in enumerate(param_batch):
        delta, gamma, k, m0, scale, t0, offset, _ = np.exp(p_sample)
        # simulate all observations together
        sol_euler = euler_maruyama(t0, m0, delta, gamma, k)
        sol_approx = measurement(sol_euler[:, 1:], scale, offset)
        # get the closest time points to t_points via interpolation
        sim_data[i, :] = np.interp(t_points, sol_euler[:, 0], sol_approx, left=np.log(offset))

    # add noise
    if with_noise:
        sim_data = add_noise(sim_data, sigmas=np.exp(param_batch[:, -1]))

    if n_sim == 1:
        # return only one simulation
        return sim_data[0]
    return sim_data[:, :, np.newaxis]  # add dimension for the channel (n_sim, n_obs, 1)


class FroehlichModelSDE(NlmeBaseAmortizer):
    def __init__(self, name: str = 'SDEFroehlichModel', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\delta$', '$\gamma$', '$k$', '$m_0$', 'scale', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-3, -3, -1, 5, 0, 0, 0, -1])
        prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
        self.prior_type = 'normal'

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         n_obs=180)

        print(f'Using the model {name}')

    def load_amortizer_configuration(self, model_idx: int = -1, load_best: bool = False) -> str:
        self.n_epochs = 750
        self.summary_dim = self.n_params * 2

        # load best
        if load_best:
            model_idx = -1

        bidirectional_LSTM = [False, True]
        n_coupling_layers = [7, 8]
        n_dense_layers_in_coupling = [2, 3]
        coupling_design = ['affine', 'spline']
        summary_network_type = ['sequence']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design, summary_network_type))

        # also test on configuration with a transformer as summary network
        combinations.append((False, 7, 2, 'affine', 'transformer'))

        if model_idx >= len(combinations) or model_idx < 0:
            model_name = f'amortizer-sde-fro' \
                         f'-{self.summary_network_type}-summary' \
                         f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                         f'-{self.n_coupling_layers}layers' \
                         f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                         f'-{self.n_epochs}epochs' \
                         f'-{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
            return model_name

        (self.summary_only_lstm,
         self.bidirectional_LSTM,
         self.n_dense_layers_in_coupling,
         self.coupling_design,
         self.summary_network_type) = combinations[model_idx]

        model_name = f'amortizer-sde-fro' \
                     f'-{self.summary_network_type}-summary' \
                     f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def build_simulator(self, with_noise: bool = True) -> Simulator:
        # build simulator
        simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                          n_obs=self.n_obs,
                                                          with_noise=with_noise))
        return simulator

    @staticmethod
    def load_data(n_data: Optional[int] = None,
                  load_egfp: bool = True, load_d2egfp: bool = False,
                  synthetic: bool = False) -> np.ndarray:
        if synthetic:
            # load synthetic data which is saved in csv
            obs_data = load_single_cell_data('data_random_cells_sde_model', real_data=False)
            if n_data is not None:
                obs_data = obs_data[:n_data]
        else:
            obs_data = load_multi_experiment_data(load_egfp=load_egfp, load_d2egfp=load_d2egfp)
            if n_data is not None:
                if load_egfp and load_d2egfp:
                    obs_data = [data[:int(n_data / 2)] for data in obs_data]
                else:
                    obs_data = obs_data[:n_data]
        return obs_data

    @staticmethod
    def load_synthetic_parameter(n_data: int) -> np.ndarray:
        true_pop_parameters = pd.read_csv(f'data/synthetic/sample_pop_parameters_sde_model.csv',
                                          index_col=0, header=0).loc[f'{n_data}'].values
        return true_pop_parameters

    def plot_example(self, params: Optional[np.ndarray] = None) -> None:
        if params is None:
            params = self.prior(10)['prior_draws']

        output = batch_simulator(params[0], n_obs=180, with_noise=True)
        ax = self.prepare_plotting(output, params)

        plt.title(f'Cell Simulation')
        plt.legend()
        plt.show()
        return

    @staticmethod
    def prepare_plotting(data: np.ndarray, params: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
        # simulate data
        sim_data = batch_simulator(param_batch=params, n_obs=180, with_noise=False)
        t_measurement = np.linspace(start=1 / 6, stop=30, num=180, endpoint=True)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)

        if len(params.shape) == 1:  # so not (batch_size, params)
            # just a single parameter set
            # plot simulated data
            ax.plot(t_measurement, sim_data, 'b', label='simulated cell')
        else:
            # remove channel dimension of bayesflow
            sim_data = sim_data[:, :, 0]
            # calculate median and quantiles
            y_median = np.median(sim_data, axis=0)
            y_quantiles = np.percentile(sim_data, [2.5, 97.5], axis=0)

            # plot simulated data
            ax.fill_between(t_measurement, y_quantiles[0], y_quantiles[1],
                            alpha=0.2, color='b')
            ax.plot(t_measurement, y_median, 'b', label='median')

        # plot observed data
        ax.scatter(t_measurement, data, color='b', label='measurements')
        ax.set_xlabel('$t\, [h]$')
        ax.set_ylabel('fluorescence intensity [a.u.]')
        return ax
