#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the simple Fröhlich NLME Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Union
from functools import partial
from datetime import datetime
import itertools

from inference.base_nlme_model import NlmeBaseAmortizer
from bayesflow.simulation import Simulator


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
    with np.errstate(over='ignore'):  # some values in exp might be too large, will return inf
        if delta == gamma:
            # the analytical solution is different in this case
            m = np.exp(-delta * (t - t_0))
            m[t - t_0 < 0] = 0
            p = k_m0_scale * np.exp(-gamma * (t - t_0)) * (t - t_0)
            p[t - t_0 < 0] = 0
            return np.row_stack((m, p))

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
                    n_obs: int = 180,
                    with_noise: bool = True,
                    exp_func: str = 'exp') -> np.ndarray:
    """
    Simulate ODE model

    param_samples: np.ndarray - (#simulations, #parameters) or (#parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)
    exp_func: str - if 'exp' then exp(log_param_samples) is used as parameter samples, otherwise
                    if exp_func == 'power10' then 10**log_param_samples is used as parameter samples

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1) or (#observations)
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
    return sim_data[:, :, np.newaxis]  # add dimension for the channel (n_sim, n_obs, 1)


class FroehlichModelSimple(NlmeBaseAmortizer):
    def __init__(self, name: str = 'SimpleFroehlichModel', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\delta$', '$\gamma$', '$k m_0$-scale', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-3., -3., 5., 0., 0., -1.])
        prior_cov = np.diag([5., 5., 11., 2., 6., 2.])

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         prior_type='normal',
                         max_n_obs=180)

        self.simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                               n_obs=180))

        print(f'Using the model {name}')

    def load_amortizer_configuration(self, model_idx: int = -1, load_best: bool = False) -> str:
        self.n_epochs = 500
        self.summary_dim = self.n_params * 2

        # load best
        if load_best:
            model_idx = 9
            # amortizer-small-fro-sequence-summary-Bi-LSTM-6layers-2coupling-spline-500epochs -> 9

        summary_network_type = ['sequence']
        bidirectional_LSTM = [False, True]
        n_coupling_layers = [6, 7]
        n_dense_layers_in_coupling = [2, 3]
        coupling_design = ['affine', 'spline']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design, summary_network_type))

        if model_idx >= len(combinations) or model_idx < 0:
            model_name = f'amortizer-simple-fro' \
                         f'-{self.summary_network_type}-summary' \
                         f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                         f'-{self.n_coupling_layers}layers' \
                         f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                         f'-{self.n_epochs}epochs' \
                         f'-{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
            return model_name

        (self.bidirectional_LSTM,
         self.n_coupling_layers,
         self.n_dense_layers_in_coupling,
         self.coupling_design,
         self.summary_network_type) = combinations[model_idx]

        model_name = f'amortizer-simple-fro' \
                     f'-{self.summary_network_type}-summary' \
                     f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    @staticmethod
    def load_data(n_data: Optional[int] = None,
                  load_egfp: bool = True, load_d2egfp: bool = False,
                  synthetic: bool = False) -> np.ndarray:
        if synthetic:
            # load synthetic data which is saved in csv
            obs_data = load_single_cell_data('data_random_cells', real_data=False)
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
        true_pop_parameters = pd.read_csv(f'../data/synthetic/sample_pop_parameters.csv',
                                          index_col=0, header=0).loc[f'{n_data}'].values
        return true_pop_parameters

    def plot_example(self, params: Optional[np.ndarray] = None) -> None:
        """Plots an individual trajectory of an individual in this model."""
        if params is None:
            params = self.prior(1)['prior_draws'][0]

        output = batch_simulator(params, n_obs=180, with_noise=True)
        ax = self.prepare_plotting(output, params)

        plt.title(f'Cell Simulation')
        plt.legend()
        plt.show()
        return

    @staticmethod
    def prepare_plotting(data: np.ndarray, params: np.ndarray, ax: Optional[plt.Axes] = None,
                         with_noise: bool = False) -> plt.Axes:
        # simulate data
        sim_data = batch_simulator(param_batch=params, n_obs=180, with_noise=with_noise)
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
                            alpha=0.2, color='b', label='95% quantiles')
            ax.plot(t_measurement, y_median, 'b', label='median')

        # plot observed data
        ax.scatter(t_measurement, data, color='b', label='measurements')
        ax.set_xlabel('$t\, [h]$')
        ax.set_ylabel('log fluorescence intensity [a.u.]')
        return ax


def load_single_cell_data(file_name: str, real_data: bool) -> np.ndarray:
    if real_data:
        # real data
        data = pd.read_excel(f'../data/froehlich_eGFP/{file_name}.xlsx', index_col=0, header=None)
    else:
        # synthetic data which is saved as csv
        data = pd.read_csv(f'../data/synthetic/{file_name}.csv', index_col=0, header=0)

    # convert to right format
    data.index = data.index / 60 / 60  # convert to hours
    data.drop(index=data.index[data.index > 30], inplace=True)

    # format for BayesFlow
    n_real_cells = data.columns.shape[0]
    n_time_points = data.index.shape[0]
    data = np.log(data.values.T).reshape(n_real_cells, n_time_points, 1)
    return data


def load_multi_experiment_data(load_egfp: bool, load_d2egfp: bool) -> Union[np.ndarray, list[np.ndarray]]:
    # name of the files for the different experiments
    names_egfp = ['20160427_mean_eGFP', '20160513_mean_eGFP', '20160603_mean_eGFP']
    names_d2egfp = ['20160427_mean_d2eGFP', '20160513_mean_d2eGFP', '20160603_mean_d2eGFP']

    if not load_egfp and not load_d2egfp:
        raise ValueError("At least one of the two options ('load_eGFP', 'load_d2eGFP') has to be True.")
    data_egfp = None  # only to silence warning

    # load data
    if load_egfp:
        data_list_egfp = []
        for name in names_egfp:
            data_list_egfp.append(load_single_cell_data(file_name=name,
                                                        real_data=True))
        # concatenate data: shape for BayesFlow
        data_egfp = np.concatenate(data_list_egfp, axis=0)

        if not load_d2egfp:
            return data_egfp

    data_list_d2egfp = []
    for name in names_d2egfp:
        data_list_d2egfp.append(load_single_cell_data(file_name=name,
                                                      real_data=True))
    # concatenate data: shape for BayesFlow
    data_d2egfp = np.concatenate(data_list_d2egfp, axis=0)

    if not load_egfp:
        return data_d2egfp

    return [data_egfp, data_d2egfp]
