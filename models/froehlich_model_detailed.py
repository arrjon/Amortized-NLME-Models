#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the detailed Fröhlich NLME Model

import itertools
import os
import pathlib
from datetime import datetime
from functools import partial
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesflow.simulation import Simulator

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

from inference.base_nlme_model import NlmeBaseAmortizer
from models.froehlich_model_simple import load_single_cell_data, load_multi_experiment_data, add_noise

env = os.path.join(pathlib.Path(__file__).parent.resolve(), 'SimulatorFroehlich')
jlPkg.activate(env)
jl.seval("using SimulatorFroehlich")


# a function which can simulate a batch of parameters or a single parameter set
def batch_simulator(param_batch: np.ndarray,
                    n_obs: int = 180,
                    with_noise: bool = True,
                    exp_func: str = 'exp') -> np.ndarray:
    """
    Simulate ODE model for multiple parameter sets

    param_batch: np.ndarray - (#simulations, #parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)
    exp_func: str - if 'exp' then exp(log_param_samples) is used as parameter samples, otherwise
                    if exp_func == 'power10' then 10**log_param_samples is used as parameter samples

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1)  or (#observations)
    """
    # simulate batch
    if param_batch.ndim == 1:  # so not (batch_size, params)
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

    # iterate over parameter samples and simulate data
    for i, params in enumerate(exp_param_batch):
        # extract parameters
        delta1_m0, delta2, e0_m0, k2_m0_scale, k2, k1_m0, r0_m0, gamma, t_0, offset, _ = params

        # solve ODE using Julia
        sim_data[i, :] = jl.simulateLargeModel(t_0, delta1_m0, delta2,
                                               e0_m0, k2_m0_scale, k2, k1_m0,
                                               r0_m0, gamma, offset).to_numpy()

    # add noise for each cell
    if with_noise:
        sim_data = add_noise(sim_data, sigmas=exp_param_batch[:, -1])

    if n_sim == 1:
        # return only one simulation
        return sim_data[0]
    return sim_data[:, :, np.newaxis]  # add dimension for the channel (n_sim, n_obs, 1)


class FroehlichModelDetailed(NlmeBaseAmortizer):
    def __init__(self, name: str = 'DetailedFroehlichModel', network_idx: int = -1, load_best: bool = False):
        param_names = ['$\delta_1 m_0$', '$\delta_2$', '$e_0 m_0$', '$k_2 m_0 scale$', '$k_2$',
                       '$k_1 m_0$', '$r_0 m_0$', '$\gamma$', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-1., -1., -1., 12., -1., 1., -1., -6., 0., 0., -1.])
        prior_diag = np.array([5., 5., 2., 2., 2., 2., 2., 5., 2., 5., 2.])
        prior_diag[3] = 1  # otherwise too many samples lead to overflow
        prior_cov = np.diag(prior_diag)

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
            # amortizer-detailed-fro-sequence-summary-Bi-LSTM-7layers-2coupling-spline-500epochs -> 9

        bidirectional_LSTM = [False, True]
        n_coupling_layers = [7, 8]
        n_dense_layers_in_coupling = [2, 3]
        coupling_design = ['affine', 'spline']
        summary_network_type = ['sequence']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design, summary_network_type))

        if model_idx >= len(combinations) or model_idx < 0:
            model_name = f'amortizer-detailed-fro' \
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

        model_name = f'amortizer-detailed-fro' \
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
            obs_data = load_single_cell_data('data_random_cells_detailed_model', real_data=False)
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
        true_pop_parameters = pd.read_csv(f'../data/synthetic/sample_pop_parameters_detailed_model.csv',
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
