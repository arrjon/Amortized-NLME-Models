#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the detailed Fröhlich NLME Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from typing import Optional
import itertools

from inference.base_nlme_model import NlmeBaseAmortizer
from models.froehlich_model_simple import load_single_cell_data, load_multi_experiment_data, add_noise

from bayesflow.simulation import Simulator

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

jlPkg.activate("models/SimulatorFroehlich")
jl.seval("using SimulatorFroehlich")


# a function which can simulate a batch of parameters or a single parameter set
def batch_simulator(param_batch: np.ndarray,
                    n_obs: int,
                    with_noise: bool = True,
                    exp_func: str = 'exp') -> np.ndarray:
    """
    Simulate ODE model for multiple parameter sets

    param_batch: np.ndarray - (#simulations, #parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)
    exp_func: str - if 'exp' then exp(log_param_samples) is used as parameter samples, otherwise
                    if exp_func == 'power10' then 10**log_param_samples is used as parameter samples

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1)
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
    return sim_data[:, np.newaxis]  # add dimension for the channel (n_sim, n_obs, 1)


class FroehlichModelDetailed(NlmeBaseAmortizer):
    def __init__(self, name: str = 'DetailedFroehlichModel', network_idx: int = -1, load_best: bool = False):
        param_names = ['$\delta_1 m_0$', '$\delta_2$', '$e_0 m_0$', '$k_2 m_0 scale$', '$k_2$',
                       '$k_1 m_0$', '$r_0 m_0$', '$\gamma$', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-1, -1, -1, 12, -1, 1, -1, -6, 0, 0, -1])
        prior_diag = np.array([5, 5, 2, 2, 2, 2, 2, 5, 2, 5, 2])
        prior_diag[3] = 1  # otherwise too many samples lead to overflow
        prior_cov = np.diag(prior_diag)
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
        n_coupling_layers = [7, 8]
        n_dense_layers_in_coupling = [2, 3]
        coupling_design = ['spline', 'affine']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design))

        if model_idx >= len(combinations):
            return None

        self.n_epochs = 750
        (self.bidirectional_LSTM,
         self.n_coupling_layers,
         self.n_dense_layers_in_coupling,
         self.coupling_design) = combinations[model_idx]

        model_name = f'amortizer-large-fro' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{"bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
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
    def load_data(n_data: int, load_eGFP: bool = True, load_d2eGFP: bool = False) -> (np.ndarray, Optional[np.ndarray]):
        if not load_eGFP and not load_d2eGFP:
            obs_data = load_single_cell_data('data_random_cells_large_model', n_data)
            true_pop_parameters = pd.read_csv(f'data/synthetic/sample_pop_parameters_large_model.csv',
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
