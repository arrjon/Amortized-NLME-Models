import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from numba import jit

from typing import Optional

from models.base_nlme_model import NlmeBaseAmortizer
from inference.helper_functions import load_single_cell_data, load_multi_experiment_data

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


@jit(nopython=True)
def add_noise_for_one_cell(y: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """
    Adds Gaussian noise to a given trajectory.

    Parameters:
    y (np.ndarray): The trajectory to which noise should be added.
    sigma (float): Standard deviation of the Gaussian noise.
    seed (int, optional): Seed for the random number generator.
            If provided, the function will always return the same noise for the same seed.

    Returns:
    np.ndarray: The noisy trajectory.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=sigma, size=y.size)
    y_noisy = y + noise
    return y_noisy


def batch_simulator(param_samples: np.ndarray, n_obs: int, with_noise: bool = True) -> np.ndarray:
    """
    Simulate ODE model

    param_samples: np.ndarray - (#simulations, #parameters) or (#parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1) or (#observations, 1)
    """

    n_sim = param_samples.shape[0]
    sim_data = np.zeros((n_sim, n_obs, 1), dtype=np.float32)
    t_points = np.linspace(start=1 / 6, stop=30, num=n_obs, endpoint=True)

    # iterate over batch
    for i, p_sample in enumerate(param_samples):
        delta, gamma, k, m0, scale, t0, offset, sigma = np.exp(p_sample)
        # simulate all observations together
        sol_euler = euler_maruyama(t0, m0, delta, gamma, k)
        sol_approx = measurement(sol_euler[:, 1:], scale, offset)
        # get the closest time points to t_points via interpolation
        sol = np.interp(t_points, sol_euler[:, 0], sol_approx, left=np.log(offset))
        # add noise
        if with_noise:
            sim_data[i, :, 0] = add_noise_for_one_cell(sol, sigma)
        else:
            sim_data[i, :, 0] = sol
    return sim_data


class FroehlichModelSDE(NlmeBaseAmortizer):
    def __init__(self, name: str = 'FroehlichModelSDE', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\delta$', '$\gamma$', '$k$', '$m_0$', 'scale', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
        prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         n_obs=180)

        print('Using the SDE version of the Froehlich model')

    def load_trained_model(self, model_idx: int = -1, load_best: bool = False) -> Optional[str]:
        # simulation time: 4.1 hours
        # total average training time
        # (929.28+1008.96+1089.31+974.45+1050.84+1113.85+908.02+1048.38+585.73+637.49+687.20+596.34+659.06+707.65)/
        # 14 = 856.90 / 60 = 14.28 hours
        # total average training time: 18.38 hours

        # load best
        if load_best:
            model_idx = 12

        if model_idx == 0:  # calibration of offset very bad, t_0, sigma bad
            self.n_epochs = 1500
            self.n_coupling_layers = 7
            self.final_loss = -5.594
            self.training_time = 929.28
        elif model_idx == 1:  # calibration of sigma, t_0 bad
            self.n_epochs = 1500
            self.n_coupling_layers = 8
            self.final_loss = -6.816
            self.training_time = 1008.96
        elif model_idx == 2:  # calibration of offset bad
            self.n_epochs = 1500
            self.n_coupling_layers = 9
            self.final_loss = -6.560
            self.training_time = 1089.31
        elif model_idx == 3:  # calibration of scale, offset bad
            self.n_epochs = 1500
            self.n_coupling_layers = 7
            self.summary_only_lstm = False
            self.final_loss = -6.591
            self.training_time = 974.45
        elif model_idx == 4:  # calibration of t_0 bad
            self.n_epochs = 1500
            self.n_coupling_layers = 8
            self.summary_only_lstm = False
            self.final_loss = -7.000
            self.training_time = 1050.84
        elif model_idx == 5:  # calibration of offset bad
            self.n_epochs = 1500
            self.n_coupling_layers = 9
            self.summary_only_lstm = False
            self.final_loss = -7.187
            self.training_time = 1113.85
        elif model_idx == 6:  # calibration of t_0, offset bad
            self.n_epochs = 1500
            self.n_coupling_layers = 7
            self.summary_loss_fun = 'MMD'
            self.final_loss = -5.844
            self.training_time = 908.02
        # elif model_idx == 7:
        #    self.n_epochs = 1500
        #    self.n_coupling_layers = 8
        #    self.summary_loss_fun = 'MMD'
        #    self.final_loss = np.Inf
        #    self.training_time = np.Inf
        elif model_idx == 8:  # calibration of offset bad
            self.n_epochs = 1500
            self.n_coupling_layers = 9
            self.summary_loss_fun = 'MMD'
            self.final_loss = -5.698
            self.training_time = 1048.38
        elif model_idx == 9:  # calibration of offset very bad, t_0 bad
            self.n_epochs = 1000
            self.n_coupling_layers = 7
            self.final_loss = -5.781
            self.training_time = 585.73
        elif model_idx == 10:  # calibration of sigma, t_0 not good
            self.n_epochs = 1000
            self.n_coupling_layers = 8
            self.final_loss = -6.693
            self.training_time = 637.49
        elif model_idx == 11:  # calibration of offset bad
            self.n_epochs = 1000
            self.n_coupling_layers = 9
            self.final_loss = -6.799
            self.training_time = 687.20
        elif model_idx == 12:  # calibration and posterior good
            self.n_epochs = 1000
            self.n_coupling_layers = 7
            self.summary_only_lstm = False
            self.final_loss = -6.554
            self.training_time = 596.34
        elif model_idx == 13:  # calibration of t_0 bad
            self.n_epochs = 1000
            self.n_coupling_layers = 8
            self.summary_only_lstm = False
            self.final_loss = -7.256
            self.training_time = 659.06
        elif model_idx == 14:  # calibration of offset bad
            self.n_epochs = 1000
            self.n_coupling_layers = 9
            self.summary_only_lstm = False
            self.final_loss = -7.010
            self.training_time = 707.65
        else:
            return None

        model_name = f'amortizer-sde-fro' \
                     f'-{self.n_coupling_layers}layers' \
                     f'{"-LSTM" if self.summary_only_lstm else ""}' \
                     f'{"-summary_loss" if self.summary_loss_fun is not None else ""}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def old_load_trained_model(self, model_idx: int = 0, load_best: bool = False) -> Optional[str]:
        # simulation time total 36681.7 minutes / 48 kernels = 764.20 minutes = 12.74 hours (500.000 simulations)
        # simulation time total 23.49 hours / 8 kernels = 2.94 hours (750.000 batches using jit)
        # simulation time new 2 total 26.03 hours / 8 kernels = 3.25 hours (750.000 batches using jit)

        # load best
        if load_best:
            model_idx = 10  # 9  # 6  # 1

        if model_idx == 0:
            self.prior_mean = np.array([-3, -3, 0, 4, 1, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 5, 11, 5, 2, 6, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 5
            self.final_loss = -3.178
            self.training_time = 190.99
            model_name = 'amortizer-fro-sde-5layers-LSTM-500epochs'
        elif model_idx == 1:
            self.prior_mean = np.array([-3, -3, 0, 4, 1, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 5, 11, 5, 2, 6, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.final_loss = -3.065
            self.training_time = 220.62
            model_name = 'amortizer-fro-sde-6layers-LSTM-500epochs'
        elif model_idx == 2:
            self.prior_mean = np.array([-3, -3, 0, 4, 1, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 5, 11, 5, 2, 6, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 7
            self.final_loss = -3.055
            self.training_time = 238.25
            model_name = 'amortizer-fro-sde-7layers-LSTM-500epochs'
        elif model_idx == 3:  # online training
            self.prior_mean = np.array([-3, -3, -3, 4, 0, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.final_loss = -0.570
            self.training_time = 1178.48  # 20 hours
            model_name = 'amortizer-fro-sde-6layers-LSTM-500epochs-new-prior'
        elif model_idx == 4:  # online training
            self.prior_mean = np.array([-3, -3, -3, 4, 0, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 7
            self.final_loss = -0.673
            self.training_time = 1202.92
            model_name = 'amortizer-fro-sde-7layers-LSTM-500epochs-new-prior'
        elif model_idx == 5:
            self.prior_mean = np.array([-3, -3, -3, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 6
            self.final_loss = -5.303
            self.training_time = 317.24  # minutes = 5.29 hours
            model_name = 'amortizer-fro-sde-6layers-LSTM-750epochs-new-prior'
        elif model_idx == 6:
            self.prior_mean = np.array([-3, -3, -3, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 7
            self.final_loss = -5.783
            self.training_time = 368.27  # minutes = 6.14 hours
            model_name = 'amortizer-fro-sde-7layers-LSTM-750epochs-new-prior'
        elif model_idx == 7:
            self.prior_mean = np.array([-3, -3, -3, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 8
            self.final_loss = -5.589
            self.training_time = 403.61  # minutes = 6.73 hours
            model_name = 'amortizer-fro-sde-8layers-LSTM-750epochs-new-prior'
        elif model_idx == 8:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 7
            self.final_loss = -7.317
            self.training_time = 378.12
            model_name = 'amortizer-fro-sde-7layers-LSTM-750epochs-new-prior2'
        elif model_idx == 9:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 8
            self.final_loss = -7.571
            self.training_time = 398.43
            model_name = 'amortizer-fro-sde-8layers-LSTM-750epochs-new-prior2'
        elif model_idx == 10:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 8
            self.final_loss = -7.197
            self.training_time = 434.58
            self.summary_only_lstm = False
            model_name = 'amortizer-fro-sde-8layers-750epochs-new-prior2'
        elif model_idx == 11:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 1000
            self.n_coupling_layers = 8
            self.final_loss = -7.032
            self.training_time = 551.85
            model_name = 'amortizer-fro-sde-8layers-LSTM-10000epochs-new-prior2'
        elif model_idx == 12:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 750
            self.n_coupling_layers = 9
            self.final_loss = -7.314
            self.training_time = 424.19
            model_name = 'amortizer-fro-sde-9layers-LSTM-750epochs-new-prior2'
        elif model_idx == 13:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 1500
            self.n_coupling_layers = 9
            self.final_loss = np.Inf
            self.training_time = np.Inf
            model_name = 'amortizer-fro-sde-9layers-LSTM-1500epochs-new-prior2'
        elif model_idx == 14:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 1500
            self.n_coupling_layers = 8
            self.summary_only_lstm = False
            self.final_loss = np.Inf
            self.training_time = np.Inf
            model_name = 'amortizer-fro-sde-8layers-1500epochs-new-prior2'
        elif model_idx == 15:
            self.prior_mean = np.array([-3, -3, -1, 5, 0, 0, 1, -3])
            self.prior_cov = np.diag([5, 5, 5, 5, 5, 2, 5, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 1500
            self.n_coupling_layers = 9
            self.summary_only_lstm = False
            self.final_loss = np.Inf
            self.training_time = np.Inf
            model_name = 'amortizer-fro-sde-9layers-15000epochs-new-prior2'
        else:
            return None
        return model_name

    def build_simulator(self, with_noise: bool = True) -> Simulator:
        # build simulator
        simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                          n_obs=self.n_obs,
                                                          with_noise=with_noise))
        return simulator

    @staticmethod
    def load_data(n_data: int,
                  load_eGFP: bool = True, load_d2eGFP: bool = False) -> (np.ndarray, Optional[np.ndarray]):
        if not load_eGFP and not load_d2eGFP:
            obs_data = load_single_cell_data('data_random_cells_sde_model', n_data)
            true_pop_parameters = pd.read_csv(f'data/synthetic/sample_pop_parameters_sde_model.csv',
                                              index_col=0, header=0).loc[f'{n_data}'].values
        else:
            obs_data = load_multi_experiment_data(n_data, load_eGFP=load_eGFP, load_d2eGFP=load_d2eGFP)
            true_pop_parameters = None
        return obs_data, true_pop_parameters

    def print_and_plot_example(self,  # define some sample parameters
                               delta: float = 0.8,  # np.log(2)/(0.8) # per hour
                               gamma: float = 0.03,  # np.log(2)/(22.8) # per hour
                               k: float = 1.44,
                               m0: float = 300,
                               scale: float = 2.12,
                               t_0: float = 0.9,  # in hours
                               offset: float = 8.,  # a.u
                               sigma: float = np.sqrt(0.001)  # must be positive
                               ) -> None:
        # give parameters names and display them
        theta_population = np.array([delta, gamma, k, m0, scale, t_0, offset, sigma]).reshape(1, self.n_params)
        df_params = pd.DataFrame(theta_population, columns=self.param_names)
        display(df_params)

        # simulate measurement without noise
        t_points = np.linspace(start=1 / 6, stop=30, num=181, endpoint=True)
        sol_euler = euler_maruyama(t_0, m0, delta, gamma, k)
        sol_approx = measurement(sol_euler[:, 1:], scale, offset)
        # get the closest time points to t_points via interpolation
        y_sim = np.interp(t_points, sol_euler[:, 0], sol_approx, left=np.log(offset))
        # add time 0
        t_points = np.append(0, t_points)
        y_sim = np.append(np.log(offset), y_sim)
        # add noise
        y_sim_noisy = add_noise_for_one_cell(y_sim, sigma)

        # Plot
        figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axis[0].plot(t_points, np.exp(y_sim_noisy), 'b', label='$y$ simulation with noise')
        axis[0].plot(t_points, np.exp(y_sim), 'r', label='$y$ simulation without noise')
        axis[0].set_xlabel('$t\, [h]$')
        axis[0].set_ylabel('fluorescence intensity [a.u.]')
        axis[0].set_title('Observed Time Points of $y$')
        axis[0].set_yscale('log')
        axis[0].legend()

        axis[1].plot(sol_euler[:, 0], np.exp(sol_approx), 'r', label='$y$ simulation without noise')
        axis[1].set_xlabel('$t\, [h]$')
        axis[1].set_ylabel('fluorescence intensity [a.u.]')
        axis[1].set_yscale('log')
        axis[1].set_title('Full SDE simulation of $y$')
        axis[1].set_yscale('log')
        axis[1].legend()

        plt.show()
        return
