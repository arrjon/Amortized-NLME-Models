import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional
from functools import partial

from models.base_nlme_model import NlmeBaseAmortizer
from inference.helper_functions import load_single_cell_data, load_multi_experiment_data
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


def batch_simulator(param_samples: np.ndarray, n_obs: int, with_noise: bool = True,
                    exp_func: str = 'exp') -> np.ndarray:
    """
    Simulate ODE model

    param_samples: np.ndarray - (#simulations, #parameters) or (#parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1) or (#observations, 1)
    """

    n_sim = param_samples.shape[0]
    sim_data = np.zeros((n_sim, n_obs, 1), dtype=np.float32)
    if exp_func == 'exp':
        exp_param_samples = np.exp(param_samples)
    elif exp_func == 'power10':
        exp_param_samples = np.power(10, param_samples)
    else:
        raise ValueError('exp_func must be "exp" or "power10"')
    t_points = np.linspace(start=1 / 6, stop=30, num=n_obs, endpoint=True)

    # iterate over batch
    for i, exp_param in enumerate(exp_param_samples):
        # simulate all observations together
        delta, gamma, k_m0_scale, t_0, offset, sigma = exp_param
        sol_ana = ode_analytical_sol(t=t_points,
                                     delta=delta,
                                     gamma=gamma,
                                     k_m0_scale=k_m0_scale,
                                     t_0=t_0)
        y_sim = measurement(y=sol_ana, offset=offset)
        if with_noise:
            # add noise for each cell
            sim_data[i, :, 0] = add_noise_for_one_cell(y=y_sim, sigma=sigma)
        else:
            sim_data[i, :, 0] = y_sim
    return sim_data


class FroehlichModelSmall(NlmeBaseAmortizer):
    def __init__(self, name: str = 'FroehlichModelSmall', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\delta$', '$\gamma$', '$k$-$m_0$-scale', '$t_0$', 'offset', '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-3, -3, 5, 0, 1, -1])
        prior_cov = np.diag([5, 5, 11, 2, 6, 2])

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         n_obs=180)

        print('Using the small Froehlich model')

    def build_simulator(self, with_noise: bool = True, exp_func: str = 'exp') -> Simulator:
        # build simulator
        simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                          n_obs=self.n_obs,
                                                          with_noise=with_noise,
                                                          exp_func=exp_func))
        return simulator

    def load_trained_model(self, model_idx: int = -1, load_best: bool = False) -> Optional[str]:
        # (450.80 + 480.13 + 466.86 + 500.45 + 449.60 + 485.27 + 289.88
        # + 320.76 + 308.71 + 335.35 + 296.93 + 328.64) / 12 = 392.78 [min] =  6.55 [h]

        # load best
        if load_best:
            model_idx = 6

        if model_idx == 0:  # nice bimodal posterior, sigma badly calibrated
            self.n_epochs = 750
            self.n_coupling_layers = 6
            self.final_loss = -7.156
            self.training_time = 450.80
        elif model_idx == 1:  # sigma too large calibrated
            self.n_epochs = 750
            self.n_coupling_layers = 7
            self.final_loss = -6.730
            self.training_time = 480.13
        elif model_idx == 2:  # calibration okayish, posterior way too narrow
            self.n_epochs = 750
            self.n_coupling_layers = 6
            self.summary_only_lstm = False
            self.final_loss = -7.052
            self.training_time = 466.86
        elif model_idx == 3:  # sigma badly calibrated
            self.n_epochs = 750
            self.n_coupling_layers = 7
            self.summary_only_lstm = False
            self.final_loss = -7.333
            self.training_time = 500.45
        elif model_idx == 4:  # offset, k_m0_scale badly calibrated
            self.n_epochs = 750
            self.n_coupling_layers = 6
            self.summary_loss_fun = 'MMD'
            self.final_loss = -6.326
            self.training_time = 449.60
        elif model_idx == 5:
            self.n_epochs = 750
            self.n_coupling_layers = 7
            self.summary_loss_fun = 'MMD'
            self.final_loss = -6.625
            self.training_time = 485.27
        elif model_idx == 6:  # okayish, posterior quite broad, but bi-modality visible
            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.final_loss = -6.676
            self.training_time = 289.88
        elif model_idx == 7:  # t_0, k_m0_scale badly calibrated
            self.n_epochs = 500
            self.n_coupling_layers = 7
            self.final_loss = -6.801
            self.training_time = 320.76
        elif model_idx == 8:  # posterior really narrow
            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.summary_only_lstm = False
            self.final_loss = -6.975
            self.training_time = 308.71
        elif model_idx == 9:  # t_0, sigma badly calibrated
            self.n_epochs = 500
            self.n_coupling_layers = 7
            self.summary_only_lstm = False
            self.final_loss = -7.139
            self.training_time = 335.35
        elif model_idx == 10:
            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.summary_loss_fun = 'MMD'
            self.final_loss = -6.350
            self.training_time = 296.93
        elif model_idx == 11:
            self.n_epochs = 500
            self.n_coupling_layers = 7
            self.summary_loss_fun = 'MMD'
            self.final_loss = -6.417
            self.training_time = 328.64
        else:
            return None

        model_name = f'amortizer-small-fro' \
                     f'-{self.n_coupling_layers}layers' \
                     f'{"-LSTM" if self.summary_only_lstm else ""}' \
                     f'{"-summary_loss" if self.summary_loss_fun is not None else ""}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def old_load_trained_model(self, model_idx: int = 0, load_best: bool = False) -> Optional[str]:

        # load best
        if load_best:
            model_idx = 2

        if model_idx == 0:
            self.prior_mean = np.array([0, 0, 5, 0, 1, -1])
            self.prior_cov = np.diag([2, 2, 10, 1, 5, 1])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))
            self.n_epochs = 500
            self.n_coupling_layers = 5
            self.final_loss = -7.340
            self.training_time = 228.65
            model_name = 'amortizer-small-fro-5layers-LSTM-500epochs_1'
        elif model_idx == 1:
            self.prior_mean = np.array([0, 0, 5, 0, 1, -1])
            self.prior_cov = np.diag([2, 2, 10, 1, 5, 1])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))
            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.final_loss = -7.189
            self.training_time = 264.23
            model_name = 'amortizer-small-fro-6layers-LSTM-500epochs'
        elif model_idx == 2:
            self.prior_mean = np.array([-3, -3, 5, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 11, 2, 6, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.final_loss = -7.895
            self.training_time = 248.07
            model_name = 'amortizer-small-fro-6layers-LSTM-500epochs-prior4'
        elif model_idx == 3:
            self.prior_mean = np.array([-3, -3, 5, 0, 1, -1])
            self.prior_cov = np.diag([5, 5, 11, 2, 6, 2])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_epochs = 500
            self.n_coupling_layers = 6
            self.final_loss = -7.850
            self.training_time = 253.70
            model_name = 'amortizer-small-fro-6layers-LSTM-500epochs-all'
        elif model_idx == 4:
            self.prior_mean = np.array([0, 0, 5, 0, 1, -1])
            self.prior_cov = np.diag([2, 2, 10, 1, 5, 1])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))
            self.final_loss = -6.727
            self.training_time = 237.98
            model_name = "amortizer-2023-05-01 17-09-14"
        elif model_idx == 5:  # using hierarchical simulations (=64)
            self.prior_mean = np.array([0, 0, 5, 0, 1, -1])
            self.prior_cov = np.diag([2, 2, 10, 1, 5, 1])
            self.prior_std = np.sqrt(np.diag(self.prior_cov))
            self.final_loss = -4.817
            self.training_time = 194.43
            model_name = "amortizer-2023-05-01 23-25-53"
        else:
            return None
        return model_name

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

    def print_and_plot_example(self,  # define some sample parameters
                               delta=0.8,  # np.log(2)/(0.8) # per hour
                               gamma=0.03,  # np.log(2)/(22.8) # per hour
                               k_m0_scale=502,  # a.u/h
                               t_0=0.9,  # in hours
                               offset=8,  # a.u
                               sigma=np.sqrt(0.001)  # must be positive
                               ) -> None:
        # give parameters names and display them
        theta_population = np.array([delta, gamma, k_m0_scale, t_0, offset, sigma])
        df_params = pd.DataFrame(theta_population.reshape(1, theta_population.size), columns=self.param_names)
        display(df_params)

        # simulate some measurements
        t_points = np.linspace(start=0, stop=30, num=181, endpoint=True)
        sol_ana = ode_analytical_sol(t_points, theta_population[0], theta_population[1], theta_population[2],
                                     theta_population[3])
        y_sim_ana = measurement(sol_ana, theta_population[4])

        # Plot
        figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axis[0].plot(t_points, sol_ana[0], 'b', label='$m$ simulation')
        axis[0].plot(t_points, y_sim_ana, 'g', label='$y$ simulation without noise')
        axis[0].set_xlabel('$t\, [h]$')
        axis[0].set_ylabel('fluorescence intensity [a.u.]')
        axis[0].set_title('Simulation of $m$ and $y$')
        axis[0].legend()

        axis[1].plot(t_points, sol_ana[1], 'r', label='$p$ simulation')
        axis[1].set_xlabel('$t\, [h]$')
        axis[1].set_ylabel('fluorescence intensity [a.u.]')
        axis[1].set_yscale('log')
        axis[1].set_title('Simulation of $p$')
        axis[1].legend()

        plt.show()
        return
