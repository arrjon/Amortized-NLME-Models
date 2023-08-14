import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
from functools import partial
from typing import Optional

from models.base_nlme_model import NlmeBaseAmortizer
from inference.helper_functions import load_single_cell_data, load_multi_experiment_data

from bayesflow.simulation import Simulator

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

jlPkg.activate("models/SimulatorFroehlich")
jl.seval("using SimulatorFroehlich")


# Define ODE model
def ode_model_large(m: float, e: float, r: float, p: float,
                    t: float, delta1_m0: float, delta2: float, e0_m0: float, k2_m0_scale: float,
                    k2: float, k1_m0: float, r0_m0: float, gamma: float, t_0: float) -> np.ndarray:
    # m: mRNA, e: enzyme, r: repressor, p: protein
    if t >= t_0:
        m_dot = -delta1_m0 * m * e - k1_m0 * m * r + k2 * (r0_m0 - r)
        e_dot = delta1_m0 * m * e - delta2 * (e0_m0 - e)
        r_dot = k2 * (r0_m0 - r) - k1_m0 * m * r
        p_dot = k2_m0_scale * (r0_m0 - r) - gamma * p
        return np.array([m_dot, e_dot, r_dot, p_dot])
    else:
        return np.zeros(4)


# define measurement function
def measurement(y: np.ndarray, offset: float) -> np.ndarray:  # y: state vector
    p = y[3]  # p: protein concentration
    return np.log(p + offset)


# define the noise function for one trajectory
def add_noise(y: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:  # y: measurement, sigma: noise level
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=sigma, size=y.size)
    y_noisy = y + noise
    return y_noisy


def single_simulator(params: np.ndarray, n_obs: int, with_noise: bool, use_julia: bool) -> np.ndarray:
    """
    Simulate ODE model
    Args:
        params: np.ndarray - parameter vector (n_params,)
        n_obs: int - number of observations to generate
        with_noise: bool - if noise should be added to the simulation or not
        use_julia: bool - if Julia should be used to solve the ODE

    Returns: np.ndarray - simulated data (n_obs,)

    """
    # extract parameters
    delta1_m0, delta2, e0_m0, k2_m0_scale, k2, k1_m0, r0_m0, gamma, t_0, offset, sigma = params

    t_points = np.linspace(start=1 / 6, stop=30, num=n_obs, endpoint=True)

    if use_julia:
        # solve ODE using Julia
        y_sim = jl.simulateLargeModel(t_0, delta1_m0, delta2, e0_m0, k2_m0_scale, k2, k1_m0,
                                      r0_m0, gamma, offset, sigma).to_numpy()
    else:
        # solve ODE using scipy
        def rhs(t, x):
            return ode_model_large(m=x[0], e=x[1], r=x[2], p=x[3], t=t,
                                   delta1_m0=delta1_m0,
                                   delta2=delta2,
                                   e0_m0=e0_m0,
                                   k2_m0_scale=k2_m0_scale,
                                   k2=k2,
                                   k1_m0=k1_m0,
                                   r0_m0=r0_m0,
                                   gamma=gamma,
                                   t_0=t_0)

        # solve ODE
        try:
            x0 = np.array([1, e0_m0, r0_m0, 0])  # x = [m, e, r, p]
            with warnings.catch_warnings():
                # some samples will generate errors because of overflow
                warnings.filterwarnings("ignore", message='divide by zero encountered in scalar divide')
                warnings.filterwarnings("ignore", message='overflow encountered in multiply')
                warnings.filterwarnings("ignore", message='invalid value encountered in subtract')
                warnings.filterwarnings("ignore", message='divide by zero encountered in double_scalars')
                sol = solve_ivp(fun=rhs, t_span=(1 / 6, 30), y0=x0, t_eval=t_points, method='Radau')
            # apply measurement function
            y_sim = measurement(sol.y, offset)

            if not sol['success']:
                # no simulation was generated
                y_sim = measurement(np.ones(n_obs), offset)

        except (ValueError, TimeoutError) as e:
            # no simulation was generated
            y_sim = measurement(np.ones(n_obs), offset)

    # add noise on top of the simulation
    if with_noise:
        return add_noise(y_sim, sigma)  # sigma
    else:
        return y_sim


# a function which can simulate a batch of parameters or a single parameter set
def batch_simulator_pool(log_param_samples: np.ndarray, n_obs: int, with_noise: bool = True,
                         pool=None, use_julia: bool = True,
                         exp_func: str = 'exp') -> np.ndarray:
    """
    Simulate ODE model for multiple parameter sets

    param_samples: np.ndarray - (#simulations, #parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)
    pool: multiprocess.Pool - multiprocessing pool (optional)
    use_julia: bool - if Julia should be used to solve the ODE

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1)
    """
    n_sim = log_param_samples.shape[0]
    pool_func = partial(single_simulator, n_obs=n_obs, with_noise=with_noise, use_julia=use_julia)

    if exp_func == 'exp':
        param_samples = np.exp(log_param_samples)
    elif exp_func == 'power10':
        param_samples = np.power(10, log_param_samples)
    else:
        raise ValueError('exp_func must be "exp" or "power10"')

    if pool is not None:
        # multiprocess
        output = pool.map(pool_func, param_samples)
        sim_data = np.array(output).reshape((n_sim, n_obs, 1)).astype(np.float32)
    else:
        # iterate over parameter samples and simulate
        sim_data = np.zeros((n_sim, n_obs, 1), dtype=np.float32)
        for i, param in enumerate(param_samples):
            sim_data[i, :, 0] = pool_func(param)  # simulate one parameter set

    return sim_data


class FroehlichModelLarge(NlmeBaseAmortizer):
    def __init__(self, name: str = 'FroehlichModelLarge',
                 network_idx: int = -1,
                 load_best: bool = False,
                 pool=None):
        param_names = ['$\\delta_1 m_0$', '$\\delta_2$', '$e_0 m_0$', '$k_2 m_0 scale$', '$k_2$',
                       '$k_1 m_0$', '$r_0 m_0$', '$\\gamma$', '$t_0$', '$offset$', '$\\sigma$']

        self.pool = pool  # if simulation should be done in parallel
        if pool is not None:
            print('Pool:', pool)

        # define prior values (for log-parameters)
        prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -6, 0, 2, -1])
        prior_diag = 2 * np.ones(prior_mean.size)
        prior_diag[0] = 5  # same as in small model
        prior_diag[1] = 5  # same as in small model
        prior_diag[3] = 1  # otherwise too many samples lead to overflow
        prior_diag[7] = 5  # same as in small model
        prior_diag[8] = 2  # same as in small model
        prior_diag[9] = 6  # same as in small model
        prior_diag[10] = 2  # same as in small model
        prior_cov = np.diag(prior_diag)

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         n_obs=180)

        print('Using the large Froehlich model')

    def load_trained_model(self, model_idx: int = -1, load_best: bool = False) -> Optional[str]:
        # simulation time with juliacall 66.29 hours / 8 kernels = 8.29 hours (1 Mio simulations)
        # (589.26 + 649.92 + 710.42 + 622.59 + 679.79 + 732.09 + 612.35 + 654.61 + 674.22) / 9
        # = 658.36 [min] = 10.97 [h]
        # total average time: 19.26 hours

        # load best
        if load_best:
            model_idx = 2

        # in general: posteriors rather wide
        if model_idx == 0:  # sigma off
            self.n_epochs = 1000
            self.n_coupling_layers = 7
            self.final_loss = -0.614
            self.training_time = 589.26
        elif model_idx == 1:  # sigma off
            self.n_epochs = 1000
            self.n_coupling_layers = 8
            self.final_loss = -0.309
            self.training_time = 649.92
        elif model_idx == 2:  # nice calibration
            self.n_epochs = 1000
            self.n_coupling_layers = 9
            self.final_loss = -0.638
            self.training_time = 710.42
        elif model_idx == 3:  # sigma a little bit off
            self.n_epochs = 1000
            self.n_coupling_layers = 7
            self.summary_only_lstm = False
            self.final_loss = -0.867
            self.training_time = 622.59
        elif model_idx == 4:  # sigma off
            self.n_epochs = 1000
            self.n_coupling_layers = 8
            self.summary_only_lstm = False
            self.final_loss = -0.844
            self.training_time = 679.79
        elif model_idx == 5:  # sigma off
            self.n_epochs = 1000
            self.n_coupling_layers = 9
            self.summary_only_lstm = False
            self.final_loss = -0.347
            self.training_time = 732.09
        elif model_idx == 6:  # nice calibration
            self.n_epochs = 1000
            self.n_coupling_layers = 7
            self.summary_loss_fun = 'MMD'
            self.final_loss = -0.173
            self.training_time = 612.35
        elif model_idx == 7:  # nice calibration
            self.n_epochs = 1000
            self.n_coupling_layers = 8
            self.summary_loss_fun = 'MMD'
            self.final_loss = -0.306
            self.training_time = 654.61
        elif model_idx == 8:  # sigma off
            self.n_epochs = 1000
            self.n_coupling_layers = 9
            self.summary_loss_fun = 'MMD'
            self.final_loss = -0.115
            self.training_time = 674.22
        else:
            return None

        model_name = f'amortizer-large-fro' \
                     f'-{self.n_coupling_layers}layers' \
                     f'{"-LSTM" if self.summary_only_lstm else ""}' \
                     f'{"-summary_loss" if self.summary_loss_fun is not None else ""}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def old_load_trained_model(self, model_idx: int = 0, load_best: bool = False) -> Optional[str]:
        # simulation time new 124818.4 minutes / 48 kernels = 2600.40 minutes = 43.4 hours (500.000 simulations)
        # simulation time in julia 90475.16 seconds / 8 kernels = 11309.395 seconds = 3.14 hours (1 Mio simulations)
        # simulation time with juliacall 66.29 hours / 8 kernels = 8.29 hours (1 Mio simulations)

        # load best
        if load_best:
            model_idx = 9  # 7  # 4  # 2

        if model_idx == 0:  # solve_ivp in python
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -10, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 5
            self.n_epochs = 500
            self.final_loss = -5.520
            self.training_time = 223.37
            model_name = 'amortizer-large-fro-5layers-LSTM-2'
        elif model_idx == 1:  # solve_ivp in python
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -10, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 6
            self.n_epochs = 500
            self.final_loss = -5.057
            self.training_time = 257.18
            model_name = 'amortizer-large-fro-6layers-LSTM-2'
        elif model_idx == 2:  # solve_ivp in python
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -10, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 7
            self.n_epochs = 500
            self.final_loss = -5.060
            self.training_time = 275.11
            model_name = 'amortizer-large-fro-7layers-LSTM-2'
        elif model_idx == 3:  # solve_ivp in python
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -10, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 7
            self.n_epochs = 500
            self.final_loss = -4.920
            self.training_time = 288.90
            model_name = 'amortizer-large-fro-7layers-LSTM-all'
        elif model_idx == 4:  # in julia
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -10, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 7
            self.n_epochs = 1000
            self.final_loss = -0.760
            self.training_time = 601.44  # 10.02 hours
            model_name = 'amortizer-large-fro-7layers-LSTM-new-julia'
        elif model_idx == 5:  # in julia
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -10, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 8
            self.n_epochs = 1000
            self.final_loss = -0.722
            self.training_time = 658.46  # 10.97 hours
            model_name = 'amortizer-large-fro-8layers-LSTM-new-julia'
        elif model_idx == 6:  # with juliacall
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -8, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 7
            self.n_epochs = 1000
            self.final_loss = -0.590
            self.training_time = 511.14  # 8.52 hours
            model_name = 'amortizer-large-fro-7layers-LSTM-new-prior'
        elif model_idx == 7:  # with juliacall
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -8, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 8
            self.n_epochs = 1000
            self.final_loss = -0.836
            self.training_time = 556.90  # 9.28 hours
            model_name = 'amortizer-large-fro-8layers-LSTM-new-prior'
        elif model_idx == 8:  # with juliacall
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -6, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 8
            self.n_epochs = 1000
            self.final_loss = -1.423
            self.training_time = 540.53  # 9.01 hours
            model_name = 'amortizer-large-fro-8layers-1000-LSTM-new-prior2-gamma'
        elif model_idx == 9:  # with juliacall
            self.prior_mean = np.array([0, -1, 0, 12, 0, 1, -3, -6, 0, 2, -1])
            prior_diag = 2 * np.ones(self.n_params)
            prior_diag[0] = 5  # same as in small model
            prior_diag[1] = 5  # same as in small model
            prior_diag[3] = 1  # otherwise too many samples lead to overflow
            prior_diag[7] = 5  # same as in small model
            prior_diag[8] = 2  # same as in small model
            prior_diag[9] = 6  # same as in small model
            prior_diag[10] = 2  # same as in small model
            self.prior_cov = np.diag(prior_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 9
            self.n_epochs = 1000
            self.final_loss = -1.635
            self.training_time = 606.99  # 10.12 hours
            model_name = 'amortizer-large-fro-9layers-1000-LSTM-new-prior-gamma'
        else:
            return None
        return model_name

    def build_simulator(self, with_noise: bool = True, use_julia: bool = True,
                        exp_func: str = 'exp') -> Simulator:
        # build simulator
        simulator = Simulator(batch_simulator_fun=partial(batch_simulator_pool,
                                                          n_obs=self.n_obs,
                                                          pool=self.pool,
                                                          use_julia=use_julia,
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

    def print_and_plot_example(self,  # define parameters as one array (values taken from the paper, beside noise)
                               delta1_m0: float = 1.2,  # [h]
                               delta2: float = 0.6,  # [h]
                               e0_m0: float = 0.85,  #
                               k2_m0_scale: float = 1e6,  # [a.u/h]
                               k2: float = 1.9,  # [1/h]
                               k1_m0: float = 5.5,  # [1/h]
                               r0_m0: float = 0.1,  # [1]
                               gamma: float = 0.01,  # [1/h]
                               t_0: float = 0.9,  # [h]
                               offset: float = 8.,  # [a.u]
                               sigma: float = np.sqrt(0.001)
                               ) -> None:
        theta_population = np.array([delta1_m0, delta2, e0_m0, k2_m0_scale, k2, k1_m0,
                                     r0_m0, gamma, t_0, offset, sigma])
        df_params = pd.DataFrame(theta_population.reshape(1, theta_population.size), columns=self.param_names)
        display(df_params)

        # make ODE a function of (t,x), theta is given
        def rhs(t, x):
            return ode_model_large(m=x[0], e=x[1], r=x[2], p=x[3], t=t,
                                   delta1_m0=delta1_m0,
                                   delta2=delta2,
                                   e0_m0=e0_m0,
                                   k2_m0_scale=k2_m0_scale,
                                   k2=k2,
                                   k1_m0=k1_m0,
                                   r0_m0=r0_m0,
                                   gamma=gamma,
                                   t_0=t_0)

        # simulate ODE
        x0 = np.array([1, e0_m0, r0_m0, 0])  # x = [m, e, r, p]
        n_obs = 180
        t_points = np.linspace(start=0, stop=30, num=n_obs, endpoint=True)
        sol = solve_ivp(fun=rhs, t_span=(0, 30), y0=x0, t_eval=t_points, method='Radau')
        y_sim = measurement(sol.y, offset)
        y_sim_noisy = add_noise(y_sim, sigma, seed=0)

        # Plot
        figure, axis = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))

        axis[0].plot(sol.t, sol.y[0], label='$m$ simulation')
        axis[0].set_title('$m$ simulation')
        axis[1].plot(sol.t, sol.y[1], label='$e$ simulation')
        axis[1].set_yscale('log')
        axis[1].set_title('$e$ simulation')
        axis[2].plot(sol.t, sol.y[2], label='$r$ simulation')
        axis[2].set_yscale('log')
        axis[2].set_title('$r$ simulation')
        axis[3].plot(sol.t, sol.y[3], label='$p$ simulation')
        axis[3].set_yscale('log')
        axis[3].set_title('$p$ simulation')
        axis[4].plot(sol.t, np.exp(y_sim_noisy), label='$y$ simulation with noise')
        axis[4].set_yscale('log')
        axis[4].set_title('simulation with noise')

        axis[0].set_xlabel('$t\, [h]$')
        axis[1].set_xlabel('$t\, [h]$')
        axis[2].set_xlabel('$t\, [h]$')
        axis[3].set_xlabel('$t\, [h]$')
        axis[4].set_xlabel('$t\, [h]$')
        axis[0].set_ylabel('fluorescence intensity [a.u.]')

        plt.show()
        return
