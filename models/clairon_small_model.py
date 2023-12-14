#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the Clairon NLME Model

import itertools
import os
import pathlib
from datetime import datetime
from functools import partial
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayesflow.simulation import Simulator, Prior
from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import convert as jlconvert
from scipy.stats import qmc

from inference.base_nlme_model import NlmeBaseAmortizer, configure_input, batch_gaussian_prior, batch_uniform_prior

env = os.path.join(pathlib.Path(__file__).parent.resolve(), 'SimulatorSmallClairon')
jlPkg.activate(env)
jl.seval("using SimulatorSmallClairon")


def measurement_model(y: np.ndarray, censoring: float = 2500., threshold: float = 0.001) -> np.ndarray:
    """
    Applies a measurement function to a given variable.

    Parameters:
    y (np.ndarray): Measurements of the variable to which the measurement function should be applied.
    censoring (float): Right-censoring value for the measurement function.
    threshold (float): Left-censoring value for the measurement function.

    Returns:
    np.ndarray: The result of applying the measurement function to the input variable.
    """
    y[y < threshold] = threshold
    y[y > censoring] = censoring
    return y


def prop_noise(y: np.ndarray, error_params: np.ndarray) -> np.ndarray:
    """
    Proportional error model for given trajectories.

    Parameters:
    y (np.ndarray): The trajectory to which noise should be added.
    error_params (np.ndarray): Standard deviations of the Gaussian noise: (a+by)*noise where noise is standard normal.

    Returns:
    np.ndarray: The noisy trajectories.
    """
    noise = np.random.normal(loc=0, scale=1, size=y.shape)
    return y + (error_params[0] + error_params[1] * y) * noise


def batch_simulator(param_batch: np.ndarray,
                    t_measurements: Optional[np.ndarray] = None,
                    n_measurements: int = 4,
                    t_doses: Optional[np.ndarray] = None,
                    dose_amount: float = 1.,
                    with_noise: bool = True,
                    convert_to_bf_batch: bool = True
                    ) -> np.ndarray:
    """
    Simulate a batch of parameter sets.

    param_batch: np.ndarray - (#simulations, #parameters) or (#parameters)

    If time points for measurements and dosing events are not given, they are sampled.
    If convert_to_bf_batch is True, the output is in the format used by the bayesflow summary model, else only the
        measurements are returned.
    """
    # sample the measurement time points
    if t_measurements is None:
        t_measurements = get_measurement_time_points(n_measurements)
    # sample the dosing time points
    if t_doses is None:
        t_doses = get_dosing_time_points()
    # starting values
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    # convert to julia types
    jl_x0 = jlconvert(jl.Vector[jl.Float64], x0)
    jl_dose_amount = jlconvert(jl.Float64, dose_amount)
    jl_dosetimes = jlconvert(jl.Vector[jl.Float64], t_doses)
    jl_t_measurement = jlconvert(jl.Vector[jl.Float64], t_measurements)

    # simulate batch
    if param_batch.ndim == 1:  # so not (batch_size, params)
        # just a single parameter set
        param_batch = param_batch[np.newaxis, :]
    n_sim = param_batch.shape[0]
    if convert_to_bf_batch:
        # create output batch containing all information for bayesflow summary model
        output_batch = np.zeros((n_sim, t_measurements.size + t_doses.size, 3),
                                dtype=np.float32)
    else:
        # just return the simulated data
        output_batch = np.zeros((n_sim, t_measurements.size))

    for pars_i, log_params in enumerate(param_batch):
        # convert to julia types
        jl_parameter = jlconvert(jl.Vector[jl.Float64], np.exp(log_params[:-2]))

        # simulate
        y_sim = jl.simulateSmallClairon(jl_parameter, jl_x0,
                                        jl_dose_amount, jl_dosetimes,
                                        jl_t_measurement).to_numpy()

        # apply noise
        if with_noise:
            y_sim = prop_noise(y_sim, error_params=np.exp(log_params[-2:]))

        # applying censoring and log-transformation
        y_sim = measurement_model(y_sim)

        # reshape the data to fit in one numpy array
        if convert_to_bf_batch:
            output_batch[pars_i, :, :] = convert_to_bf_format(y=y_sim,
                                                              t_measurements=t_measurements,
                                                              dose_amount=dose_amount,
                                                              doses_time_points=t_doses)
        else:
            output_batch[pars_i, :] = y_sim

    if n_sim == 1:
        # remove batch dimension
        return output_batch[0]
    return output_batch


def simulate_single_patient(param_batch: np.ndarray,
                            patient_data: np.ndarray,
                            full_trajectory: bool = False,
                            with_noise: bool = False,
                            convert_to_bf_batch: bool = False
                            ) -> np.ndarray:
    """uses the batch simulator to simulate a single patient"""
    y, t_measurements, doses_time_points, dose_amount = convert_bf_to_observables(patient_data)
    if full_trajectory:
        t_measurements = np.linspace(0, t_measurements[-1], 100)

    y_sim = batch_simulator(param_batch,
                            t_measurements=t_measurements,
                            t_doses=doses_time_points,
                            dose_amount=dose_amount,
                            with_noise=with_noise,
                            convert_to_bf_batch=convert_to_bf_batch)
    return y_sim


def convert_to_bf_format(y: np.ndarray,
                         t_measurements: np.ndarray,
                         dose_amount: float,
                         doses_time_points: np.ndarray,
                         scaling_time: float = 500.
                         ) -> np.ndarray:
    """
    converts all data to the format used by the bayesflow summary model
        (np.log(y), timepoints / scaling_time, 0) concatenated with (log(dose_amount), timepoints / scaling_time, 1)
    and then sort by time
    """
    # reshape the data to fit in one numpy array
    measurements = np.stack((np.log(y),
                             t_measurements / scaling_time,
                             np.zeros(t_measurements.size)),
                            axis=1)
    doses = np.stack(([np.log(dose_amount)] * doses_time_points.size,
                      doses_time_points / scaling_time,
                      np.ones(doses_time_points.size)),
                     axis=1)
    bf_format = np.concatenate((measurements, doses), axis=0)
    bf_format_sorted = bf_format[bf_format[:, 1].argsort()]
    return bf_format_sorted


def convert_bf_to_observables(output: np.ndarray,
                              scaling_time: float = 500.
                              ) -> (np.ndarray, np.ndarray, float, np.ndarray):
    """
    converts data in the bayesflow summary model format back to observables
        (y, timepoints / scaling_time, 0) concatenated with (log(dose_amount), timepoints / scaling_time, 1)
    """
    measurements = output[np.where(output[:, 2] == 0)]
    y = np.exp(measurements[:, 0])
    t_measurements = measurements[:, 1] * scaling_time

    doses = output[np.where(output[:, 2] == 1)]
    dose_amount = np.exp(doses[0, 0])
    doses_time_points = doses[:, 1] * scaling_time
    return y, t_measurements, doses_time_points, dose_amount


class ClaironSmallModel(NlmeBaseAmortizer):
    def __init__(self, name: str = 'ClaironModel', network_idx: int = -1, load_best: bool = False,
                 prior_type: str = 'normal',  # normal or uniform
                 n_measurements: int = 4
                 ):
        # define names of parameters
        param_names = ['fM2', 'fM3', 'theta', 'deltaV', 'deltaS',
                       'error_constant', 'error_prop']

        # define prior values (for log-parameters)
        prior_mean = np.log([4, 12, 18, 3, 0.01, 0.1, 0.1])
        prior_cov = np.diag(np.array([3, 3, 3, 3, 3, 3, 3]) * 2)

        # define prior bounds for uniform prior
        # self.prior_bounds = np.array([[-10, 5], [-5, 10], [-5, 10], [-20, 0], [-10, 0], [-10, 0], [-10, 0]])
        self.prior_bounds = np.array([[-5, 7], [-5, 7], [-5, 7], [-5, 7], [-5, 0], [-5, 0], [-5, 0]])
        self.n_measurements = n_measurements

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         prior_type=prior_type,
                         max_n_obs=7)  # 4 measurements, 3 doses

        # define simulator
        batch_simulator_fun = partial(batch_simulator,
                                      n_measurements=self.n_measurements)
        self.simulator = Simulator(batch_simulator_fun=batch_simulator_fun)

        print(f'Using the model {name}')

    def _build_prior(self) -> None:
        """
        Build prior distribution.
        Returns: prior, configured_input - prior distribution and function to configure input

        """
        if self.prior_type == 'uniform':
            print('Using uniform prior')
            print(self.prior_bounds)
            self.prior = Prior(batch_prior_fun=partial(batch_uniform_prior,
                                                       prior_bounds=self.prior_bounds),
                               param_names=self.log_param_names)
            self.prior_mean = (np.diff(self.prior_bounds) / 2).flatten() + self.prior_bounds[:, 0]
            self.prior_std = (np.diff(self.prior_bounds) / 2).flatten() ** 2 / 12  # uniform variance
            self.prior_cov = np.diag(self.prior_std ** 2)

        elif self.prior_type == 'normal':
            print('Using normal prior')
            print('prior mean:', self.prior_mean)
            print('prior covariance diagonal:', self.prior_cov.diagonal())
            self.prior = Prior(batch_prior_fun=partial(batch_gaussian_prior,
                                                       mean=self.prior_mean,
                                                       cov=self.prior_cov),
                               param_names=self.log_param_names)
        else:
            raise ValueError('Unknown prior type')

        self.configured_input = partial(configure_input,
                                        prior_means=self.prior_mean,
                                        prior_stds=self.prior_std)
        return

    def load_amortizer_configuration(self, model_idx: int = 0, load_best: bool = False) -> str:
        self.n_epochs = 500
        self.summary_dim = self.n_params * 2
        self.n_obs_per_measure = 3  # time and measurement + event type (measurement = 0, dosing = 1)

        # load best
        if load_best:
            if self.prior_type == 'normal':
                model_idx = 6
                # amortizer-clairon-normal-sequence-summary-LSTM-8layers-3coupling-affine-500epochs -> 6
            else:
                model_idx = 11
                # amortizer-clairon-uniform-sequence-summary-Bi-LSTM-7layers-3coupling-spline-500epochs -> 11

        if self.n_measurements == 4:
            bidirectional_LSTM = [False, True]
            n_coupling_layers = [7, 8]
            n_dense_layers_in_coupling = [2, 3]
            coupling_design = ['affine', 'spline']
            summary_network_type = ['sequence']
        else:
            bidirectional_LSTM = [True]
            n_coupling_layers = [7, 8]
            n_dense_layers_in_coupling = [3]
            coupling_design = ['affine', 'spline']
            summary_network_type = ['sequence']
            if load_best:
                model_idx = 0

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design, summary_network_type))

        if model_idx >= len(combinations) or model_idx < 0:
            model_name = f'amortizer-clairon-{self.prior_type}' \
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

        model_name = f'amortizer-clairon-{self.prior_type}' \
                     f'-{self.summary_network_type}-summary' \
                     f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{self.n_epochs}epochs'
        if self.n_measurements != 4:
            model_name = "new-" + model_name
        return model_name

    def load_data(self,
                  n_data: Optional[int] = None,
                  load_covariates: bool = False,
                  synthetic: bool = False,
                  return_synthetic_params: bool = False,
                  synthetic_fixed_indices: Optional[np.ndarray] = None,
                  seed: int = 0) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if synthetic:
            assert isinstance(n_data, int)
            np.random.seed(seed)
            # mean and variances (if existent) taken from the clairon paper
            clairon_mean = np.log(np.array([4.5, 12.4, 18.7, 2.7, 0.01, 0.01, 0.2]))
            #clairon_mean[:-2] -= 1
            clairon_cov = np.diag(np.array([0.8, 0.2, 0.5, 0.1, 0.3, 0., 0.]) ** 2)  # no fixed parameters
            params = batch_gaussian_prior(mean=clairon_mean,
                                          cov=clairon_cov,
                                          batch_size=n_data)
            if synthetic_fixed_indices is not None:
                # fix parameters to test identifiability
                params[:, synthetic_fixed_indices] = clairon_mean[synthetic_fixed_indices]
            patients_data = batch_simulator(param_batch=params,
                                            n_measurements=self.n_measurements)
            if return_synthetic_params:
                return patients_data, params
            return patients_data

        df_dosages = pd.read_csv("../data/clairon/bf_data_dosages.csv")
        df_dosages.drop_duplicates(inplace=True)
        df_dosages.set_index('id_hcw', inplace=True)  # only now, so duplicates are removed correctly

        df_measurements = pd.read_csv("../data/clairon/bf_data_measurements.csv", index_col=0)
        df_measurements['gender_code'] = df_measurements['gender'].astype('category').cat.codes
        df_measurements['age_standardized'] = np.log(df_measurements['age'])

        patients_data = []
        patients_covariates = []
        for p_id, patient in enumerate(df_measurements.index.unique()):
            m_times = df_measurements.loc[patient, ['days_after_first_dose']].values.flatten()
            y = df_measurements.loc[patient, ['res_serology']].values.flatten()
            d_times = df_dosages.loc[patient, ['day_1dose', 'day_2dose', 'day_3dose']].values
            covariates = df_measurements.loc[patient, ['age', 'gender_code']].values[0].flatten()

            # log-transform data
            y = measurement_model(y)

            data = convert_to_bf_format(y=y,
                                        t_measurements=m_times,
                                        dose_amount=1.,
                                        doses_time_points=d_times)
            if np.isnan(data).any():
                # remove patients with nan values # todo: nan values
                continue
            patients_data.append(data)
            patients_covariates.append(covariates)

            if n_data is not None and len(patients_data) == n_data:
                break

        if load_covariates:
            return np.stack(patients_data, axis=0), np.stack(patients_covariates, axis=0)
        return np.stack(patients_data, axis=0)

    def plot_example(self, params: Optional[np.ndarray] = None) -> None:
        """Plots an individual trajectory of an individual in this model."""
        if params is None:
            params = self.prior(1)['prior_draws'][0]

        output = batch_simulator(params, n_measurements=self.n_measurements)
        _ = self.prepare_plotting(output, params)

        plt.title(f'Patient Simulation')
        plt.legend()
        plt.show()
        return

    @staticmethod
    def prepare_plotting(data: np.ndarray, params: np.ndarray, ax: Optional[plt.Axes] = None,
                         with_noise: bool = False) -> plt.Axes:
        # convert BayesFlow format to observables
        y, t_measurements, doses_time_points, dose_amount = convert_bf_to_observables(data)
        t_measurement_full = np.linspace(0, t_measurements[-1] + 100, 100)

        # simulate data
        sim_data = batch_simulator(param_batch=params,
                                   t_measurements=t_measurement_full,
                                   t_doses=doses_time_points,
                                   with_noise=with_noise,
                                   convert_to_bf_batch=False)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)

        if len(params.shape) == 1:  # so not (batch_size, params)
            # just a single parameter set
            # plot simulated data
            ax.plot(t_measurement_full, sim_data, 'b', label='simulation')
        else:
            # calculate median and quantiles
            y_median = np.median(sim_data, axis=0)
            y_quantiles = np.percentile(sim_data, [2.5, 97.5], axis=0)

            # plot simulated data
            ax.fill_between(t_measurement_full, y_quantiles[0], y_quantiles[1],
                            alpha=0.2, color='orange', label='95% quantiles')
            ax.plot(t_measurement_full, y_median, 'b', label='median')

        # plot observed data
        ax.scatter(t_measurements, y, color='b', label='measurements')

        # plot dosing events
        ax.vlines(doses_time_points, 0, 2500,
                  color='grey', alpha=0.5, label='dosing events')

        # plot censoring
        ax.hlines(2500, xmin=0, xmax=t_measurement_full[-1], linestyles='--',
                  color='green', label='censoring')

        ax.set_xlabel('Time (in days)')
        ax.set_ylabel('Measurements')
        return ax


def get_measurement_time_points(n_measurements: int) -> np.ndarray:
    """sample the measurement time points"""
    if n_measurements == 4:  # real data looks like this
        # list of possible measurement time points taken from data
        zero_measurement = np.array([23, 29, 34, 41, 43, 48, 49, 50, 52, 53, 54, 55, 56,
                                     57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                     70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                                     83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                                     96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109,
                                     110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 124, 125, 126,
                                     127, 134, 136, 142, 143, 153])
        # frequency of the measurement time points
        p_zero_measurement = np.array([8, 4, 4, 8, 4, 4, 8, 4, 4, 4, 12, 8, 8,
                                       32, 40, 24, 52, 60, 68, 52, 56, 116, 96, 92, 100, 124,
                                       152, 96, 88, 136, 136, 108, 100, 56, 80, 56, 72, 56, 44,
                                       44, 28, 44, 28, 60, 24, 64, 32, 52, 20, 32, 36, 24,
                                       28, 20, 40, 24, 24, 12, 16, 12, 16, 16, 16, 12, 20,
                                       4, 8, 12, 4, 12, 8, 8, 12, 4, 8, 8, 4, 4,
                                       4, 4, 4, 4, 8, 4])
        # list of possible measurement time points taken from data
        fist_diff_measurement = np.array([5, 9, 13, 15, 19, 20, 22, 27, 28, 30, 33, 34, 35,
                                          36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49,
                                          50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                          63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                          76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                                          89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                                          102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                                          115, 116, 117, 118, 119, 120, 124, 127, 128, 130, 132, 133, 134,
                                          140, 147, 153, 158, 184, 195, 200, 217, 219])
        # frequency of the measurement time points
        p_fist_diff_measurement = np.array([4, 4, 4, 4, 4, 8, 4, 4, 16, 4, 4, 12, 12, 4, 4, 8, 12,
                                            16, 20, 4, 20, 8, 20, 16, 8, 20, 28, 24, 20, 16, 24, 28, 56, 40,
                                            48, 44, 68, 36, 76, 76, 88, 92, 44, 56, 44, 64, 72, 68, 72, 32, 44,
                                            52, 80, 68, 28, 48, 16, 24, 28, 32, 56, 36, 44, 36, 28, 44, 68, 60,
                                            72, 72, 52, 56, 40, 52, 48, 24, 8, 20, 16, 12, 16, 20, 24, 8, 4,
                                            20, 12, 12, 16, 8, 16, 20, 4, 8, 16, 4, 12, 8, 4, 8, 12, 8,
                                            12, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4])
        # list of possible measurement time points taken from data
        second_diff_measurement = np.array([7, 9, 22, 26, 27, 28, 30, 32, 33, 34, 35, 37, 38,
                                            40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53,
                                            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                                            67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                            81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                                            94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                                            107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                            120, 121, 122, 123, 125, 126, 127, 128, 129, 131, 132, 133, 134,
                                            136, 138, 139, 140, 141, 142, 146, 148, 149, 150, 151, 153, 154,
                                            155, 156, 159, 162, 163, 173, 174, 177, 182, 201])
        # frequency of the measurement time points
        p_second_diff_measurement = np.array([4, 4, 4, 8, 4, 4, 12, 4, 4, 4, 8, 12, 8,
                                              12, 12, 4, 8, 8, 8, 4, 16, 8, 24, 20, 20, 8,
                                              8, 8, 24, 20, 20, 12, 16, 28, 48, 16, 20, 32, 68,
                                              36, 28, 24, 32, 16, 52, 44, 68, 36, 48, 68, 52, 20,
                                              40, 36, 52, 56, 32, 44, 52, 48, 40, 56, 100, 120, 108,
                                              56, 60, 84, 44, 72, 44, 72, 44, 32, 48, 32, 28, 24,
                                              32, 28, 16, 32, 44, 48, 24, 8, 12, 16, 12, 16, 8,
                                              28, 8, 4, 16, 4, 24, 12, 8, 4, 4, 4, 4, 4,
                                              8, 12, 8, 12, 8, 4, 4, 4, 4, 4, 4, 4, 4,
                                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        # list of possible measurement time points taken from data
        third_diff_measurement = np.array([10, 15, 17, 20, 21, 25, 28, 30, 32, 33, 34, 35, 37,
                                           39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                           53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                           66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                                           79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
                                           92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                                           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                                           118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                                           132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 148, 149,
                                           150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 162, 166,
                                           167, 168, 171, 172, 175, 176, 178, 179, 181, 182, 190, 195, 197,
                                           217, 221, 239, 248, 250])
        # frequency of the measurement time points
        p_third_diff_measurement = np.array([4, 4, 4, 4, 8, 4, 8, 4, 4, 4, 12, 12, 4, 8, 4, 4, 20,
                                             12, 8, 4, 8, 16, 24, 24, 12, 16, 16, 12, 20, 44, 24, 12, 32, 36,
                                             20, 4, 56, 44, 32, 52, 68, 24, 56, 72, 60, 36, 56, 52, 40, 76, 68,
                                             56, 48, 36, 40, 32, 20, 60, 36, 28, 32, 48, 44, 76, 84, 56, 28, 56,
                                             24, 20, 48, 36, 24, 48, 8, 24, 32, 44, 28, 28, 28, 36, 12, 16, 12,
                                             24, 32, 24, 8, 4, 20, 8, 20, 12, 12, 16, 4, 12, 12, 8, 24, 8,
                                             8, 8, 28, 16, 4, 12, 12, 8, 16, 4, 4, 12, 4, 4, 4, 4, 4,
                                             4, 4, 8, 12, 8, 4, 4, 4, 4, 4, 4, 8, 4, 8, 4, 4, 4,
                                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8])

        zero_m = np.random.choice(zero_measurement,
                                  p=p_zero_measurement / p_zero_measurement.sum())
        fist_diff_m = np.random.choice(fist_diff_measurement,
                                       p=p_fist_diff_measurement / p_fist_diff_measurement.sum())
        second_diff_m = np.random.choice(second_diff_measurement,
                                         p=p_second_diff_measurement / p_second_diff_measurement.sum())
        third_diff_m = np.random.choice(third_diff_measurement,
                                        p=p_third_diff_measurement / p_third_diff_measurement.sum())
        t_measurements = np.array([zero_m, zero_m + fist_diff_m, zero_m + fist_diff_m + second_diff_m,
                                   zero_m + fist_diff_m + second_diff_m + third_diff_m])
    else:
        sampler = qmc.Halton(d=1)
        sample = sampler.random(n=n_measurements)
        t_measurements = qmc.scale(sample, 0, 500).flatten()
    return t_measurements


def get_dosing_time_points() -> np.ndarray:
    """sample the dosing time points"""
    # list of possible dosing time points taken from data
    day_2dose_data = np.array([22, 23, 24, 26, 28, 30, 34, 41, 42, 96, 215, 231, 253, 260])
    # frequency of the measurement time points
    p2_in_data = np.array([1, 724, 10, 1, 1, 15, 1, 1, 1, 1, 1, 1, 1, 1])

    # list of possible dosing time points taken from data
    day_3dose_after2_data = np.array([141., 209., 210., 222., 224., 226., 227., 229., 231., 233., 241.,
                                      244., 245., 246., 247., 248., 250., 252., 253., 255., 257., 258.,
                                      259., 260., 262., 263., 264., 265., 266., 267., 268., 269., 270.,
                                      271., 272., 273., 274., 275., 276., 277., 278., 279., 280., 281.,
                                      282., 283., 284., 285., 286., 287., 288., 289., 290., 291., 292.,
                                      293., 294., 295., 296., 297., 298., 299., 300., 301., 302., 303.,
                                      304., 305., 306., 307., 308., 309., 310., 311., 312., 313., 314.,
                                      315., 316., 317., 318., 319., 320., 321., 322., 323., 324., 325.,
                                      350., 367., 369., 385., 403., 416., 432., 447., 454., 460., 468.,
                                      491.])
    # frequency of the measurement time points
    p3_in_data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 4,
                           1, 1, 4, 2, 2, 1, 2, 5, 2, 3, 3, 2, 2, 4, 7, 9, 9,
                           5, 8, 11, 5, 6, 8, 13, 10, 13, 14, 10, 14, 13, 11, 23, 13, 7,
                           12, 10, 10, 8, 11, 12, 10, 8, 10, 19, 20, 15, 19, 23, 17, 16, 16,
                           16, 27, 27, 12, 18, 26, 21, 13, 13, 7, 11, 13, 4, 3, 5, 4, 4,
                           2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # sample the measurement time points as in the data
    t_2dose = np.random.choice(day_2dose_data, p=p2_in_data / p2_in_data.sum())
    t_3dose = t_2dose + np.random.choice(day_3dose_after2_data, p=p3_in_data / p3_in_data.sum())
    return np.array([2., t_2dose, t_3dose])
