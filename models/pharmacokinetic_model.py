#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the pharmacokinetic NLME Model

import itertools
import os
import pathlib
from datetime import datetime
from functools import partial
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesflow.simulation import Simulator

from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import convert as jlconvert

from inference.base_nlme_model import NlmeBaseAmortizer, batch_gaussian_prior

env = os.path.join(pathlib.Path(__file__).parent.resolve(), 'SimulatorPharma')
jlPkg.activate(env)
# jlPkg.activate("models/SimulatorPharma")
jl.seval("using SimulatorPharma")


def measurement_model(a: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    if a.shape[1] == 5:
        y = a[:, 1:3].copy()
    else:
        y = a
    # log_y[y < threshold] = threshold
    y[y < threshold] = threshold  # nonmem uses np.exp(threshold), but wrong?
    log_y = np.log(y)
    return log_y


def add_noise(y: np.ndarray, theta_12: float, theta_13: float, sigma: float = 1) -> np.ndarray:
    eps = np.random.normal(loc=0, scale=sigma, size=y.shape)
    y[:, 0] += theta_12 * eps[:, 0]
    y[:, 1] += theta_13 * eps[:, 1]
    return y


def batch_simulator(param_batch: np.ndarray,
                    t_measurement: Optional[np.ndarray] = None,  # one list for all samples
                    t_doses: Optional[np.ndarray] = None,  # one list for all samples
                    wt: Optional[float] = None,  # one float for all samples
                    dos: Optional[float] = None,  # one float for all samples
                    with_noise: bool = True,
                    convert_to_bf_batch: bool = True) -> np.ndarray:
    """
    Simulate a batch of parameter sets.

    param_batch: np.ndarray - (#simulations, #parameters) or (#parameters)

    If time points for measurements and dosing events are not given, they are sampled.
    If convert_to_bf_batch is True, the output is in the format used by the bayesflow summary model, else only the
        measurements are returned.
    """
    # sample the measurement time points
    if t_measurement is None:
        t_measurement = get_measurement_regime()
    if t_doses is None:
        t_doses = get_dosing_regim(last_measurement_time=float(t_measurement[-1]))
    if wt is None or dos is None:
        wt, dos = get_covariates()

    # convert float to julia float
    jl_wt = jlconvert(jl.Float64, wt)
    jl_dos = jlconvert(jl.Float64, dos)

    # simulate batch
    if param_batch.ndim == 1:  # so not (batch_size, params)
        # just a single parameter set
        param_batch = param_batch[np.newaxis, :]
    n_sim = param_batch.shape[0]

    if convert_to_bf_batch:
        # create output batch containing all information for bayesflow summary model
        output_batch = np.zeros((n_sim, t_measurement.size + t_doses.size, 4),
                                dtype=np.float32)
    else:
        # just return the simulated data
        output_batch = np.zeros((n_sim, t_measurement.size, 2))

    for pars_i, log_params in enumerate(param_batch):
        # simulate the data
        (theta_1, theta_2, theta_4, theta_5, theta_6, theta_7,
         theta_7, theta_10, theta_12, theta_13, eta_4) = np.exp(log_params)
        jl_parameter = jlconvert(jl.Vector[jl.Float64], np.array([theta_1, theta_2, theta_4,
                                                                  theta_5, theta_6, theta_7,
                                                                  theta_7, theta_10, eta_4]))

        # convert to julia types
        jl_t_doses = jlconvert(jl.Vector[jl.Float64], t_doses)
        jl_t_measurement = jlconvert(jl.Vector[jl.Float64], t_measurement)
        # simulate
        a_sim = jl.simulatePharma(jl_parameter,
                                  jl_wt,
                                  jl_dos,
                                  jl_t_doses,
                                  jl_t_measurement).to_numpy().T

        # convert to measurements
        y_sim_log = measurement_model(a_sim)
        if with_noise:
            y_sim_log = add_noise(y_sim_log, theta_12=theta_12, theta_13=theta_13)

        # reshape the data to fit in one numpy array
        if convert_to_bf_batch:
            output_batch[pars_i, :, :] = convert_to_bf_format(y=y_sim_log,
                                                              t_measurements=t_measurement,
                                                              doses_time_points=t_doses,
                                                              dos=dos,
                                                              wt=wt)
        else:
            # still on log scale, since error model is additive on log scale
            output_batch[pars_i, :] = y_sim_log

    if n_sim == 1:
        # remove batch dimension
        return output_batch[0]
    return output_batch


def simulate_single_patient(param_batch: np.ndarray,
                            patient_data: np.ndarray,
                            full_trajectory: bool = False,
                            with_noise: bool = False,
                            convert_to_bf_batch: bool = False) -> np.ndarray:
    """uses the batch simulator to simulate a single patient"""
    y, t_measurements, doses_time_points, dos, wt = convert_bf_to_observables(patient_data)
    if full_trajectory:
        t_measurements = np.linspace(0, t_measurements[-1], 100)

    y_sim = batch_simulator(param_batch,
                            t_measurement=t_measurements,
                            t_doses=doses_time_points,
                            wt=wt,
                            dos=dos,
                            with_noise=with_noise,
                            convert_to_bf_batch=convert_to_bf_batch)
    return y_sim


def convert_to_bf_format(y: np.ndarray,
                         t_measurements: np.ndarray,
                         doses_time_points: np.ndarray,
                         dos: float,
                         wt: float,
                         scaling_time: float = 4000.) -> np.ndarray:
    """
       converts data in the bayesflow summary model format back to observables
           (y1, y2, timepoints / scaling_time, 0) concatenated with
           (log10(dose_amount), log10(wt), timepoints / scaling_time, 1)
        and then sort by time
    """
    # reshape the data to fit in one numpy array
    measurements = np.concatenate((y,
                                   t_measurements[:, np.newaxis] / scaling_time,
                                   np.zeros((t_measurements.size, 1))),
                                  axis=1)
    doses = np.stack(([np.log10(dos)] * doses_time_points.size,
                      [np.log10(wt)] * doses_time_points.size,
                      doses_time_points / scaling_time,
                      np.ones(doses_time_points.size)),
                     axis=1)
    bf_format = np.concatenate((measurements, doses), axis=0)
    bf_format_sorted = bf_format[bf_format[:, 2].argsort()]
    return bf_format_sorted


def convert_bf_to_observables(output: np.ndarray,
                              scaling_time: float = 4000.) -> (np.ndarray, np.ndarray, np.ndarray, float, float):
    """
    Convert the output of the simulator to a reasonable format for plotting
    Args:
        output: output of the simulator
        scaling_time: scaling time for the output

    Returns: y, y_time_points, doses_time_points, DOS, wt

    """
    measurements = output[np.where(output[:, 3] == 0)]
    y = measurements[:, :2]
    t_measurements = measurements[:, 2] * scaling_time

    doses = output[np.where(output[:, 3] == 1)]
    dos = np.power(10, doses[0, 0])
    wt = np.power(10, doses[0, 1])
    doses_time_points = doses[:, 2] * scaling_time

    return y, t_measurements, doses_time_points, dos, wt


class PharmacokineticModel(NlmeBaseAmortizer):
    def __init__(self, name: str = 'PharmacokineticModel', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\\theta_1$',
                       '$\\theta_2-\eta_1$',
                       '$\\theta_4-\eta_3$',
                       '$\\theta_5$',
                       '$\\theta_6-\eta_2$',
                       '$\\theta_7$',
                       '$\\theta_8$',
                       '$\\theta_{10}$',
                       '$\\theta_{12}$',
                       '$\\theta_{13}$',
                       '$\eta_4$',
                       ]

        # define prior values (for log-parameters)
        prior_mean = np.array([-5., 6.5, 2.5, 2.5, 6.5, 0., 6.5, -3., -1., -1., 0.])
        prior_cov = np.diag(np.array([4.5, 1., 1., 1., 1., 1., 1., 4.5, 2., 2., 1.]))

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         prior_type='normal',
                         max_n_obs=230,
                         changeable_obs_n=True)  # 26 measurement max, 179 dosages max
        self.simulator = Simulator(batch_simulator_fun=batch_simulator)

        print('Using the PharmacokineticModel')

    def load_amortizer_configuration(self, model_idx: int = 0, load_best: bool = False) -> str:
        self.n_obs_per_measure = 4  # time and two measurements + event type (measurement = 0, dosing = 1)
        self.n_epochs = 500
        self.summary_dim = self.n_params * 2

        # load best
        if load_best:
            model_idx = 5
            # amortizer-pharma-split-sequence-summary-Bi-LSTM-8layers-2coupling-spline-750epochs -> 5

        summary_network_type = ['sequence', 'split-sequence']
        bidirectional_LSTM = [True, False]
        n_coupling_layers = [7, 8]
        n_dense_layers_in_coupling = [2, 3]
        coupling_design = ['spline']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, coupling_design, summary_network_type))

        if model_idx >= len(combinations) or model_idx < 0:
            model_name = f'amortizer-pharma' \
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

        model_name = f'amortizer-pharma' \
                     f'-{self.summary_network_type}-summary' \
                     f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def load_data(self,
                  n_data: Optional[int] = None,
                  synthetic: bool = False,
                  return_synthetic_params: bool = False) -> Union[list[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        if not synthetic:
            data = load_data(number_data_points=n_data)
        else:
            assert n_data is not None, 'n_data must be given for synthetic data'
            params = batch_gaussian_prior(mean=self.prior_mean,
                                          cov=self.prior_cov / 10,
                                          batch_size=n_data) - 1
            params[:, [0, 3, 5, 6, 7, 8, 9]] = self.prior_mean[[0, 3, 5, 6, 7, 8, 9]] - 1
            params[:, -1] += 1  # eta_4 is centered around 0
            data = batch_simulator(params)
            if return_synthetic_params:
                return data, params
        return data

    def plot_example(self, params: Optional[np.ndarray] = None) -> None:
        """Plots an individual trajectory of an individual in this model."""
        if params is None:
            params = self.prior(1)['prior_draws'][0]

        output = batch_simulator(params)
        ax = self.prepare_plotting(output, params)

        plt.title(f'Patient Simulation')
        plt.legend()
        plt.show()
        return

    @staticmethod
    def prepare_plotting(data: np.ndarray, params: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
        # convert BayesFlow format to observables
        y, t_measurement, doses_time_points, dos, wt = convert_bf_to_observables(data)
        t_measurement_full = np.linspace(0, t_measurement[-1] + 100, 100)

        # simulate data
        sim_data = batch_simulator(param_batch=params,
                                   t_measurement=t_measurement_full,
                                   t_doses=doses_time_points,
                                   wt=wt,
                                   dos=dos,
                                   with_noise=False,
                                   convert_to_bf_batch=False)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)

        if len(params.shape) == 1:  # so not (batch_size, params)
            # just a single parameter set
            # plot simulated data
            ax.plot(t_measurement_full, sim_data[:, 0], 'orange', label='simulated $A_{2}$')
            ax.plot(t_measurement_full, sim_data[:, 1], 'red', label='simulated $A_{3}$')
        else:
            y1 = sim_data[:, :, 0]
            y2 = sim_data[:, :, 1]
            # calculate median and quantiles
            y1_median = np.median(y1, axis=0)
            y2_median = np.median(y2, axis=0)

            y1_quantiles = np.percentile(y1, [2.5, 97.5], axis=0)
            y2_quantiles = np.percentile(y2, [2.5, 97.5], axis=0)

            # plot simulated data
            ax.fill_between(t_measurement_full, y1_quantiles[0], y1_quantiles[1],
                            alpha=0.2, color='orange')
            ax.plot(t_measurement_full, y1_median, 'orange', label='median $A_{2}$')

            ax.fill_between(t_measurement_full, y2_quantiles[0], y2_quantiles[1],
                            alpha=0.2, color='red')
            ax.plot(t_measurement_full, y2_median, 'red', label='median $A_{3}$')

        # plot observed data
        ax.scatter(t_measurement, y[:, 0], color='orange', label='observed $A_{2}$')
        ax.scatter(t_measurement, y[:, 1], color='red', label='observed $A_{3}$')

        # plot dosing events
        ax.vlines(doses_time_points, np.log(0.001), np.max(y[:, 0]),
                  color='grey', alpha=0.5, label='dosing events')
        return ax


def convert_csv_to_simulation_data(csv_data: pd.DataFrame) -> list:
    """
    Convert csv data to simulation data
    Args:
        csv_data:  pd.DataFrame

    Returns:
        data_list: list of tuples (patient_id, [measurement_data_A2, measurement_data_A3],
                                   measurement_data_time, DOS, wt)

    """
    data_list = []
    for patient_id in csv_data['ID'].unique():
        patient_data = csv_data[csv_data['ID'] == patient_id]

        wt = patient_data['WT'].unique()[0]
        dos = patient_data['DOS'].unique()[0]

        measurement_data = patient_data.loc[patient_data['EVID'] == 0, ['TIME', 'CMT', 'LNDV']]
        measurement_data_time = measurement_data.loc[measurement_data['CMT'] == 2, ['TIME']].to_numpy().flatten()
        measurement_data_A2 = measurement_data.loc[measurement_data['CMT'] == 2, ['LNDV']].to_numpy().flatten()
        measurement_data_A3 = measurement_data.loc[measurement_data['CMT'] == 3, ['LNDV']].to_numpy().flatten()
        if measurement_data_time.shape[0] == 0:
            continue

        measurements = np.stack((measurement_data_A2,
                                 measurement_data_A3), axis=1)

        dosage_data = patient_data.loc[patient_data['EVID'] == 1, ['TIME']].to_numpy().flatten()
        data_list.append([patient_id, measurements,
                          measurement_data_time, dosage_data, dos, wt])
    return data_list


def read_csv_pharma(csv_file: str) -> pd.DataFrame:
    data = pd.read_csv(csv_file)

    data = data[['ID', 'TRTW', 'TIME', 'DOS', 'DV', 'LNDV', 'EVID', 'CMT', 'AGE', 'SEX', 'WEIGHT']]
    data = data.where(data != ".")
    data = data.where(data != "LOQ")

    data = data.convert_dtypes()
    convert_dict = {'TIME': float,
                    'DOS': float,
                    'DV': float,
                    'LNDV': float
                    }
    data = data.astype(convert_dict)

    # create wt column
    data['WT'] = data['WEIGHT']
    data.loc[(data['SEX'] == 1) & (data['WEIGHT'] == -99), 'WT'] = 83
    data.loc[(data['SEX'] == 0) & (data['WEIGHT'] == -99), 'WT'] = 75
    return data


# load data from files
def load_data(number_data_points: Optional[int] = None) -> list[np.ndarray]:
    file_name = '../data/pharma/Suni_PK_final.csv'
    if not os.path.exists(file_name):
        print('data not found')
        return []
    data_raw = read_csv_pharma(file_name)
    data = convert_csv_to_simulation_data(data_raw)

    if number_data_points is not None:
        data = data[:number_data_points]

    # convert data to a BayesFlow format
    data_bayesflow = []
    for d in data:
        p_id, y_log, measurements, doses_time_points, dos, wt = d
        observables = convert_to_bf_format(y=y_log,
                                           t_measurements=measurements,
                                           doses_time_points=doses_time_points,
                                           dos=dos,
                                           wt=wt)
        data_bayesflow.append(observables)
    return data_bayesflow


def get_nonmem_data_helper(nonmem_a2: pd.DataFrame, nonmem_a3: pd.DataFrame, obs_data: list[np.ndarray],
                           patient_idx: int):
    nonmem_idx = pd.unique(nonmem_a2.index)[patient_idx]  # convert to index in nonmem
    # get data from nonmem for correct EVID (so IRES != IPRED)
    patient_data_a2 = pd.DataFrame()
    if nonmem_idx in nonmem_a2.loc[nonmem_idx].index:
        patient_data_a2['IRES'] = nonmem_a2.loc[nonmem_idx, 'IRES']
        patient_data_a2['IWRES'] = nonmem_a2.loc[nonmem_idx, 'IWRES']
        patient_data_a2['IPRED'] = nonmem_a2.loc[nonmem_idx, 'IPRED']
        patient_data_a2['TIME'] = nonmem_a2.loc[nonmem_idx, 'TIME']
    else:
        # only one entry
        patient_data_a2['IRES'] = [nonmem_a2.loc[nonmem_idx, 'IRES']]
        patient_data_a2['IWRES'] = [nonmem_a2.loc[nonmem_idx, 'IWRES']]
        patient_data_a2['IPRED'] = [nonmem_a2.loc[nonmem_idx, 'IPRED']]
        patient_data_a2['TIME'] = [nonmem_a2.loc[nonmem_idx, 'TIME']]

    patient_data_a3 = pd.DataFrame()
    if nonmem_idx in nonmem_a3.loc[nonmem_idx].index:
        patient_data_a3['IRES'] = nonmem_a3.loc[nonmem_idx, 'IRES']
        patient_data_a3['IWRES'] = nonmem_a3.loc[nonmem_idx, 'IWRES']
        patient_data_a3['IPRED'] = nonmem_a3.loc[nonmem_idx, 'IPRED']
        patient_data_a3['TIME'] = nonmem_a3.loc[nonmem_idx, 'TIME']
    else:
        # only one entry
        patient_data_a3['IRES'] = [nonmem_a3.loc[nonmem_idx, 'IRES']]
        patient_data_a3['IWRES'] = [nonmem_a3.loc[nonmem_idx, 'IWRES']]
        patient_data_a3['IPRED'] = [nonmem_a3.loc[nonmem_idx, 'IPRED']]
        patient_data_a3['TIME'] = [nonmem_a3.loc[nonmem_idx, 'TIME']]

    # clean data
    y, y_time_points, doses_time_points, DOS, wt = convert_bf_to_observables(obs_data[patient_idx])

    # find all values in patient_data_a2 close to times in obs_data
    dist = np.abs(y_time_points[:, np.newaxis] - patient_data_a2['TIME'].values)
    nonmem_index_2 = np.unique(dist.argmin(axis=1))

    dist_obs_2 = np.abs(patient_data_a2['TIME'].values[nonmem_index_2, np.newaxis] - y_time_points)
    obs_index = np.unique(dist_obs_2.argmin(axis=1))
    # y contains now only values close to times in patient_data_a2
    y = y[obs_index]
    y_time_points = y_time_points[obs_index]

    # find all values in patient_data_a3  close to times in obs_data
    dist = np.abs(y_time_points[:, np.newaxis] - patient_data_a3['TIME'].values)
    nonmem_index_3 = np.unique(dist.argmin(axis=1))
    dist_obs_3 = np.abs(patient_data_a3['TIME'].values[nonmem_index_3, np.newaxis] - y_time_points)
    obs_index_2 = np.unique(dist_obs_3.argmin(axis=1))
    # y contains now only values close to times in patient_data_a2 and patient_data_a3
    y = y[obs_index_2]
    y_time_points = y_time_points[obs_index_2]

    # now remove times from patient_data_a2 which are not in patient_data_a3
    dist = np.abs(y_time_points[:, np.newaxis] - patient_data_a2['TIME'].values)
    nonmem_index_2 = np.unique(dist.argmin(axis=1))

    # save observed data
    obs_a2 = list(y[:, 0])
    obs_a3 = list(y[:, 1])
    measurement_time = list(y_time_points)
    pred_a2_nonmem = list(patient_data_a2['IPRED'].values[nonmem_index_2])
    pred_a3_nonmem = list(patient_data_a3['IPRED'].values[nonmem_index_3])

    return [obs_a2, pred_a2_nonmem], [obs_a3, pred_a3_nonmem], measurement_time


result_file = '../output/results_nonmem/sunitinib_final_init1.SDTABFboot_029.csv'
if os.path.exists(result_file):
    obs_data = load_data()
    results_nonmen = pd.read_csv(result_file,
                                 index_col=0, header=1)
    nonmem_a2 = results_nonmen.loc[results_nonmen['CMT'] == 2, ['TIME', 'IPRED', 'IRES', 'IWRES']]
    nonmem_a3 = results_nonmen.loc[results_nonmen['CMT'] == 3, ['TIME', 'IPRED', 'IRES', 'IWRES']]
    get_nonmem_patient_data = partial(get_nonmem_data_helper, nonmem_a2=nonmem_a2, nonmem_a3=nonmem_a3,
                                      obs_data=obs_data)


def nonmem_best_results(full_param_names):
    raw_data = pd.read_csv(f'../output/results_nonmem/retries_sunitinib_lognor.csv', delimiter=',',
                           index_col=0, header=0)
    # remove uninformative columns and add missing columns
    raw_data = raw_data[
        raw_data.columns[[0, 1, 3, 4, 5, 6, 7, 9, 11, 12] + [13, 15, 18, 22] + [14, 16, 17, 19, 20, 21] + [25, 39, 40]]]
    raw_data.sort_values(by=['ofv'], inplace=True)

    # remove uninformative columns and add missing columns
    corr_names_nonmem = ['eta_1_eta_0', 'eta_2_eta_0', 'eta_2_eta_1', 'eta_3_eta_0', 'eta_3_eta_1', 'eta_3_eta_2']
    raw_data.columns = (full_param_names[:10] + ['var-$\\theta_2-\eta_1$',
                                                 'var-$\\theta_6-\eta_2$',
                                                 'var-$\\theta_4-\eta_3$',
                                                 'var-$\eta_4$']
                        + corr_names_nonmem + list(raw_data.columns[-3:]))
    # raw_data.columns = full_param_names[:10] +
    # full_param_names[25:29] + corr_names_nonmem + list(raw_data.columns[-3:])
    # log transform population parameters
    raw_data[full_param_names[:10]] = raw_data[full_param_names[:10]].abs().apply(np.log)
    # add variances
    # raw_data[full_param_names[10:25]] = np.zeros(15)
    # raw_data[full_param_names[29]] = 0
    raw_data[['var-$\\theta_1$', 'var-$\\theta_5$',
              'var-$\\theta_7$',
              'var-$\\theta_8$',
              'var-$\\theta_{10}$',
              'var-$\\theta_{12}$',
              'var-$\\theta_{13}$']] = np.zeros(7)
    raw_data['pop-$\eta_4$'] = 0
    # get results to compare
    results_to_compare = raw_data[full_param_names[:30] + corr_names_nonmem].values

    # get best results
    pop_mean_nonmem = results_to_compare[0, :11]
    pop_cov_nonmem = np.diag(results_to_compare[0, 11:-6])
    pop_cov_nonmem[pop_cov_nonmem == 0] = 0.008
    # todo: load corr
    # raw_data[['eta_1_eta_0', 'eta_2_eta_1', 'eta_3_eta_2']].values[0]

    return pop_mean_nonmem, pop_cov_nonmem


regimes_m = [[222.5833333, 387.1666667, 555.5, 699.2833333, 890.75, 1060.016667, 1203.35],
             [198.0],
             [30.0, 51.0, 54.0],
             [30.0, 75.58333333, 78.75, 174.0, 198.0, 222.0],
             [32.0, 51.0, 54.5, 77.91666667, 100.0, 387.0, 391.0, 723.1666667, 726.1666667, 1059.0, 1062.0,
              1514.833333, 1517.833333],
             [172.75, 340.1666667, 508.6666667, 844.9, 1012.866667, 1180.666667, 1348.5, 1517.0, 1684.9],
             [821.5, 1157.5, 1542.0, 1829.0, 2165.5, 2837.75, 3677.0, 4252.0, 4684.5],
             [341.75, 461.1666667, 798.0, 1181.0, 1805.0, 2651.0, 3315.0, 3652.166667],
             [268.5],
             [174.8333333, 342.0, 508.1666667],
             [7.75, 173.5, 341.5, 509.5],
             [343.0],
             [198.0, 388.5, 531.8333333, 698.5, 915.1666667, 1061.666667, 1227.583333, 1371.25, 1708.666667, 2379.75,
              2883.5],
             [221.5, 389.5833333, 556.5833333, 748.5, 892.5, 1084.5, 1227.333333, 2380.5, 2404.416667, 2908.25],
             [218.75, 388.8333333, 555.0, 749.0, 892.0, 1059.333333, 1228.5, 1564.5, 1899.416667, 2235.666667,
              2571.5],
             [171.4166667, 387.1666667, 555.3333333, 704.25, 891.25, 1228.083333, 1660.0, 1898.75, 2571.5],
             [220.25, 389.5, 533.25, 679.5, 1038.0, 1201.0, 1708.75, 2070.166667, 2238.666667, 2335.0, 2573.75],
             [200.1666667, 368.4166667, 557.0833333, 698.6666667, 872.25, 1040.333333, 1181.333333, 1520.583333,
              1664.0, 2309.166667, 2649.416667],
             [200.3, 344.0, 511.5, 680.3333333, 845.25, 1016.5, 1180.5, 1352.166667, 1520.166667, 2024.583333,
              2380.666667],
             [99.25, 268.1666667, 434.0833333, 652.1666667, 772.25, 940.25, 1108.0, 1276.333333, 1612.166667,
              2284.833333, 2836.5],
             [77.16666667, 245.0833333, 413.1666667, 557.25, 725.25, 893.0, 1061.416667, 1229.5, 1565.25, 2070.25,
              2742.0],
             [77.58333333, 244.8333333, 436.8333333, 604.8333333, 772.5833333, 940.8333333, 1109.0, 1445.916667,
              1757.666667, 2117.5, 2477.666667],
             [148.25, 219.5, 363.6666667, 483.75, 674.75, 1059.5, 1155.75, 1227.5, 1683.583333, 2019.583333,
              2187.083333],
             [32.08333333, 49.0, 73.0, 97.5, 169.6666667, 362.5, 530.5],
             [104.6666667, 146.1666667, 267.9166667, 315.8333333, 435.3333333, 1302.0],
             [30.0, 102.0, 150.0, 174.0, 198.0],
             [175.3333333, 340.3333333, 508.25, 675.8333333, 843.8333333, 1036.5, 1180.5, 1348.5, 2022.25],
             [123.75, 291.5, 460.3333333, 602.5833333],
             [198.3333333, 363.5833333, 436.25, 508.25, 698.75, 702.3333333, 771.0833333, 1010.5, 1180.833333,
              1706.666667, 2474.333333, 2882.666667],
             [101.25, 220.3333333],
             [291.75, 294.75, 363.1666667, 627.4166667, 723.75, 964.1666667, 1299.916667, 1303.083333, 1804.0,
              1875.916667, 2092.416667, 2213.666667],
             [51.83333333, 243.0],
             [347.1666667, 680.5, 1016.0, 2404.0, 3079.0, 3583.0],
             [368.5, 1666.0],
             [675.25, 1018.833333, 1683.833333, 2355.5],
             [173.0833333, 339.5, 507.0833333, 723.4166667, 1564.75, 2859.0],
             [172.0, 338.6666667],
             [315.0, 674.3333333, 967.3333333, 1539.0, 1853.25, 2164.666667, 2379.0],
             [0.0, 271.5833333, 436.5],
             [362.0, 530.0, 698.0, 1058.0],
             [411.5, 699.5, 1088.916667, 1588.083333, 2092.333333, 2595.5],
             [339.3333333, 508.0],
             [339.8333333, 699.5],
             [24.8, 338.0, 672.2, 1177.25, 1538.0, 1874.2, 2186.0],
             [24.7, 336.75, 673.0, 961.5, 1370.25, 1706.0, 2044.0],
             [24.75, 336.5, 673.75, 1080.0, 1610.0, 1946.3, 2378.5],
             [25.42, 341.33, 672.5, 1008.67, 1514.92, 1850.17, 2210.17],
             [24.67, 336.67, 672.75, 1008.83],
             [24.33, 312.0, 672.56, 1180.0, 1513.0, 2018.5, 2353.0],
             [25.33, 336.0, 673.33, 984.08, 1176.42, 1512.33, 1824.3, 1994.2, 2352.58],
             [26.0, 336.33, 696.0, 840.0, 1176.92, 1512.7, 2352.83],
             [26.6, 312.0, 359.7],
             [27.6, 337.37],
             [27.25, 47.83, 359.67],
             [336.0, 720.33, 839.83, 1176.33, 1514.58],
             [360.5, 696.17, 1008.5],
             [23.83, 336.17, 1320.0],
             [312.16, 624.17, 960.17, 1296.67, 1632.0, 2136.67, 2304.67],
             [29.17, 337.33, 672.25, 1009.17, 1345.17, 1681.0, 2017.5],
             [25.33, 336.5, 674.83, 1009.08, 1344.42, 1683.25, 2016.25],
             [24.25, 337.0, 672.83, 840.58, 1177.92, 1345.33, 1681.33, 2041.28, 2352.67],
             [26.17, 361.5, 673.5, 1033.25, 1345.33, 1681.5],
             [26.0, 336.75, 673.5, 1008.67, 1177.5],
             [27.75, 336.25, 672.25, 1008.33, 1344.5, 1680.67, 2016.67]]

regimes_d = [[26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 674.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0],
             [26.0, 50.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0],
             [28.0, 52.0, 76.0, 100.0, 124.0, 148.0, 172.0, 196.0, 220.0, 244.0, 268.0, 292.0, 316.0, 340.0, 364.0,
              388.0, 412.0, 436.0, 460.0, 484.0, 508.0, 532.0, 556.0, 580.0, 604.0,
              628.0, 652.0, 676.0, 1036.0, 1060.0, 1084.0, 1108.0, 1132.0, 1156.0, 1180.0, 1204.0, 1228.0, 1252.0,
              1276.0, 1300.0, 1324.0, 1348.0, 1372.0, 1396.0, 1420.0, 1444.0, 1468.0, 1492.0, 1516.0],
             [38.0, 62.0, 86.0, 110.0, 134.0, 158.0, 182.0, 206.0, 230.0, 254.0, 278.0, 302.0, 326.0, 350.0, 374.0,
              398.0, 422.0, 446.0, 470.0, 494.0, 518.0, 542.0, 566.0, 590.0, 614.0, 638.0, 662.0, 686.0, 1046.0,
              1070.0, 1094.0, 1118.0, 1142.0, 1166.0, 1190.0, 1214.0, 1238.0, 1262.0, 1286.0, 1310.0, 1334.0, 1358.0,
              1382.0, 1406.0, 1430.0, 1454.0, 1478.0, 1502.0, 1526.0, 1550.0, 1574.0, 1598.0, 1622.0, 1646.0, 1670.0],
             [29.5, 53.5, 77.5, 101.5, 125.5, 149.5, 173.5, 197.5, 221.5, 245.5, 269.5, 293.5, 317.5, 341.5, 365.5,
              389.5, 413.5, 437.5, 461.5, 485.5, 509.5, 533.5, 557.5, 581.5, 605.5, 629.5, 653.5, 677.5, 701.5, 725.5,
              749.5, 773.5, 797.5, 821.5, 845.5, 869.5, 893.5, 917.5, 941.5, 965.5, 989.5, 1013.5, 1037.5, 1061.5,
              1085.5, 1109.5, 1133.5, 1157.5, 1181.0, 1205.0, 1229.0, 1253.0, 1277.0, 1302.0, 1685.0, 1709.0,
              1733.0, 1757.0, 1781.0, 1805.0, 1829.0, 1853.0, 1877.0, 1901.0, 1925.0, 1949.0, 1973.0, 1997.0,
              2021.0, 2045.0, 2069.0, 2093.0, 2117.0, 2141.0, 2165.5, 2189.0, 2213.0, 2237.0, 2261.0, 2285.0,
              2309.0, 2333.0, 2357.0, 2381.0, 2405.0, 2429.0, 2453.0, 2477.0, 2501.0, 2525.0, 2549.0, 2573.0,
              2597.0, 2621.0, 2645.0, 2669.0, 2693.0, 2717.0, 2741.0, 2765.0, 2789.0, 2813.0, 2837.75, 2861.0,
              2885.0, 2909.0, 2933.0, 2957.0, 2981.0, 3005.0, 3029.0, 3053.0, 3077.0, 3101.0, 3125.0, 3149.0,
              3173.0, 3197.0, 3221.0, 3245.0, 3269.0, 3293.0, 3317.0, 3341.0, 3365.0, 3389.0, 3413.0, 3437.0,
              3461.0, 3485.0, 3509.0, 3533.0, 3557.0, 3581.0, 3605.0, 3629.0, 3653.0, 3677.0, 3701.0, 3725.0,
              3749.0, 3773.0, 3797.0, 3821.0, 3845.0, 3869.0, 3893.0, 3917.0, 3941.0, 3965.0, 3989.0, 4013.0,
              4037.0, 4061.0, 4085.0, 4109.0, 4133.0, 4157.0, 4181.0, 4205.0, 4229.0, 4253.0, 4277.0, 4301.0,
              4325.0, 4349.0, 4373.0, 4397.0, 4421.0, 4445.0, 4469.0, 4493.0, 4517.0, 4541.0, 4565.0, 4589.0,
              4613.0, 4637.0, 4663.5],
             [29.0, 53.0, 77.0, 101.0, 125.0, 149.0, 173.0, 197.0, 221.0, 245.0, 269.0, 293.0, 317.0, 341.0, 365.0,
              389.0, 413.0, 437.0, 460.75, 484.75, 508.75, 532.75, 556.75, 580.75, 604.75, 628.75, 652.75, 676.75,
              700.75, 724.75, 748.75, 772.75, 797.75, 821.75, 845.75, 869.75, 893.75, 917.75, 941.75, 965.75, 989.75,
              1013.75, 1037.75, 1061.75, 1085.75, 1109.75, 1133.75, 1163.5, 1187.5, 1211.5, 1235.5, 1259.5, 1283.5,
              1307.5, 1331.5, 1355.5, 1379.5, 1403.5, 1427.5, 1451.5, 1475.5, 1499.5, 1523.5, 1547.5, 1571.5, 1595.5,
              1619.5, 1643.5, 1667.5, 1691.5, 1715.5, 1739.5, 1763.5, 1793.75, 1817.75, 1841.75, 1865.75, 1889.75,
              1913.75, 1937.75, 1961.75, 1985.75, 2009.75, 2033.75, 2057.75, 2081.75, 2105.75, 2129.75, 2153.75,
              2177.75, 2201.75, 2225.75, 2249.75, 2273.75, 2297.75, 2321.75, 2345.75, 2369.75, 2393.75, 2417.75,
              2441.75, 2465.75, 2489.75, 2513.75, 2537.75, 2561.75, 2585.75, 2609.75, 2633.75, 2634.0, 2658.0, 2682.0,
              2706.0, 2730.0, 2754.0, 2778.0, 2802.0, 2826.0, 2850.0, 2874.0, 2898.0, 2922.0, 2946.0, 2970.0, 2994.0,
              3018.0, 3042.0, 3066.0, 3090.0, 3114.0, 3138.0, 3162.0, 3186.0, 3210.0, 3234.0, 3258.0, 3282.0, 3293.0,
              3317.0, 3341.0, 3365.0, 3389.0, 3413.0, 3437.0, 3461.0, 3485.0, 3509.0, 3533.0, 3557.0, 3581.0, 3605.0,
              3631.833333],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0],
             [18.83333333, 42.83333333, 66.83333333, 90.83333333, 114.8333333, 162.8333333, 186.8333333,
              210.8333333, 234.8333333, 258.8333333, 282.8333333, 307.0, 331.0, 355.0, 379.0, 403.0, 427.0,
              451.0, 476.1666667],
             [41.5, 65.5, 89.5, 113.5, 137.5, 161.5, 185.5, 209.5, 233.5, 257.5, 281.5, 305.5, 329.5, 353.5, 377.5,
              401.5, 425.5, 449.5, 473.5, 497.5],
             [19.0, 43.0, 67.0, 91.0, 115.0, 139.0, 163.0, 187.0, 211.0, 235.0, 259.0, 283.0, 307.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0,
              338.0, 362.0, 386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0,
              674.0, 1034.0, 1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0, 1298.0,
              1322.0, 1346.0, 1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0, 1610.0,
              1634.0, 1658.0, 1682.0, 2042.0, 2066.0, 2090.0, 2114.0, 2138.0, 2162.0, 2186.0, 2210.0, 2234.0, 2258.0,
              2282.0, 2306.0, 2330.0, 2354.0, 2378.0, 2402.0, 2426.0, 2450.0, 2474.0, 2498.0, 2522.0, 2546.0, 2570.0,
              2594.0, 2618.0, 2642.0, 2666.0],
             [0.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 674.0, 1034.0,
              1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0, 1298.0, 1322.0,
              1346.0, 1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0, 1610.0,
              1634.0, 1658.0, 1682.0, 2042.0, 2066.0, 2090.0, 2114.0, 2138.0, 2162.0, 2186.0, 2210.0, 2234.0,
              2258.0, 2282.0, 2306.0, 2330.0, 2354.0, 2378.0, 2402.0, 2426.0, 2450.0, 2474.0, 2498.0, 2522.0,
              2546.0, 2570.0, 2594.0, 2618.0, 2642.0, 2666.0, 2690.0],
             [14.0, 38.0, 62.0, 86.0, 110.0, 134.0, 158.0, 182.0, 206.0, 230.0, 254.0, 278.0, 302.0, 326.0, 350.0,
              374.0, 398.0, 422.0, 446.0, 470.0, 494.0, 518.0, 542.0, 566.0, 590.0, 614.0, 638.0, 662.0, 686.0,
              710.0, 734.0, 758.0, 782.0, 806.0, 830.0, 854.0, 878.0, 902.0, 926.0, 950.0, 974.0, 998.0, 1022.0,
              1046.0, 1070.0, 1094.0, 1118.0, 1142.0, 1166.0, 1478.0, 1502.0, 1526.0, 1550.0, 1574.0, 1586.0,
              1610.0, 1634.0, 1658.0, 1682.0, 1706.0, 1730.0, 1754.0, 1778.0, 1802.0, 1826.0, 1850.0, 1874.0,
              1898.0, 1922.0, 1946.0, 1970.0, 1994.0, 2018.0, 2042.0, 2066.0, 2090.0, 2114.0, 2138.0, 2162.0,
              2186.0, 2210.0, 2234.0, 2258.0, 2282.0, 2306.0, 2330.0, 2354.0, 2378.0, 2402.0, 2426.0, 2450.0,
              2474.0, 2498.0, 2522.0, 2546.0, 2570.0],
             [37.5, 61.5, 85.5, 109.5, 133.5, 156.0, 180.0, 204.0, 228.0, 252.0, 276.0, 300.0, 324.0, 348.0, 372.0,
              397.5, 421.5, 445.5, 469.5, 493.5, 517.5, 541.5, 565.5, 589.5, 613.5, 637.5, 661.5, 685.5, 1045.5,
              1069.5, 1093.5, 1117.5, 1141.5, 1165.5, 1189.5, 1213.75, 1237.5, 1261.5, 1285.5, 1309.5, 1333.5,
              1357.5, 1381.5, 1405.5, 1429.5, 1453.5, 1477.5, 1501.5, 1525.5, 1549.5, 1573.5, 1597.5, 1621.5,
              1645.5, 1669.5, 1693.5, 2053.5, 2077.5, 2101.5, 2125.5, 2149.5, 2173.5, 2197.5, 2221.5, 2245.5,
              2269.5, 2293.5, 2317.5, 2341.5, 2365.5, 2389.5, 2413.5, 2437.5, 2461.5, 2485.5, 2509.5, 2533.5, 2558.5],
             [15.5, 39.5, 63.5, 87.5, 111.5, 135.5, 159.5, 183.5, 207.5, 231.5, 255.5, 279.5, 303.5, 327.5, 351.5,
              375.5, 399.5, 423.5, 447.5, 471.5, 495.5, 519.5, 543.5, 567.5, 591.5, 615.5, 639.5, 663.5, 687.5,
              711.5, 735.5, 759.5, 783.5, 807.5, 831.5, 855.5, 879.5, 903.5, 927.5, 951.5, 975.5, 999.5, 1023.5,
              1047.5, 1071.5, 1095.5, 1119.5, 1143.5, 1167.5, 1191.5, 1215.5, 1239.5, 1263.5, 1287.5, 1311.5,
              1335.5, 1359.5, 1383.5, 1407.5, 1431.5, 1455.5, 1527.5, 1551.5, 1575.5, 1599.5, 1623.5, 1647.5,
              1671.5, 1695.5, 1719.5, 1743.5, 1767.5, 1791.5, 1815.5, 1839.5, 1863.5, 1887.5, 1911.5, 1935.5,
              1959.5, 1983.5, 2079.5, 2103.5, 2127.5, 2151.5, 2175.5, 2199.5, 2223.5, 2247.5, 2271.5, 2295.5,
              2319.5, 2343.5, 2367.5, 2391.5, 2415.5, 2439.5, 2463.5, 2487.5, 2511.5, 2535.5, 2559.5],
             [38.0, 62.0, 86.0, 110.0, 134.0, 158.0, 181.5, 206.0, 230.0, 254.0, 278.0, 302.0, 326.0, 350.0, 374.0,
              398.0, 422.0, 446.0, 470.0, 494.0, 518.0, 542.0, 566.0, 590.0, 614.0, 638.0, 662.0, 687.5, 1046.0,
              1070.0, 1094.0, 1118.0, 1142.0, 1166.0, 1190.0, 1214.0, 1238.0, 1262.0, 1286.0, 1310.0, 1334.0,
              1358.0, 1382.0, 1406.0, 1430.0, 1454.0, 1478.0, 1503.5, 1526.0, 1550.0, 1574.0, 1598.0, 1622.0,
              1646.0, 1670.0, 1694.0],
             [7.0, 31.0, 55.0, 79.0, 103.0, 127.0, 151.0, 175.0, 199.0, 223.0, 247.0, 271.0, 295.0, 319.0, 343.0,
              369.0, 393.0, 417.0, 441.0, 465.0, 489.0, 493.5, 513.0, 537.0, 561.0, 585.0, 609.0, 633.0, 657.0,
              681.0, 705.0, 729.0, 753.0, 777.0, 801.0, 825.0, 849.0, 873.0, 897.0, 921.0, 945.0, 969.0, 993.0,
              1017.0, 1041.0, 1065.0, 1089.0, 1113.0, 1137.0, 1161.0, 1185.0, 1209.0, 1233.0, 1257.0, 1281.0,
              1305.0, 1329.0, 1353.0, 1377.0, 1401.0, 1425.0, 1449.0, 1473.0, 1497.0, 1521.0, 1545.0, 1569.0,
              1593.0, 1617.0, 1641.0, 1665.0, 1689.0, 1713.0, 1737.0, 1761.0, 1785.0, 1809.0, 1833.0, 1857.0,
              1881.0, 1905.0, 1929.0, 1953.0, 1977.0, 2001.0, 2025.0, 2049.0, 2073.0, 2097.0, 2121.0, 2145.0,
              2169.0, 2193.0, 2217.0, 2241.0, 2265.0, 2289.0, 2313.0, 2337.0, 2361.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 433.0833333, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 674.0, 1034.0,
              1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0, 1298.0, 1322.0, 1346.0,
              1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0, 1610.0, 1634.0, 1658.0,
              1682.0, 2042.0, 2066.0, 2090.0, 2114.0, 2138.0, 2162.0, 2186.0, 2210.0, 2234.0, 2258.0, 2282.0, 2306.0,
              2330.0, 2354.0, 2378.0, 2402.0, 2426.0, 2450.0, 2474.0, 2498.0, 2522.0, 2546.0, 2570.0, 2594.0, 2618.0,
              2642.0, 2666.0, 2690.0],
             [27.0, 51.0, 75.0, 99.0, 123.0, 147.0, 171.0, 195.0, 219.0, 243.0, 267.0, 291.0, 315.0, 339.0, 363.0,
              387.0, 411.0, 435.0, 459.0, 483.0, 507.0, 531.0, 555.0, 579.0, 603.0, 627.0, 651.0, 675.0, 699.0,
              723.0, 1083.0, 1107.0, 1131.0, 1155.0, 1179.0, 1203.0, 1227.0, 1251.0, 1275.0, 1299.0, 1323.0,
              1347.0, 1371.0, 1395.0, 1419.0, 1443.0, 1467.0, 1491.0, 1515.0, 1539.0, 1563.0, 1587.0, 1611.0,
              1635.0, 1659.0, 1683.0, 1707.0, 1731.0, 1755.0, 1779.0, 2091.0, 2115.0, 2139.0, 2163.0, 2187.0,
              2211.0, 2235.0, 2259.0, 2283.0, 2307.0, 2331.0, 2355.0, 2379.0, 2403.0, 2427.0, 2451.0, 2475.0,
              2499.0, 2523.0, 2547.0, 2571.0, 2595.0, 2619.0, 2643.0, 2667.0, 2691.0, 2715.0, 2739.0],
             [50.5, 75.5, 98.5, 122.5, 146.5, 170.5, 194.5, 218.5, 242.5, 266.5, 290.5, 314.5, 338.5, 362.5, 386.5,
              410.5, 434.5, 458.5, 482.5, 506.5, 530.5, 554.5, 578.5, 602.5, 626.5, 650.5, 674.5, 698.5, 722.5, 746.5,
              770.5, 794.5, 818.5, 842.5, 866.5, 890.5, 914.5, 938.5, 962.5, 986.5, 1010.5, 1034.5, 1058.5, 1082.5,
              1106.5, 1130.5, 1154.5, 1178.5, 1202.5, 1226.5, 1250.5, 1274.5, 1298.5, 1322.5, 1346.5, 1370.5, 1394.5,
              1418.5, 1442.5, 1466.5, 1490.5, 1514.5, 1538.5, 1562.5, 1586.5, 1610.5, 1634.5, 1658.5, 1682.5, 1706.5,
              1730.5, 1754.5, 1778.5, 1802.5, 1826.5, 1850.5, 1874.5, 1898.5, 1922.5, 1946.5, 1970.5, 1994.5, 2018.5,
              2042.5, 2066.5, 2090.5, 2114.5, 2138.5, 2162.5, 2186.5, 2210.5, 2234.5, 2258.5, 2282.5, 2306.5, 2330.5,
              2354.5, 2378.5, 2402.5, 2426.5, 2450.5, 2474.5],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 674.0, 1034.0,
              1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0, 1298.0, 1322.0, 1346.0,
              1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0, 1610.0, 1634.0, 1658.0,
              1682.0, 1994.0, 2018.0, 2042.0, 2066.0, 2090.0, 2114.0, 2138.0, 2162.0, 2186.0],
             [30.0, 54.0, 78.0, 102.0, 126.0, 150.0, 174.0, 198.0, 222.0, 246.0, 270.0, 294.0, 318.0, 342.0, 366.0,
              390.0, 414.0, 438.0, 462.0, 486.0, 510.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 410.0, 434.0,
              458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 674.0, 1010.0, 1034.0, 1058.0, 1082.0,
              1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 172.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 505.25, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 673.25, 1033.75,
              1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1177.666667, 1202.0, 1226.0, 1250.0, 1274.0, 1298.0, 1322.0,
              1346.5, 1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0, 1610.0, 1634.0,
              1658.0, 1682.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 288.25, 312.25, 336.25, 360.25,
              384.25, 408.25, 432.25, 456.25, 480.25],
             [26.25, 50.25, 74.25, 98.25, 122.25, 146.25, 170.25, 195.0, 218.25, 242.25, 266.25, 290.25, 314.25,
              338.25, 362.25, 386.25, 410.25, 434.25, 458.25, 482.25, 506.25, 530.25, 554.25, 578.25, 602.25, 626.25,
              650.25, 674.25, 1682.25, 1705.5, 1730.25, 1754.25, 1778.25, 1802.25, 1826.25, 1850.25, 1874.25, 1898.25,
              1922.25, 1946.25, 1970.25, 1994.25, 2018.25, 2042.25, 2066.25, 2090.25, 2114.25, 2138.25, 2162.25,
              2186.25, 2498.25, 2522.25, 2546.25, 2570.25, 2594.25, 2618.25],
             [36.0, 60.0, 84.0, 108.0, 132.0, 156.0, 180.0, 201.8333333],
             [31.75, 55.75, 79.75, 103.75, 127.75, 151.75, 175.75, 199.75, 223.75, 247.75, 271.75, 295.75, 319.75,
              343.6666667, 367.75, 391.75, 415.75, 439.75, 463.75, 487.75, 511.75, 535.75, 559.75, 583.75,
              608.2833333, 631.75, 655.75, 680.75, 1040.75, 1064.75, 1088.75, 1112.75, 1136.75, 1160.75, 1184.75,
              1208.75, 1232.75, 1256.75, 1281.75, 1304.75, 1328.75, 1352.75, 1376.75, 1400.75, 1424.75, 1448.75,
              1472.75, 1496.75, 1520.75, 1544.75, 1568.75, 1592.75, 1616.75, 1640.75, 1664.75, 1689.5, 2049.5, 2072.5,
              2096.5, 2120.5, 2144.5, 2168.5, 2193.666667],
             [26.0, 50.75, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 227.5],
             [40.0, 64.0, 88.0, 112.0, 136.0, 160.0, 184.0, 208.0, 232.0, 256.0, 280.0, 304.0, 328.5, 352.0, 376.0,
              400.0, 424.0, 448.0, 472.0, 496.0, 520.0, 544.0, 568.0, 592.0, 616.0, 640.0, 664.0, 688.0, 712.0, 736.0,
              760.0, 784.0, 808.0, 832.0, 856.0, 880.0, 904.0, 928.0, 952.0, 976.0, 1000.0, 1024.0, 1048.0, 1072.0,
              1096.0, 1120.0, 1144.0, 1168.0, 1192.0, 1216.0, 1240.0, 1264.0, 1288.0, 1312.0, 1336.0, 1360.0, 1384.0,
              1408.0, 1432.0, 1456.0, 1480.0, 1504.0, 1528.0, 1552.0, 1576.0, 1600.0, 1624.0, 1648.0, 1672.0, 1696.0,
              1720.0, 1744.0, 1768.0, 1792.0, 1816.0, 1840.0, 1864.0, 1888.0, 1912.0, 1936.0, 1960.0, 1984.0, 2008.0,
              2032.0, 2056.0, 2080.0, 2104.0, 2128.0, 2152.0, 2176.0, 2200.0, 2224.0, 2248.0, 2272.0, 2296.0, 2320.0,
              2344.0, 2368.0, 2392.0, 2416.0, 2440.0, 2464.0, 2488.0, 2512.0, 2536.0, 2560.0, 2584.0, 2608.0, 2632.0,
              2656.0, 2680.0, 2704.0, 2728.0, 2752.0, 2776.0, 2800.0, 2824.0, 2848.0, 2872.0, 2896.0, 2920.0, 2944.0,
              2968.0, 2992.0, 3016.0, 3040.0, 3064.0, 3088.0, 3112.0, 3136.0, 3160.0, 3184.0, 3208.0, 3232.0, 3256.0,
              3280.0, 3304.0, 3328.0, 3352.0, 3376.0, 3400.0, 3424.0, 3448.0, 3472.0, 3496.0, 3520.0, 3544.0, 3568.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              1322.0, 1346.0, 1370.0, 1394.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 674.0, 698.0,
              722.0, 746.0, 770.0, 794.0, 818.0, 842.0, 866.0, 890.0, 914.0, 938.0, 962.0, 986.0, 1010.0, 1034.0,
              1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0, 1298.0, 1322.0, 1346.0,
              1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0, 1610.0, 1634.0, 1658.0,
              1682.0, 1706.0, 1730.0, 1754.0, 1778.0, 1802.0, 1826.0, 1850.0, 1874.0, 1898.0, 1922.0, 1946.0, 1970.0,
              1994.0, 2018.0, 2042.0, 2066.0, 2090.0, 2114.0, 2138.0, 2162.0, 2186.0, 2210.0, 2234.0, 2258.0, 2282.0,
              2306.0, 2330.0],
             [27.83333333, 51.83333333, 75.83333333, 99.83333333, 123.8333333, 147.8333333, 171.8333333, 195.8333333,
              219.8333333, 243.8333333, 267.8333333, 291.8333333, 315.8333333, 339.8333333, 363.8333333, 387.8333333,
              11.8333333, 435.8333333, 459.8333333, 483.8333333, 507.8333333, 531.8333333, 555.8333333, 579.8333333,
              603.8333333, 627.8333333, 651.8333333, 675.8333333, 699.8333333, 723.8333333, 747.8333333, 771.8333333,
              795.8333333, 819.8333333, 843.8333333, 867.8333333, 891.8333333, 915.8333333, 939.8333333, 963.8333333,
              987.8333333, 1011.833333, 1035.833333, 1059.833333, 1083.833333, 1107.833333, 1131.833333, 1155.833333,
              1179.833333, 1203.833333, 1227.833333, 1251.833333, 1275.833333, 1299.833333, 1323.833333, 1347.833333,
              1371.833333, 1395.833333, 1419.833333, 1443.833333, 1467.833333, 1491.833333, 1515.833333, 1539.833333,
              1563.833333, 1587.833333, 1611.833333, 1635.833333, 1659.833333, 1683.833333, 1707.833333, 1731.833333,
              1755.833333, 1779.833333, 1803.833333, 1827.833333, 1851.833333, 1875.833333, 1899.833333, 1923.833333,
              1947.833333, 1971.833333, 1995.833333, 2019.833333, 2043.833333, 2067.833333, 2091.833333, 2115.833333,
              2139.833333, 2163.833333, 2187.833333, 2211.833333, 2235.833333, 2259.833333, 2283.833333, 2307.833333,
              2331.833333, 2355.833333, 2379.833333, 2403.833333, 2427.833333, 2451.833333, 2475.833333, 2499.833333,
              2523.833333, 2547.833333, 2571.833333, 2595.833333, 2619.833333, 2643.833333, 2667.833333, 2691.833333,
              2715.833333, 2739.833333, 2763.833333, 2787.833333, 2811.833333, 2835.833333],
             [28.66666667, 52.66666667, 76.66666667, 100.6666667, 124.6666667, 148.6666667, 172.6666667, 196.6666667,
              220.6666667, 244.6666667, 268.6666667, 292.6666667],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0, 673.25, 962.0,
              986.0, 1010.0, 1034.0, 1058.0, 1082.0, 1106.0, 1130.0, 1154.0, 1178.0, 1202.0, 1226.0, 1250.0, 1274.0,
              1298.0, 1322.0, 1346.0, 1370.0, 1394.0, 1418.0, 1442.0, 1466.0, 1490.0, 1514.0, 1538.0, 1562.0, 1586.0,
              1610.0, 1850.0, 1874.0, 1898.0, 1922.0, 1946.0, 1970.0, 1994.0, 2018.0, 2042.0],
             [25.0, 49.0, 73.0, 97.0, 121.0, 145.0, 169.0, 193.0, 217.0, 241.0, 265.0, 289.0, 313.0, 337.0, 361.0,
              385.0, 409.0, 433.0],
             [28.0, 52.0, 76.0, 100.0, 124.0, 148.0, 172.0, 196.0, 220.0, 244.0, 268.0, 292.0, 316.0, 340.0, 364.0,
              388.0, 412.0, 436.0, 460.0, 484.0, 508.0, 532.0, 556.0, 580.0, 604.0, 628.0, 652.0, 676.0],
             [27.0, 51.0, 75.0, 99.0, 123.0, 147.0, 171.0, 195.0, 219.0, 243.0, 267.0, 291.0, 315.0, 339.0, 363.0,
              387.0, 411.0, 435.0, 459.0, 483.0, 507.0, 531.0, 555.0, 579.0, 603.0, 627.0, 651.0, 675.0, 723.0, 747.0,
              1107.0, 1131.0, 1155.0, 1179.0, 1203.0, 1227.0, 1251.0, 1275.0, 1299.0, 1323.0, 1347.0, 1371.0, 1395.0,
              1419.0, 1443.0, 1467.0, 1491.0, 1515.0, 1539.0, 1563.0, 1587.0, 1611.0, 1635.0, 1659.0, 1683.0, 1707.0,
              1731.0, 1755.0, 1779.0, 1803.0, 2139.0, 2163.0, 2187.0, 2211.0, 2235.0, 2259.0, 2283.0, 2307.0, 2331.0,
              2355.0, 2379.0, 2403.0, 2427.0, 2451.0, 2475.0, 2499.0, 2523.0, 2547.0, 2571.0, 2595.0],
             [27.0, 51.0, 75.0, 99.0, 123.0, 147.0, 171.0, 195.0, 219.0, 243.0, 267.0, 291.0, 315.0, 339.0, 363.0],
             [26.0, 50.0, 74.0, 98.0, 122.0, 146.0, 170.0, 194.0, 218.0, 242.0, 266.0, 290.0, 314.0, 338.0, 362.0,
              386.0, 410.0, 434.0, 458.0, 482.0, 506.0, 530.0, 554.0, 578.0, 602.0, 626.0, 650.0],
             [0.42, 25.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 338.17,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1200.0,
              224.0, 1248.0, 1272.0, 1296.0, 1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0,
              1538.17, 1560.0, 1584.0, 1608.0, 1632.0, 1656.0, 1680.0, 1704.0, 1728.0, 1752.0, 1776.0, 1800.0,
              1824.0, 1848.0],
             [0.75, 24.8, 49.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.92,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 961.67,
              984.0, 1008.0, 1032.0, 1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.0, 1200.0, 1224.0, 1248.0, 1272.0,
              1296.0, 1320.0, 1344.0, 1370.42, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0, 1584.0,
              1608.0, 1632.0],
             [0.17, 24.92, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.67,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 673.92,
              1272.0, 1296.0, 1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0,
              1584.0, 1610.17, 1632.0, 1656.0, 1680.0, 1704.0, 1728.0, 1752.0, 1776.0, 1800.0, 1824.0, 1848.0, 1872.0,
              1896.0, 1920.0],
             [3.33, 25.58, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 341.5,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1200.0,
              1224.0, 1248.0, 1272.0, 1296.0, 1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1515.08,
              1536.0, 1560.0, 1584.0, 1608.0, 1632.0, 1656.0, 1680.0, 1704.0, 1728.0, 1752.0, 1776.0, 1800.0, 1824.0],
             [2.42, 24.83, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.83,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0],
             [4.42, 24.5, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.17, 336.0,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1180.17,
              1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0,
              1513.17, 1536.0, 1560.0, 1584.0, 1608.0, 1632.0, 1656.0, 1680.0, 1704.0, 1728.0, 1752.0, 2353.17],
             [0.17, 25.5, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.17,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1176.58,
              1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0,
              1512.5, 1536.0, 1560.0, 1584.0, 1608.0, 1632.0, 1656.0, 1680.0, 1704.0, 1728.0, 1752.0, 1776.0, 1800.0,
              1824.47, 2352.75],
             [0.85, 26.17, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.5,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1177.08,
              1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0,
              1512.87, 1536.0, 1560.0, 1584.0, 2353.0],
             [0.17, 26.77, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0],
             [0.55, 27.77, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 337.53],
             [0.25, 27.42, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.0,
              359.84],
             [0.25, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.17,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 840.0, 864.0, 888.0, 912.0, 936.0, 960.0, 984.0,
              1008.0, 1032.0, 1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.5, 1200.0, 1224.0, 1248.0, 1272.0, 1296.0,
              1320.0, 1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1514.75],
             [0.42, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.0,
              360.67, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1008.67],
             [0.17, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.33,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 672.0,
              1320.17],
             [0.17, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.33, 336.0,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 960.33, 984.0, 1008.0,
              1032.0, 1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.0, 1200.0, 1224.0, 1248.0, 1272.0, 1296.83, 1320.0,
              1344.0, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0, 1584.0, 1608.0,
              2304.83],
             [2.5, 29.33, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 337.5,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1009.33,
              1032.0, 1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.0, 1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0,
              2017.67],
             [0.92, 25.5, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.67,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1009.25,
              1032.0, 1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.0, 1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0,
              1344.58, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0, 1584.0, 1608.0, 1632.0,
              1656.0, 2016.42],
             [0.17, 24.42, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 337.17,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1345.5,
              1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0, 1584.0, 1608.0, 1632.0, 1656.0,
              1681.5, 1704.0, 1728.0, 1752.0, 1776.0, 1800.0, 1824.0, 1848.0, 1872.0, 1896.0, 1920.0, 1944.0, 1968.0,
              1992.0, 2352.83],
             [0.83, 26.33, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.0,
              361.67, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1033.42,
              1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.0, 1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0, 1345.5,
              1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0, 1584.0, 1608.0, 1632.0, 1656.0,
              1681.67],
             [1.17, 26.17, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.92,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0],
             [0.33, 27.92, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.42,
              360.0, 384.0, 408.0, 432.0, 456.0, 480.0, 504.0, 528.0, 552.0, 576.0, 600.0, 624.0, 648.0, 1008.5,
              1032.0, 1056.0, 1080.0, 1104.0, 1128.0, 1152.0, 1176.0, 1200.0, 1224.0, 1248.0, 1272.0, 1296.0, 1320.0,
              1344.67, 1368.0, 1392.0, 1416.0, 1440.0, 1464.0, 1488.0, 1512.0, 1536.0, 1560.0, 1584.0, 1608.0, 1632.0,
              1656.0, 2016.83]]


def get_measurement_regime() -> np.ndarray:
    """sample a random measurement regime from the list of regimes"""
    regim_index = np.random.choice(range(len(regimes_m)))
    random_regim = np.array(regimes_m[regim_index])
    return random_regim


def get_dosing_regim(last_measurement_time: Optional[float] = None) -> np.ndarray:
    """sample a random dosing regime from the list of regimes, if last_measurement_time is given, regimes are cut off
    at this time"""
    regim_index = np.random.choice(range(len(regimes_d)))
    random_regim = np.array(regimes_d[regim_index])
    # cut dosing regime after the last measurement
    if last_measurement_time is not None:
        random_regim = random_regim[random_regim < last_measurement_time]
    return random_regim


def get_covariates() -> (float, float):
    """get the covariates for the simulation"""
    wt = np.random.choice([83.4, 80.0, 79.3, 83.0, 80.0, 75.5, 90.0, 98.0, 83.0, 96.0, 105.0, 88.0, 106.0, 71.0,
                           83.0, 76.0, 65.0, 87.5, 83.0, 83.0, 86.0, 83.0, 83.0, 83.0, 77.0, 75.0, 66.0, 72.7, 87.3,
                           61.9, 84.4, 84.0, 57.0, 96.0, 105.0, 70.0, 63.0, 65.0, 106.0, 92.0, 66.0, 70.0, 78.7,
                           64.0,
                           72.0, 96.0, 75.0])
    dos = np.random.choice([50.0, 37.5, 25.0])
    return wt, dos
