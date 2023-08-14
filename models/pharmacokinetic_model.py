#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for a Pharmacokinetic NLME Model

# load necessary packages
import numpy as np
import pandas as pd
from numba import jit

# for plots
import matplotlib.pyplot as plt

# solving model
from scipy.integrate import solve_ivp
from scipy.stats import expon

# minor stuff
from functools import partial
from typing import Optional

from models.base_nlme_model import NlmeBaseAmortizer
from bayesflow.simulation import Simulator
from bayesflow.trainers import Trainer
from bayesflow.simulation import GenerativeModel

from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import convert as jlconvert

jlPkg.activate("models/SimulatorPharma")
jl.seval("using SimulatorPharma")


def measurement_model(a: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    if a.shape[1] == 5:
        y = a[:, 1:3].copy()
    else:
        y = a
    y[y < threshold] = np.exp(threshold)
    y = np.log(y)
    return y


def add_noise(y: np.ndarray, theta: np.ndarray, sigma: float) -> np.ndarray:
    eps = np.random.normal(loc=0, scale=sigma, size=y.shape)
    y[:, 0] += theta[8] * eps[:, 0]  # theta11
    y[:, 1] += theta[9] * eps[:, 1]  # theta12
    return y


@jit(nopython=True)
def ode_model_pk(t: float,
                 A: np.ndarray,
                 theta: np.ndarray,
                 eta: np.ndarray,
                 wt: float) -> np.ndarray:
    # helper variables
    ASCL = (wt / 70.) ** 0.75
    ASV = wt / 70.

    k_a = theta[0]
    V_2 = theta[1] * ASV * eta[0]  # np.exp(eta[0])
    Q_H = 80 * ASCL  # theta[2] * ASCL 
    CLP = theta[2] * ASCL * eta[2]  # np.exp(eta[2])
    CLM = theta[3] * ASCL
    V_3 = theta[4] * ASV * eta[1]  # np.exp(eta[1])
    Q_34 = theta[5] * ASCL
    V_4 = theta[6] * ASV
    f_m = 0.21 * eta[3]  # theta[8] * eta[3]  #np.exp(eta[3])
    Q_25 = theta[7] * ASCL
    V_5 = 588. * ASV  # theta[10] * ASV

    CLIV = (k_a * A[0] + Q_H / V_2 * A[1]) / (Q_H + CLP)

    # ODEs
    dA1dt = -k_a * A[0]
    dA2dt = Q_H * CLIV - Q_H / V_2 * A[1] - Q_25 / V_2 * A[1] + Q_25 / V_5 * A[4]
    dA3dt = f_m * CLP * CLIV - CLM / V_3 * A[2] - Q_34 / V_3 * A[2] + Q_34 / V_4 * A[3]
    dA4dt = Q_34 / V_3 * A[2] - Q_34 / V_4 * A[3]
    dA5dt = Q_25 / V_2 * A[1] - Q_25 / V_5 * A[4]

    return np.array([dA1dt, dA2dt, dA3dt, dA4dt, dA5dt])


def inner_simulator(theta: np.ndarray,
                    eta: np.ndarray,
                    wt: float,
                    DOS: float,
                    times_new_dose: np.ndarray,
                    t_measurement: np.ndarray) -> np.ndarray:
    x_start = np.array([0., 0., 0., 0., 0.])

    t_start = 0
    final_time = t_measurement[-1]
    values = []
    if times_new_dose[-1] < final_time:
        stopping_list = np.concatenate((times_new_dose, [final_time]), axis=0)

    else:
        stopping_list = times_new_dose

    for t_index, t_next in enumerate(stopping_list):

        eval_points = [t for t in t_measurement if t_start <= t < t_next] + [t_next]
        sol = solve_ivp(fun=ode_model_pk, args=(theta, eta, wt),
                        t_span=(t_start, t_next), t_eval=eval_points,
                        y0=x_start)

        # compute new starting values
        t_start = sol['t'][-1]
        x_start = sol['y'][:, -1]

        if t_index < times_new_dose.size:
            # final_time might not be a dose
            x_start[0] += DOS

        # append values and time points
        if t_index < stopping_list.size - 1:
            values.append(sol['y'][:, :-1])
        else:
            # append the last time point
            values.append(sol['y'])

    values = np.concatenate(values, axis=1).T

    if t_measurement.size != values.shape[0]:
        raise ValueError('t_measurement and time_points have different sizes')
        # interpolate to the correct time points
        # sol = np.interp(t_measurement, time_points, values.T)
    return values


def get_random_measurements(exp_scale: float, final_time: float, max_n: int) -> np.ndarray:
    # sample the measurement time points
    t_measurement = []
    t_current = 0
    for i in range(max_n):
        t_until_next_measurement = expon.rvs(scale=exp_scale)
        t_current += t_until_next_measurement
        if t_current < final_time:
            t_measurement.append(t_current)
        else:
            # out of time
            break
    return np.array(t_measurement)


def full_simulator(log_params: np.ndarray,
                   wt: float,
                   DOS: float,
                   t_measurement: np.ndarray,
                   t_doses: np.ndarray,
                   use_julia: bool = True,
                   return_all_states: bool = True,
                   with_noise: bool = False) -> np.ndarray:
    if use_julia and return_all_states:
        raise ValueError('Julia and return_all_states=True is not supported.')

    # simulate batch
    theta = np.exp(log_params[:10])
    eta = np.exp(log_params[10:14])
    sigma = np.exp(log_params[-1])

    # simulate the data
    if use_julia:
        # convert to julia types
        (theta_0, theta_1, theta_2, theta_3, theta_4, theta_5,
         theta_6, theta_7, theta_8, theta_9) = theta
        eta_0, eta_1, eta_2, eta_3 = eta
        jl_t_doses = jlconvert(jl.Vector[jl.Float64], t_doses)
        jl_t_measurement = jlconvert(jl.Vector[jl.Float64], t_measurement)
        # simulate
        a_sim = jl.simulatePharma(theta_0, theta_1, theta_2, theta_3, theta_4,
                                  theta_5, theta_6, theta_7, theta_8, theta_9,
                                  eta_0, eta_1, eta_2, eta_3,
                                  wt, DOS,
                                  jl_t_doses, jl_t_measurement).to_numpy().T
    else:
        a_sim = inner_simulator(theta=theta,
                                eta=eta,
                                wt=wt,
                                DOS=DOS,
                                times_new_dose=t_doses,
                                t_measurement=t_measurement)
    if return_all_states:
        return a_sim
    y_sim = measurement_model(a_sim)

    if with_noise:
        y_sim = add_noise(y_sim, theta=theta, sigma=sigma)
    return y_sim


def batch_simulator(param_batch: np.ndarray,
                    final_time: float,
                    exp_scale_measurement: Optional[float] = None,
                    exp_scale_dose: Optional[float] = None,
                    use_julia: bool = True,
                    with_noise: bool = True) -> np.ndarray:
    # sample the measurement time points
    if exp_scale_measurement is not None:
        max_n_measurements = np.random.randint(1, 30)
        t_measurement = get_random_measurements(exp_scale=exp_scale_measurement,
                                                final_time=final_time,
                                                max_n=max_n_measurements)
    else:
        # fixed measurement time points
        t_measurement = np.linspace(0, final_time, 100)

    if t_measurement.size == 0:
        print('no measurement')
        return np.nan

    if exp_scale_dose is not None:
        # sample the dosing time points
        t_doses = get_random_measurements(exp_scale=exp_scale_dose,
                                          final_time=t_measurement[-1],
                                          max_n=200)
    else:
        # fixed dosing time points
        t_doses = np.array([26, 50, 74, 98, 122, 146, 170, 194, 218,
                            242, 266, 290, 314, 338, 362, 386, 410, 434,
                            458, 482, 506, 530, 554, 578, 602, 626, 650, 674])

    # sample parameters which are not optimized since they are observed
    # wt = np.random.normal(loc=80, scale=15)  # size=param_batch.shape[0]))
    # if wt < 10:
    #    wt = 10.
    # DOS = np.random.choice([50.0, 37.5, 25.0])  # size=param_batch.shape[0]))

    # simulate batch
    n_sim = param_batch.shape[0]
    output_batch = np.zeros((n_sim, t_measurement.size + t_doses.size, 4),
                            dtype=np.float32)

    for pars_i, log_params in enumerate(param_batch):
        theta = np.exp(log_params[:10])
        eta = np.exp(log_params[10:14])
        wt, DOS = np.exp(log_params[14:16])
        # wt, DOS = log_params[14:16]
        sigma = np.exp(log_params[-1])

        # simulate the data
        if use_julia:
            # convert to julia types
            (theta_0, theta_1, theta_2, theta_3, theta_4, theta_5,
             theta_6, theta_7, theta_8, theta_9) = theta
            eta_0, eta_1, eta_2, eta_3 = eta
            jl_t_doses = jlconvert(jl.Vector[jl.Float64], t_doses)
            jl_t_measurement = jlconvert(jl.Vector[jl.Float64], t_measurement)
            # simulate
            a_sim = jl.simulatePharma(theta_0, theta_1, theta_2, theta_3, theta_4,
                                      theta_5, theta_6, theta_7, theta_8, theta_9,
                                      eta_0, eta_1, eta_2, eta_3,
                                      wt, DOS,
                                      jl_t_doses, jl_t_measurement).to_numpy().T
        else:
            a_sim = inner_simulator(theta=theta,
                                    eta=eta,
                                    wt=wt,
                                    DOS=DOS,
                                    times_new_dose=t_doses,
                                    t_measurement=t_measurement)

        # convert to measurements
        y_sim = measurement_model(a_sim)
        if with_noise:
            y_sim = add_noise(y_sim, theta=theta, sigma=sigma)

        # reshape the data to fit in one numpy array
        output_batch[pars_i, :, :] = convert_observables(y=y_sim,
                                                         t_measurements=t_measurement,
                                                         doses_time_points=t_doses,
                                                         DOS=DOS,
                                                         wt=wt)
    return output_batch


def convert_output_to_simulation(output: np.ndarray,
                                 scaling_measurements: float = 5.,
                                 scaling_time: float = 4000.) -> (np.ndarray, np.ndarray, float, float):
    """
    Convert the output of the simulator to a reasonable format for plotting
    Args:
        output: output of the simulator
        scaling_measurements: scaling factor for the output
        scaling_time: scaling time for the output

    Returns: y, y_time_points, doses_time_points, DOS, wt

    """
    n_time, _ = output.shape

    measurements = output[np.where(output[:, 3] == 0)]
    y = measurements[:, :3]
    y[:, :2] *= scaling_measurements
    y[:, 2] *= scaling_time

    doses = output[np.where(output[:, 3] == 1)]
    DOS = np.power(10, doses[0, 0])
    wt = np.power(10, doses[0, 1])
    doses_time_points = doses[:, 2] * scaling_time

    return y, doses_time_points, DOS, wt


def convert_observables(y: np.ndarray,
                        t_measurements: np.ndarray,
                        doses_time_points: np.ndarray,
                        DOS: float,
                        wt: float,
                        scaling_measurements: float = 5.,
                        scaling_time: float = 4000.) -> np.ndarray:
    # reshape the data to fit in one numpy array
    measurements = np.concatenate((y / scaling_measurements,
                                   t_measurements[:, np.newaxis] / scaling_time,
                                   np.zeros((t_measurements.size, 1))),
                                  axis=1)
    doses = np.stack(([np.log10(DOS)] * doses_time_points.size,
                      [np.log10(wt)] * doses_time_points.size,
                      doses_time_points / scaling_time,
                      np.ones(doses_time_points.size)),
                     axis=1)
    return np.concatenate((measurements, doses), axis=0)


def convert_csv_to_simulation_data(csv_data: pd.DataFrame) -> list:
    """
    Convert csv data to simulation data
    Args:
        csv_data:  pd.DataFrame

    Returns:
        data_list: list of tuples (patient_id, wt, DOS, measurement_data_time, measurement_data_A2, measurement_data_A3)

    """
    data_list = []
    for patient_id in csv_data['ID'].unique():
        patient_data = csv_data[csv_data['ID'] == patient_id]

        wt = patient_data['WT'].unique()[0]
        DOS = patient_data['DOS'].unique()[0]

        measurement_data = patient_data.loc[patient_data['EVID'] == 0, ['TIME', 'CMT', 'LNDV']]
        measurement_data_time = measurement_data.loc[measurement_data['CMT'] == 2, ['TIME']].to_numpy().flatten()
        measurement_data_A2 = measurement_data.loc[measurement_data['CMT'] == 2, ['LNDV']].to_numpy().flatten()
        measurement_data_A3 = measurement_data.loc[measurement_data['CMT'] == 3, ['LNDV']].to_numpy().flatten()
        if measurement_data_time.shape[0] == 0:
            continue

        measurements = np.stack((measurement_data_A2,
                                 measurement_data_A3,
                                 measurement_data_time), axis=1)

        dosage_data = patient_data.loc[patient_data['EVID'] == 1, ['TIME']].to_numpy().flatten()
        data_list.append([patient_id, measurements, dosage_data, DOS, wt])
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
def load_data(file_name: str = 'data/Suni_PK_final.csv', number_data_points: int = None) -> list[np.ndarray]:
    data_raw = read_csv_pharma(file_name)
    data = convert_csv_to_simulation_data(data_raw)

    if number_data_points is not None:
        data = data[:number_data_points]

    # convert data to a BayesFlow format
    data_bayesflow = []
    for d in data:
        patient_id, measurements, doses_time_points, DOS, wt = d
        observables = convert_observables(y=measurements[:, :2],
                                          t_measurements=measurements[:, 2],
                                          doses_time_points=doses_time_points, DOS=DOS, wt=wt)
        data_bayesflow.append(observables)
    return data_bayesflow


def batch_gaussian_prior_pharma(mean: np.ndarray,
                                cov: np.ndarray,
                                batch_size: int) -> np.ndarray:
    # it is used as a log-normal prior
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:h
    mean : np.ndarray - mean of the normal distribution
    cov: np.ndarray - covariance of the normal distribution
    batch_size : int - number of samples to draw from the prior
    ----------

    Output:
    p_samples : np.ndarray of shape (batch size, parameter dimension) -- the samples batch of parameters
    """
    # Prior ranges for the simulator
    norm_mean = np.concatenate((mean[:-3], [mean[-1]]))
    var = cov.diagonal()
    norm_var = np.concatenate((var[:-3], [var[-1]]))
    norm_cov = np.diag(norm_var)
    p_samples_norm = np.random.multivariate_normal(mean=norm_mean,
                                                   cov=norm_cov,
                                                   size=batch_size)
    wt = np.random.normal(loc=mean[-3], scale=var[-3], size=batch_size)
    wt[wt < 10] = 10
    wt_log = np.log(wt)
    DOS = np.random.choice([50.0, 37.5, 25.0], size=batch_size)
    DOS_log = np.log(DOS)

    p_samples = np.concatenate((p_samples_norm[:, :-1],
                                wt_log[:, np.newaxis], DOS_log[:, np.newaxis],
                                p_samples_norm[:, -1][:, np.newaxis]), axis=1)
    return p_samples


class PharmacokineticModel(NlmeBaseAmortizer):
    def __init__(self, name: str = 'PharmacokineticModel', network_idx: int = -1, load_best: bool = False):
        # define names of parameters
        param_names = ['$\\theta_0$',
                       '$\\theta_1$',
                       '$\\theta_2$',
                       '$\\theta_3$',
                       '$\\theta_4$',
                       '$\\theta_5$',
                       '$\\theta_6$',
                       '$\\theta_7$',
                       '$\\theta_8$',
                       '$\\theta_9$',
                       '$\eta_0$',
                       '$\eta_1$',
                       '$\eta_2$',
                       '$\eta_3$',
                       'wt',
                       'DOS',
                       '$\sigma$']

        # define prior values (for log-parameters)
        prior_mean = np.array([-5, 6.5, 2.5, 2.5, 6.5, 0, 6.5, -3, -3, -3, 0, 0, 0, 0, 80, 1, -1])
        prior_cov = np.diag(np.array([4.5, 1, 1, 1, 1, 1, 1, 4.5, 4.5, 4.5, 1, 1, 1, 1, 15, 1 / 3, 2]))

        # simulation variables
        self.final_time = 4000  # maximal time of modelling in minutes
        self.exp_scale_measurement = 182  # mean distance between measurements in minutes
        self.exp_scale_dose = 24  # mean distance between doses in minutes

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         n_obs=230)  # 26 measurement max, 179 dosages max

        print('Using the PharmacokineticModel')

    def _build_prior(self) -> None:
        from models.base_nlme_model import configure_input
        from bayesflow.simulation import Prior
        """
        Build prior distribution.
        Returns: prior, configured_input - prior distribution and function to configure input

        """
        self.prior = Prior(batch_prior_fun=partial(batch_gaussian_prior_pharma,
                                                   mean=self.prior_mean,
                                                   cov=self.prior_cov),
                           param_names=self.log_param_names)

        self.configured_input = partial(configure_input, prior_means=self.prior_mean,
                                        prior_stds=self.prior_std)
        return

    def load_trained_model(self, model_idx: int = 0, load_best: bool = False) -> Optional[str]:
        # presimulation time: 453.03 h / 8 kernels = 56.63 h
        # training time: (775.09 + 227.69 + 771.95 + 766.13 + 612.81) / 5 / 60 = 10.51 h
        # total time: 67.14 h

        # load best
        if load_best:
            model_idx = 3  # or 1,3,4

        if model_idx == 0:  # eta_3, wt, DOS badly calibrated, rest nicely
            self.n_epochs = 1000  # 867 by early_stopping
            self.n_coupling_layers = 7
            self.summary_only_lstm = True
            self.split_summary = True

            self.final_loss = 8.525
            self.training_time = 775.09
        elif model_idx == 1:  # wt, DOS badly calibrated, rest nicely
            self.n_epochs = 1000  # 306 by early_stopping
            self.n_coupling_layers = 8
            self.summary_only_lstm = True
            self.split_summary = True

            self.final_loss = 15.144
            self.training_time = 227.69
        elif model_idx == 2:  # theta_8, wt, DOS badly calibrated, rest nicely
            self.n_epochs = 1000  # 973 by early_stopping
            self.n_coupling_layers = 7
            self.summary_only_lstm = False
            self.split_summary = True

            self.final_loss = 12.414
            self.training_time = 771.95
        elif model_idx == 3:  # wt, DOS badly calibrated, rest nicely
            self.n_epochs = 1000  # 896 by early_stopping
            self.n_coupling_layers = 8
            self.summary_only_lstm = False
            self.split_summary = True

            self.final_loss = 10.462
            self.training_time = 766.13
        elif model_idx == 4:  # wt, DOS badly calibrated, rest nicely
            self.n_epochs = 1000  # 930 by early_stopping
            self.n_coupling_layers = 7
            self.summary_only_lstm = True
            self.split_summary = False

            self.final_loss = 11.390
            self.training_time = 612.81
        else:
            return None

        model_name = f'amortizer-pharma' \
                     f'-{self.n_coupling_layers}layers' \
                     f'{"-LSTM" if self.summary_only_lstm else ""}' \
                     f'{"-summary_loss" if self.summary_loss_fun is not None else ""}' \
                     f'{"-without_split_summary" if not self.split_summary else ""}' \
                     f'-{self.n_epochs}epochs'
        return model_name

    def old_load_trained_model(self, model_idx: int = 0, load_best: bool = False) -> Optional[str]:
        # simulation time 123937.63 minutes / 48 kernels = 2582.04 minutes = 43.03 hours
        # simulation time new pior 83424.22 minutes / 8 kernels = 10428.03 minutes = 173.80 hours

        # define prior values (for log-parameters)
        theta = np.array([0.133, 1820, 33.9, 16.5, 730, 2.75, 592, 0.371, 0.367, 0.281])
        eta = np.array([1, 1, 1, 1])
        wt = np.array([80])
        DOS = np.array([40])
        sigma = np.array([1])

        exp_mean = np.concatenate((theta, eta, wt, DOS, sigma))
        exp_var = 5 * np.ones(exp_mean.size)

        self.prior_mean = np.floor(np.log(exp_mean ** 2 / np.sqrt(exp_mean ** 2 + exp_var)))
        self.prior_cov = np.ceil(np.diag(np.log(1 + exp_var / (exp_mean ** 2))))
        self.prior_std = np.sqrt(np.diag(self.prior_cov))

        # load best
        if load_best:
            model_idx = 8  # 7 #3

        if model_idx == 0:
            self.n_coupling_layers = 6
            self.n_epochs = 500
            self.final_loss = 1.595
            self.training_time = 258.48
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-6layers-LSTM-500epochs'
        elif model_idx == 1:
            self.n_coupling_layers = 6
            self.n_epochs = 500
            self.final_loss = 2.321
            self.training_time = 265.83
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-6layers-LSTM-500epochs-2'
        elif model_idx == 2:
            self.n_coupling_layers = 7
            self.n_epochs = 500
            self.final_loss = 3.705
            self.training_time = 281.65
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-7layers-LSTM-500epochs'
        elif model_idx == 3:
            self.n_coupling_layers = 6
            self.n_epochs = 1000
            self.final_loss = 0.172
            self.training_time = 529.46
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-6layers-LSTM-1000epochs'
        elif model_idx == 4:
            self.n_coupling_layers = 7
            self.n_epochs = 1000
            self.final_loss = 0.506
            self.training_time = 580.04
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-7layers-LSTM-1000epochs'
        elif model_idx == 5:
            self.prior_mean = np.array([-5, 6.5, 2.5, 2.5, 6.5, 0, 6.5, -3, -3, -3, 0, 0, 0, 0, 3, 3, -1])
            cov_diag = np.array([4.5, 1, 1, 1, 1, 1, 1, 4.5, 4.5, 4.5, 1, 1, 1, 1, 1, 1, 2])
            self.prior_cov = np.diag(cov_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 7
            self.n_epochs = 500
            self.final_loss = 2.487
            self.training_time = 283.57
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-7layers-LSTM-500epochs-prior2'
        elif model_idx == 6:
            self.prior_mean = np.array([-5, 6.5, 2.5, 2.5, 6.5, 0, 6.5, -3, -3, -3, 0, 0, 0, 0, 3, 3, -1])
            cov_diag = np.array([4.5, 1, 1, 1, 1, 1, 1, 4.5, 4.5, 4.5, 1, 1, 1, 1, 1, 1, 2])
            self.prior_cov = np.diag(cov_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 6
            self.n_epochs = 500
            self.final_loss = 4.960
            self.training_time = 275.39
            self.summary_only_lstm = True
            model_name = 'amortizer-pharma-6layers-LSTM-500epochs-prior2'
        elif model_idx == 7:
            self.prior_mean = np.array([-5, 6.5, 2.5, 2.5, 6.5, 0, 6.5, -3, -3, -3, 0, 0, 0, 0, 3, 3, -1])
            cov_diag = np.array([4.5, 1, 1, 1, 1, 1, 1, 4.5, 4.5, 4.5, 1, 1, 1, 1, 1, 1, 2])
            self.prior_cov = np.diag(cov_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 7
            self.n_epochs = 500
            self.final_loss = -1.544
            self.training_time = 324.48
            self.summary_only_lstm = False
            model_name = 'amortizer-pharma-7layers-summary-500epochs-prior2'
        elif model_idx == 8:
            self.prior_mean = np.array([-5, 6.5, 2.5, 2.5, 6.5, 0, 6.5, -3, -3, -3, 0, 0, 0, 0, 3, 3, -1])
            cov_diag = np.array([4.5, 1, 1, 1, 1, 1, 1, 4.5, 4.5, 4.5, 1, 1, 1, 1, 1, 1, 2])
            self.prior_cov = np.diag(cov_diag)
            self.prior_std = np.sqrt(np.diag(self.prior_cov))

            self.n_coupling_layers = 6
            self.n_epochs = 500
            self.final_loss = -0.239
            self.training_time = 293.64
            self.summary_only_lstm = False
            model_name = 'amortizer-pharma-6layers-summary-500epochs-prior2'

        else:
            return None
        return model_name

    def build_simulator(self, with_noise: bool = True, use_julia: bool = True) -> Simulator:
        # build simulator
        simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                          final_time=self.final_time,
                                                          exp_scale_measurement=self.exp_scale_measurement,
                                                          exp_scale_dose=self.exp_scale_dose,
                                                          use_julia=use_julia,
                                                          with_noise=with_noise))
        return simulator

    def generate_simulations_from_prior(self,
                                        n_samples: int,
                                        trainer: Trainer,
                                        simulator: Simulator) -> dict:
        """
        Function to generate samples from the prior distribution.
        Takes care of different data formats and normalization.
        """
        generative_model = GenerativeModel(prior=self.prior,
                                           simulator=simulator,
                                           simulator_is_batched=True,
                                           prior_is_batched=True)

        # sample separately for each data point since points may have different number of observations
        new_sims = {'summary_conditions': [], 'parameters': []}
        for i_sample in range(n_samples):
            single_new_sims = trainer.configurator(generative_model(1))
            single_new_sims['parameters'] = self._reconfigure_samples(single_new_sims['parameters'])
            new_sims['summary_conditions'].append(single_new_sims['summary_conditions'].squeeze(axis=0))
            new_sims['parameters'].append(single_new_sims['parameters'].squeeze(axis=0))
        new_sims['parameters'] = np.array(new_sims['parameters'])
        return new_sims

    @staticmethod
    def load_data(file_name: str = 'data/Suni_PK_final.csv', n_data: int = None) -> list[np.ndarray]:
        data = load_data(file_name=file_name, number_data_points=n_data)
        return data

    def print_and_plot_example(self,
                               # data from patient
                               DOS: float = 50.,
                               sex: (0, 1) = 0,
                               weight: float = 73.) -> None:
        if weight == -99 and sex == 1:
            wt = 83
        elif weight == -99 and sex == 0:
            wt = 75
        else:
            wt = weight

        times_new_dose = np.array([26, 50, 74, 98, 122, 146, 170, 194, 218,
                                   242, 266, 290, 314, 338, 362, 386, 410, 434,
                                   458, 482, 506, 530, 554, 578, 602, 626, 650, 674])

        LNDV_A2 = np.array([[222.5833333, 387.1666667, 555.5, 699.2833333, 890.75, 1060.016667, 1203.35],
                            [2.838203227, 3.115810008, 3.156100868, 3.217534125, 0.224334841, -1.415023108,
                             -2.202106474]])
        LNDV_A3 = np.array([[222.5833333, 387.1666667, 555.5, 699.2833333, 890.75, 1060.016667, 1203.35],
                            [2.547075177, 2.850446352, 2.841870469, 2.694431891, 1.260141395, 0.200709786,
                             -0.309736834]])

        # set some parameters
        eta = np.array([1, 1, 1, 1])
        # theta = [0.133, 1820, 80, 33.9, 16.5, 730, 2.75, 592, 0.21, 0.371, 588, 0.367, 0.281]
        theta = np.array([0.133, 1820, 33.9, 16.5, 730, 2.75, 592, 0.371, 0.367, 0.281])
        sigma = 1

        params = np.concatenate((theta, eta, [wt], [DOS], [sigma]), axis=0)
        df_params = pd.DataFrame(params.reshape(1, params.size), columns=self.param_names)
        display(df_params)

        log_params = np.log(params)
        param_batch = log_params.reshape((1, log_params.size))
        output = batch_simulator(param_batch,
                                 final_time=self.final_time,
                                 exp_scale_measurement=self.exp_scale_measurement,
                                 exp_scale_dose=self.exp_scale_dose)
        (y_sim,
         doses_time_points,
         DOS_sim,
         wt_sim) = convert_output_to_simulation(output[0, :, :])

        plt.figure(tight_layout=True, figsize=(10, 5))
        plt.scatter(y_sim[:, 2], y_sim[:, 0], color='blue', label=f'simulated $A_{2}$')
        plt.scatter(y_sim[:, 2], y_sim[:, 1], color='red', label=f'simulated $A_{3}$')
        plt.vlines(doses_time_points, 0, 1)
        plt.title(f'Patient Simulation with DOS={np.round(DOS_sim, 2)}, wt={np.round(wt_sim, 2)}')
        plt.legend()
        plt.show()

        final_time = LNDV_A2[0, -1]
        t_sim = np.linspace(0, final_time, 1000)
        a_sim_2 = inner_simulator(
            theta, eta, wt, DOS, times_new_dose, t_sim)

        y_sim_2 = measurement_model(a_sim_2)
        # y_sim_2 = add_noise(y_sim_2, theta=theta, sigma=sigma)

        plt.plot(t_sim, y_sim_2[:, 0], color='blue', label=f'simulated $A_{2}$')
        plt.plot(t_sim, y_sim_2[:, 1], color='red', label=f'simulated $A_{3}$')
        plt.vlines(times_new_dose, 0, 1)

        plt.scatter(LNDV_A2[0, :], LNDV_A2[1, :], color='blue', label=f'real $A_2$')
        plt.scatter(LNDV_A3[0, :], LNDV_A3[1, :], color='red', label=f'real $A_3$')
        plt.title(f'Patient with DOS={DOS}, wt={wt} compared to real data')

        plt.legend()
        plt.show()
        return
