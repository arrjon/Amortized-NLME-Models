import os
import numpy as np
import pandas as pd
from sklearn.covariance import empirical_covariance

path = ''  # use local path

# Load the individual model
# specify which model to use
model_name = ['fröhlich-simple', 'fröhlich-detailed', 'fröhlich-sde'][1]

if model_name == 'fröhlich-simple':
    from models.froehlich_model_simple import FroehlichModelSimple, batch_simulator
    model = FroehlichModelSimple()

    # mean of the variables (not of the normal distribution!)
    delta = 0.8  # np.log(2)/(0.8) # per hour
    gamma = 0.03  # np.log(2)/(22.8) # per hour
    k_m0_scale = 502  # a.u/h
    t_0 = 0.9  # in hours
    offset = 8  # a.u
    sigma = np.sqrt(0.001)  # must be positive
    theta_population = np.array([delta, gamma, k_m0_scale, t_0, offset, sigma])
    # covariance of random effects
    cov = np.diag(np.array([1, 1, 1000, 0.1, 0, 0]))  # variances of the variables
elif model_name == 'fröhlich-detailed':
    from models.froehlich_model_detailed import FroehlichModelDetailed, batch_simulator
    model = FroehlichModelDetailed()

    # mean of the variables (not of the normal distribution!)
    delta1_m0 = 1.2  # [h]
    delta2 = 0.6  # [h]
    e0_m0 = 0.85  #
    k2_m0_scale = 1e6  # [a.u/h]
    k2 = 1.9  # [1/h]
    k1_m0 = 5.5  # [1/h]
    r0_m0 = 1e-1  # [1]
    gamma = 0.01  # [1/h]
    t_0 = 0.9  # [h]
    offset = 8  # [a.u]
    sigma = np.sqrt(0.001)
    theta_population = np.array([delta1_m0, delta2, e0_m0, k2_m0_scale, k2,
                                 k1_m0, r0_m0, gamma, t_0, offset, sigma])
    # covariance of random effects
    cov = np.diag(np.array([1.1, 0.4, 0.5, 10, 2, 100, 0.1, 0.01, 0.5, 0, 0]))  # variances of the variables
elif model_name == 'fröhlich-sde':
    from models.froehlich_model_sde import FroehlichModelSDE, batch_simulator
    model = FroehlichModelSDE()

    # mean of the variables (not of the normal distribution!)
    delta = 0.8  # np.log(2)/(0.8) # per hour
    gamma = 0.03  # np.log(2)/(22.8) # per hour
    k = 1.44
    m0 = 300
    scale = 2.12
    t_0 = 0.9  # in hours
    offset = 8  # a.u
    sigma = np.sqrt(0.001)  # must be positive
    theta_population = np.array([delta, gamma, k, m0, scale, t_0, offset, sigma])
    # covariance of random effects
    cov = np.diag(np.array([1, 1, 2, 5, 0, 0.1, 0, 0]))  # variances of the variables
else:
    raise NotImplementedError('model not implemented')

# create lists with names of parameters
pop_param_names = ['pop-' + name for name in model.param_names]
var_param_names = ['var-' + name for name in model.param_names]
full_param_names = pop_param_names + var_param_names

# Creating Synthetic Data
# We sample cell-specific parameters from population parameters and create synthetic observations.
# - all parameters are log-normally distributed
# - offset and sigma do not vary over the data set


def create_synthetic_data(n_cells: int,
                          population_param: np.ndarray,
                          cov_full: np.ndarray,
                          n_obs: int = 180,
                          seed: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):
    np.random.seed(seed)

    params_with_var = len(population_param) - 2

    # cell-specific parameters with mixed effects
    mean = population_param[:params_with_var]
    cov = np.diag(np.diag(cov_full)[:params_with_var])

    # convert to log-normal mean and cov
    log_mean = np.log(mean ** 2 / np.sqrt(mean ** 2 + np.diag(cov)))
    log_cov = np.log(1 + np.diag(cov) / (mean ** 2))

    # cell-specific parameters with fixed effects
    offset = population_param[params_with_var]
    sigma = population_param[params_with_var + 1]

    # all parameters except offset and noise are log-normal, those are fixed (and log) per experiment
    log_mean = np.concatenate((log_mean, np.array([np.log(offset), np.log(sigma)])), axis=0)
    log_cov = np.concatenate((log_cov, np.array([0, 0])), axis=0)

    # sample
    cell_param_log = np.random.multivariate_normal(log_mean, np.diag(log_cov), n_cells)

    # simulate cells for parameters
    obs_data = batch_simulator(cell_param_log, n_obs=180).reshape((n_cells, n_obs))

    # true parameter vector
    pop_params = np.concatenate((log_mean, log_cov), axis=0)

    return cell_param_log, obs_data, pop_params


# %%
def get_sample_parameter_estimate(cell_param_log: np.ndarray) -> np.ndarray:
    # estimate sample population parameters (will differ more for smaller samples)
    true_param_mean = np.mean(cell_param_log, axis=0)
    true_param_cov_diag = np.diag(empirical_covariance(cell_param_log))
    true_param_sample = np.concatenate((true_param_mean, true_param_cov_diag))
    return true_param_sample


# %%
# create synthetic data
n_cells = [50, 100, 200, 500, 1000, 5000, 10000]

# create data and true parameters
cell_param_log, obs_data, true_pop_params = create_synthetic_data(n_cells[-1], theta_population, cov)

# parameters estimated from sample (only those can be recovered)
true_pop_param_sample = np.zeros((len(n_cells), model.n_params * 2))
for i, cells in enumerate(n_cells):
    true_pop_param_sample[i] = get_sample_parameter_estimate(cell_param_log[:cells])

all_params = np.concatenate((true_pop_params[np.newaxis, :], true_pop_param_sample)).round(5)
# %%
# save data
t_points = np.linspace(start=1 / 6, stop=30, num=180, endpoint=True)

# save into csv data
df_obs = pd.DataFrame(np.exp(obs_data.reshape(n_cells[-1], 180).T),
                      index=(t_points * 60 * 60).round(0).astype(int))
df_params = pd.DataFrame(cell_param_log, columns=model.log_param_names)
df_sample_pop_parameters = pd.DataFrame(all_params, columns=full_param_names, index=['true'] + n_cells)

if model_name == 'fröhlich-simple':
    filename_obs = path + "data_random_cells.csv"
    filename_params = path + "synthetic_individual_cell_params.csv"
    filename_pop_params = path + "sample_pop_parameters.csv"
elif model_name == 'fröhlich-detailed':
    filename_obs = path + "data_random_cells_detailed_model.csv"
    filename_params = path + "synthetic_individual_cell_params_detailed_model.csv"
    filename_pop_params = path + "sample_pop_parameters_detailed_model.csv"
elif model_name == 'fröhlich-sde':
    filename_obs = path + "data_random_cells_sde_model.csv"
    filename_params = path + "synthetic_individual_cell_params_sde_model.csv"
    filename_pop_params = path + "sample_pop_parameters_sde_model.csv"
else:
    raise NotImplementedError

# only save if not there
if not os.path.exists(filename_obs):
    df_obs.to_csv(filename_obs)
if not os.path.exists(filename_params):
    df_params.to_csv(filename_params)
if not os.path.exists(filename_pop_params):
    df_sample_pop_parameters.to_csv(filename_pop_params)
