import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from typing import Optional, Union
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import LSTM
from bayesflow.simulation import Prior, Simulator
from bayesflow.summary_networks import SequentialNetwork, SplitNetwork
from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer
from bayesflow.simulation import GenerativeModel


def split_data(i: int, x: tf.Tensor):
    """
    i index of network, x inputs to the SplitNetwork
    # Should return the input for the corresponding network
    """
    selector = tf.where(x[:, :, -1] == i, 1.0, 0.0)
    x0 = x[:, :, 0] * selector
    x1 = x[:, :, 1] * selector
    x2 = x[:, :, 2] * selector
    split_x = tf.stack((selector, x0, x1, x2), axis=-1)
    return split_x


class NlmeBaseAmortizer(ABC):
    """
       This model class contains all information needed for simulation, entity-posterior inference.
    """

    def __init__(self, name: str = 'NLMEModel',
                 network_idx: int = -1,
                 load_best: bool = False,
                 param_names: Optional[list] = None,
                 prior_mean: Optional[np.ndarray] = None,
                 prior_cov: Optional[np.ndarray] = None,
                 n_obs: Optional[int] = None, ):

        self.name = name
        # define names of parameters
        self.param_names = [] if param_names is None else param_names
        self.log_param_names = ['$\log$ ' + name for name in self.param_names]
        self.n_params = len(self.param_names)

        # define prior values (for log-parameters)
        self.prior_mean: np.ndarray = np.empty(self.n_params) if prior_mean is None else prior_mean
        self.prior_cov: np.ndarray = np.empty(self.n_params) if prior_cov is None else prior_cov
        self.prior_std: np.ndarray = np.empty(self.n_params) if prior_cov is None else np.sqrt(np.diag(prior_cov))

        # define maximal number of observations
        self.n_obs = 180 if n_obs is None else n_obs

        # training parameters
        self.n_epochs = 500
        self.n_coupling_layers = 7
        self.batch_size = 128
        self.summary_only_lstm = True
        self.split_summary = False
        self.summary_dim = 20  # default, only if summary_only_lstm is False
        self.num_conv_layers = 2  # default, only if summary_only_lstm is False
        self.summary_loss_fun = None  # 'MMD' or None
        self.final_loss = np.Inf
        self.training_time = np.Inf

        # amortizer and prior
        # must be before calling the generative model so prior etc. are up-to-date
        self.network_name = self.load_trained_model(model_idx=network_idx, load_best=load_best)
        self._build_amortizer()
        self._build_prior()
        """
        # to create trainer call
        self.build_trainer()
        """

    @abstractmethod
    def load_trained_model(self, model_idx: int = -1, load_best: bool = False) -> Optional[str]:
        # load trained model
        if model_idx == 0:
            self.n_epochs = 500
            self.n_coupling_layers = 5
            self.final_loss = np.Inf
            self.training_time = np.Inf
            model_name = 'model_XY'
            raise NotImplementedError('Model not implemented yet.')
        else:
            model_name = None
        return model_name

    def build_simulator(self, with_noise: bool = True) -> Simulator:
        """
        should be something like
        self.simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                          with_noise=with_noise))

        """
        raise NotImplementedError('Simulator not implemented yet.')

    def build_trainer(self, path_store_network: str, max_to_keep: int = 3) -> Trainer:
        simulator = self.build_simulator(with_noise=True)
        generative_model = GenerativeModel(prior=self.prior,
                                           simulator=simulator,
                                           simulator_is_batched=True,
                                           prior_is_batched=True)

        # build the trainer with networks and generative model
        trainer = Trainer(amortizer=self.amortizer,
                          generative_model=generative_model,
                          configurator=self.configured_input,
                          checkpoint_path=path_store_network,
                          max_to_keep=max_to_keep)

        print(self.amortizer.summary())
        return trainer

    def _build_amortizer(self,
                         n_coupling_layers: Optional[int] = None,
                         summary_only_lstm: Optional[bool] = None,
                         split_summary: Optional[bool] = None,
                         summary_dim: Optional[int] = None,
                         summary_loss_fun: Optional[str] = None) -> None:
        """
        build amortizer, i.e., neural networks (summary and inference networks)
        Args:
            n_coupling_layers: int - number of coupling layers
            summary_only_lstm: bool - if True, the summary network is a LSTM, otherwise it is a SequentialNetwork
            summary_dim: int - dimension of the summary network output (SequentialNetwork only)
            summary_loss_fun: str - additional loss function for latent space (misspecification detection)

        Returns: bayesflow.amortizers.AmortizedPosterior -- the amortizer
        """
        if n_coupling_layers is None:
            n_coupling_layers = self.n_coupling_layers
        if summary_only_lstm is None:
            summary_only_lstm = self.summary_only_lstm
        if split_summary is None:
            split_summary = self.split_summary
        if summary_dim is None:
            summary_dim = self.summary_dim
        if summary_loss_fun is None:
            summary_loss_fun = self.summary_loss_fun

        # 2^k hidden units s.t. 2^k > #datapoints = 8
        power_k_hidden_units = int(np.ceil(np.log2(self.n_obs)))

        if split_summary:
            if summary_only_lstm:
                network_kwargs = {'units': 2 ** power_k_hidden_units}
                network_type = LSTM
                print(f'using a split network with 2 splits, '
                      f'in each a LSTM with {2 ** power_k_hidden_units} units as summary network')
            else:
                network_kwargs = {'summary_dim': summary_dim,  # 20
                                  'num_conv_layers': self.num_conv_layers,  # 2
                                  'lstm_units': 2 ** power_k_hidden_units}
                network_type = SequentialNetwork
                print(f'using a split network with 2 splits, '
                      f'in each {self.num_conv_layers} layers of MultiConv1D,a LSTM with {2 ** power_k_hidden_units} '
                      f'units and a dense layer with output dimension {summary_dim} as summary network')

            summary_net = SplitNetwork(num_splits=2,
                                       split_data_configurator=split_data,
                                       network_type=network_type,
                                       network_kwargs=network_kwargs)
        elif summary_only_lstm:
            summary_net = LSTM(units=2 ** power_k_hidden_units)
            print(f'using LSTM with {2 ** power_k_hidden_units} units as summary network')
        else:
            summary_net = SequentialNetwork(summary_dim=summary_dim,  # 20
                                            num_conv_layers=self.num_conv_layers,  # 2
                                            lstm_units=2 ** power_k_hidden_units)
            print(
                f'using {self.num_conv_layers} layers of MultiConv1D,a LSTM with {2 ** power_k_hidden_units} '
                f'units and a dense layer with output dimension {summary_dim} as summary network')

        inference_net = InvertibleNetwork(num_params=self.n_params,
                                          num_coupling_layers=n_coupling_layers)
        print(f'using a {n_coupling_layers}-layer cINN as inference network')
        if summary_loss_fun is not None:
            print('Using MMD as additional loss function for latent space.')

        self.amortizer = AmortizedPosterior(inference_net, summary_net, summary_loss_fun=summary_loss_fun)
        return

    def _build_prior(self) -> None:
        """
        Build prior distribution.
        Returns: prior, configured_input - prior distribution and function to configure input

        """
        print('prior mean:', self.prior_mean)
        print('prior covariance diagonal:', self.prior_cov.diagonal())

        self.prior = Prior(batch_prior_fun=partial(batch_gaussian_prior,
                                                   mean=self.prior_mean,
                                                   cov=self.prior_cov),
                           param_names=self.log_param_names)

        self.configured_input = partial(configure_input, prior_means=self.prior_mean,
                                        prior_stds=self.prior_std)
        return

    def _reconfigure_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Reconfigure samples from the prior distribution to the original parameter space.
        Args:
            samples: np.ndarray - (#samples, #parameters)

        Returns: np.ndarray - (#samples, #parameters)

        """
        return self.prior_mean + samples * self.prior_std

    def draw_posterior_samples(self, data: Union[list, np.ndarray], n_samples: int = 100) -> np.ndarray:
        """
        Function to draw samples from the posterior distribution.
        Takes care of different data formats and normalization.

        Args:
            data: [list, np.ndarray] - the data
            n_samples: int - number of samples to draw

        Returns: samples - np.ndarray - (#data, #samples, #parameters)

        """
        if isinstance(data, np.ndarray):
            posterior_draws = self.amortizer.sample({'summary_conditions': data}, n_samples=n_samples)
        else:
            # data is a list (different lengths, e.g. number of observations)
            n_data = len(data)
            posterior_draws = np.zeros((n_data, n_samples, self.n_params))
            for d_idx, d in enumerate(data):
                # sample separately for each data point since points may have different number of observations
                posterior_draws[d_idx] = self.amortizer.sample({'summary_conditions': d[np.newaxis, :]},
                                                               n_samples=n_samples)
        return self._reconfigure_samples(posterior_draws)

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

        new_sims = trainer.configurator(generative_model(n_samples))
        new_sims['parameters'] = self._reconfigure_samples(new_sims['parameters'])
        return new_sims

    @abstractmethod
    def print_and_plot_example(self) -> None:
        """
            Print and plot example.
        """


def batch_gaussian_prior(mean: np.ndarray,
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
    p_samples = np.random.multivariate_normal(mean=mean,
                                              cov=cov,
                                              size=batch_size)
    return p_samples


def configure_input(forward_dict: dict,
                    prior_means: np.ndarray = None,
                    prior_stds: np.ndarray = None,
                    show_more: bool = False) -> dict:
    """
        Function to configure the simulated quantities (i.e., simulator outputs)
        into a neural network-friendly (BayesFlow) format.
    """

    # Prepare placeholder dict
    out_dict = {}

    # Convert data to float32
    data = forward_dict['sim_data'].astype(np.float32)

    # Extract prior draws
    params = forward_dict['prior_draws'].astype(np.float32)

    # z-standardize with previously computed means
    if prior_means is not None and prior_stds is not None:
        params = (params - prior_means) / prior_stds

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    if not np.all(idx_keep):
        print(f'Invalid value(s) encountered...removing {idx_keep.size - np.sum(idx_keep)} entry(ies) from batch')
        if show_more:
            bad_params = np.exp(params[~idx_keep])
            for p in bad_params.T:
                plt.figure()
                plt.hist(p)
                plt.show()

    # Add to keys
    out_dict['summary_conditions'] = data[idx_keep]
    out_dict['parameters'] = params[idx_keep]

    return out_dict
