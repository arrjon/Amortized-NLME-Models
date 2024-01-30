# Amortized-NLME-Models

Non-linear mixed-effects models are a powerful tool for studying heterogeneous populations in biology, medicine, economics, engineering, and related fields.
However, fitting these models is computationally challenging when the description of individuals is complex and the population is large.
To address this issue, we propose the use of neural density estimation to approximate individual-specific posterior distributions in an amortized fashion. 
This approximation is then used to efficiently infer population-level parameters.

More details on the method can be found in the preprint: https://doi.org/10.1101/2023.08.22.554273

In the folder `Examples` you can find a notebook `Amortized NLME Simulation & Training` which shows how to train the 
neural posterior estimator. 
The notebook `Amortized NLME Inference` shows how to use the trained neural posterior estimator for inference of a NLME model.

In the folder `Experiments` you can find the code for the experiments of the paper, to reproduce the results and generate the plots.

Below you find a step-by-step guide to use the Amortized-NLME-Models package.
The package is still under development and will be extended in the future.
If you have any questions or suggestions, please contact us.


## Installation
You can install the two main dependencies using the package manager [pip](https://pip.pypa.io/en/stable/):
- [BayesFlow](https://bayesflow.org): `pip install bayesflow` 
- [PyPesto](https://pypesto.readthedocs.io): `pip install pypesto`

Then you can clone the AmortizedNLME package using the `git clone`.

## Simulation and Training Phase using BayesFlow

Import the necessary packages
```
import numpy as np
from functools import partial

from bayesflow.simulation import Simulator
from bayesflow.simulation import GenerativeModel
from bayesflow import diagnostics
```

Define a function for simulation of data. Takes in a batch of parameters in the form
(#simulations, #parameters) and returns a batch of data in the form (#simulations, #data).

This function should be implemented in an efficient way (e.g. using [numba](https://numba.pydata.org) or import a julia function).
```
def batch_simulator(param_samples: np.ndarray, other_args) -> np.ndarray:
```

Define a class with your model, which takes in the name of your parameters and the simulator.
Here you define the prior on your individual parameters as well.
At the moment only a normal or log-normal prior is supported 
(depends on weather the simulator takes log-transformed parameters or not).
```
class myModel(NlmeBaseAmortizer):
    def __init__(self, name: str = 'myModel'):
        # define names of parameters
        param_names = ['name1', 'name2']

        # define prior values (for log-parameters)
        prior_mean = np.array([0, 0])
        prior_cov = np.diag([1, 1])
        prior_type = 'normal'

        super().__init__(name=name,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov
                         prior_type=prior_type)

        self.simulator = Simulator(batch_simulator_fun=partial(batch_simulator,
                                                                  other_args=other_args))
```
Next you can train the neural posterior estimator following the examples in
Amortized NLME Training Diagnostics.ipynb. 
The steps are explained in detail in the documentation of [BayesFlow](https://bayesflow.org), which is at the moment the backbone of the
neural posterior estimator.
One can either presimulate training data or train them online. 
Usually the first one is faster for longer simulation times, and you can reuse the data for multiple training runs of 
different architectures.
There are diagnostic tools available to check the training convergence.


## Inference Phase
Here we will use an objective class to define our objective function for the population model.
```
from inference.inference_functions import run_population_optimization
from inference.nlme_objective import ObjectiveFunctionNLME
from pypesto import visualize, optimize, profile, engine
```
First, we initialize the objective function.
Here you specify the population model, e.g. which type of covariance structure you want to use and how 
covariates might influence the population parameters.
The objective function needs to know the prior you used to train the neural posterior estimator.
At the moment only normal or uniform distributions are supported, but this can be easily extended.
```
cov_type = ['diag', 'cholesky'][0]
obj_fun_amortized = ObjectiveFunctionNLME(model_name=model.name,
                                          prior_mean=prior_mean,
                                          prior_std=prior_std,
                                          covariance_format=cov_type)
```
Next, you can start the optimization for your actual data.
For that load your data as `obs_data` in the format (#individuals, #timepoints, #data) (can be a list or an array).

```
result_optimization = run_population_optimization(
    individual_model=individual_model,  # the trained individual posterior amortizer (NlmeBaseAmortizer)
    data=obs_data,  # your data
    param_names=mixed_effect_params_names,  # the names of the population parameters
    objective_function=obj_fun_amortized,  # the objective function
    n_multi_starts=20,  # number of mulit-starts
    n_samples_opt=100,  # number of samples from posterior used for the optimization (increases accuracy and simulation time)
    x_fixed_indices=fixed_indices,  # indices of fixed parameters (e.g. variances of fixed effects should be fixed)
    x_fixed_vals=fixed_vals,  # values of fixed parameters
    file_name='Auto',  # file name for saving the results
    verbose=True,  # print optimization progress
    trace_record=True,  # record the optimization trace (can be used for visualization, faster without)
    pesto_multi_processes=8,  # number of processes used in parallel for the optimization
    )
```
The result will come in the format of a `pypesto.Result` object.
Keep in mind that different optimizer runs use different samples for the approximation of the likelihood, hence
a direct comparison of the objective values is not possible.
Therefore, `run_population_optimization` computes a new objective value on `n_samples_opt*100` samples 
(the error of the Monte Carlo integration is reduced by a factor of 10) to make results more comparable.
More details can be found here [pyPesto](https://pypesto.readthedocs.io/en/latest/api/pypesto.result.html).
pyPesto also offers a lot of tools for visualization and analysis of the results.
For example, one can quantify the uncertainty of the population parameters with profile likelihoods as follows:
```
result_optimization = profile.parameter_profile(
    problem=result_optimization.problem,
    result=result_optimization,
    optimizer=optimize.ScipyOptimizer(),
    engine=engine.MultiProcessEngine(10),
)
visualize.profiles(result_optimization)
```
By default, the profile likelihoods are computed for all parameters with an increased number of samples compared
to the optimization run.

## Contributing

We are happy about any contributions.
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
