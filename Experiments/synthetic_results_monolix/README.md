# Readme
This repository contains all necessary information to reproduce the parameter estimation results obtained with Monolix (Monolix 2023R1, Lixoft SAS, a Simulations Plus company).

## General information about the setup
To facilitate reproducibility, perform multistart optimizations and enable headless operation of Monolix, we installed Monolix and all depedencies of this project in an Ubuntu Docker container. We then interfaced with Monolix using their R library called `lixoftConnectors`. Additionally, we wrote a custom R library that mainly enables multistart optimization and saving of the results of each start. This library is called `monolixScRipts` and can be found in `miscellaneous_R_packages/`. Since the library was started for our specific usecase, it is not documented well, yet.

### Dependencies
These scripts can still be run without the container setup. One mainly needs to make sure that Monolix, `lixoftConnectors`, and `monolixScRipts` is installed and available in the environment. Other R libraries used in this project are:
- `lubridate`
- `stringr`
- `dplyr`
- `reshape2`

### Monolix optimization settings
- Project settings:
    - Automatic number of chains: off
    - Number of chains: 1
- SAEM:
    - Burn in iterations: 5
    - Exploratory phase: 
        - Auto stop criteria: on
        - Maximum number of iterations: 10000
        - Minimum number of iterations: 150
        - Stepsize exponent: 0
        - Simulated annealing: on
        - Decreasing rate of variance of residual error and individual parameters: 0.95
    - Smoothing phase:
        - Auto stop criteria: on
        - Maximum number of iterations: 1000
        - Minimum number of iterations: 50
        - Stepsize exponent: 0.7
- Standard errors:
    - Maximum number of iterations: 1000
    - Minimum number of iterations: 50
- Likelihood:
    - Monte Carlo size: 50000
    - Degrees of freedom of the t-distribution: 5

### Multistart optimization
To assess the convergence of SAEM to a global optimum, we ran the algorithm multiple times with different start points (or "initial estimates", as they are called in Monolix). (When run with the same start points, the algorithm is likely going to converge to the same result.) The start points are drawn from the prior distribution of the parameters and are reported in the paper.

The scripts are currently set up to only perform one SAEM run.

## Folder structure
### projects
**Overview** of `FroehlichModelMonolixNLME`:
```
projects/froehlich/$MODEL/synthetic_$CELLNUMBER/
    ∟code/
        ∟/project_init
        ∟multistart_monolix.R
        ∟start_multistart.sh
    ∟monolix_files/
        ∟froehlich_$MODEL/
            ∟assessment/
                ∟ ... .csv
            ∟data.csv
            ∟model.mlx
            ∟froehlich_$MODEL.mlxtran
            ∟froehlich_$MODEL.mlxproperties
```
- `$MODEL` can be `simple` or `detailed`
- `$CELLNUMBER` can be `50`, `100`, `200`, `500`, `1000`, `5000`, `10000`

**Explanation**: Monolix calls their parameter estimation problems "projects". Therefore, the folder structure here has inherited this name:
- `projects/` contains the Monolix projects for the simple and detailed Fröhlich et al. (2018) models described in the paper
    - Each setting, meaning number of cells used in the estimation, receives its own folder: `projects/froehlich/$MODEL/synthetic_$CELLNUMBER/`
- `code/` cotains all code necessary to start a multistart optimization with 100 runs
    - `project_init/` contains information about the setup of the project and necessary tables for the parameter initialization for the multistart optimization
        - `parameter_bounds_froehlich_$MODEL.csv` defines the parameter initialization for each start. Since `monolixScRipts` started with just sampling initial parameters from a uniform distribution, the naming is a bit misleading. Inside the file, you'll see three important columns `lowerBound`, `upperBound`, and `distribution`:
            - If `distribution`="uniform", `lowerBound` and `upperBound` are the bounds of the distribution
            - If `distribution`="normal", `lowerBound` is the mean and `upperBound` is the standard deviation
    - `start_multistart.sh` can be used to begin a multistart optimization. The second argument given in `Rscript ./multistart_monolix.R $(echo $currentTime) 1` being the number of starts to perform (in this case `1`)
- `monolix_files/` contains the files that monolix needs to run their project
    - `froehlich_$MODEL` contains a copy of the data and a copy of the model for each project
        - `assessment` contains the multistart results. By default the optimization is run in several rounds. For example, if you only want to do 20 starts on one day and then another 20 on another day, we create two folders and later summarize the optimization results. The summary is then placed in `assessment`.
            - The result files are separated by Monolix' "tasks" and are saved in `.csv` tables

### models
**Overview**:
```
models/
    ∟antimony/
        ∟froehlich_$MODEL.ant
    ∟monolix/
        ∟froehlich_$MODEL.mlx
```

**Explanation**:
The `models/` folder contains the ODE model files. Monolix model files, written in the [Mlxtran](https://mlxtran.lixoft.com/) language can be found in `monolix/` and are appended by `.mlx`. (While the Monolix GUI on Ubuntu, did not allow selection of a file with the extension `.mlx`, the R library `lixoftConnectors` did. The default file extension for Monolix is `.txt`). We additionally wrote models in the `antimony/` language, which was used to check if simulations looked okay.

### model_analysis
**Overview**:
```
model_analysis/
    ∟projects/
        ∟... (similar structure as `projects` folder described above)
    ∟summarize_estimation/
        ∟combine_parameter_estimation.R
        ∟create_folder_structure.R
        ∟export_estimations.py
    ∟available_project_and_models.txt
    ∟monolix_project_names.txt
```

**Explanation**:
- `model_analysis/` is used to summarize all parameter estimation results
    - For example, if you ran 20 starts on two separate days (as described above), the results would be combined into a single table containing all the information
- To separate the monolix files and the analysis files, we use `create_folder_structure.R` to create a similar folder structure in `model_analysis/projects/` as it can be found in `projects/`
- Then, we would normally combine all the parameter estimation results into single tables using `combine_parameter_estimation.R`
    - If several multistart rounds were run
- Then we can order the parameter estimations by likelihood and export them using `export_estimations.R` (if needed)
    - The exported estimations can be used with the code supplied with the paper
- The scripts in `summarize_estimation` assume that they are run with the folder as the current working directory and using R's `source()` function.
