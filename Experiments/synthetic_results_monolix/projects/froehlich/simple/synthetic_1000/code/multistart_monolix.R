#!/usr/bin/R
# This will use R to run a multistart optimization using Monolix
# The working directory should be the directory of this script
# Autoload the right environment
renv::autoload()
# Load libraries
library(lixoftConnectors)
library(monolixScRipts)
library(lubridate)
# Initialize the LixoftConnectors
initializeLixoftConnectors(software = "monolix")

# Get commandline arguments (should contain a unique ID for the multistart)
args <- commandArgs(trailingOnly = TRUE)

# Get the unique ID for the multistart
multistartId <- as.character(args[1])

currWd <- getwd()
baseDir <- paste0(currWd, "/../monolix_files/")
# Define directory where the project initialization stuff was placed
projectInitDir <- "project_init/"

model <- "froehlich_simple"
# Load the project into monolix
path <- paste0(baseDir, model, ".mlxtran")
loadProject(path)

# Save directory (in projects folder)
saveDir <- paste0(baseDir, model, "/assessment/", multistartId, "/")
print(paste0("The results will be saved in ", saveDir))
# Create directory if it doesn't exist yet
if (!dir.exists(saveDir)) {
    dir.create(saveDir, recursive = TRUE)
}

# Change number of chains to run
generalSettings <- getGeneralSettings()
generalSettings$autochains <- FALSE
generalSettings$nbchains <- 1
setGeneralSettings(generalSettings)

# Set SAEM settings
saemSettings <- getPopulationParameterEstimationSettings()
saemSettings$nbsmoothingiterations <- 1000 # 5 times the default
saemSettings$nbexploratoryiterations <- 10000 # 50 times the default
setPopulationParameterEstimationSettings(saemSettings)

# Get and change settings for the standard error task
seSettings <- getStandardErrorEstimationSettings()
seSettings$maxiterations <- 1000
setStandardErrorEstimationSettings(seSettings)

# Get and change settings for likelihood estimation task
llSettings <- getLogLikelihoodEstimationSettings()
llSettings$nbfixediterations <- 50000
setLogLikelihoodEstimationSettings(llSettings)

condDistSettings <- getConditionalDistributionSamplingSettings()
condDistSettings$nbsimulatedparameters <- 100
setConditionalDistributionSamplingSettings(condDistSettings)

# Get current Monolix scenario and set my own
scenario <- getScenario()
scenario$tasks <- c(
    populationParameterEstimation = TRUE,
    conditionalModeEstimation = FALSE,
    conditionalDistributionSampling = FALSE,
    standardErrorEstimation = TRUE,
    logLikelihoodEstimation = TRUE,
    plots = FALSE
)
# Use linearization for the standard error estimation
linearization <- TRUE
scenario$linearization <- linearization

# Set the scenario
setScenario(scenario)

# Get current population parameters
popparams <- getPopulationParameterInformation()
# Subset to contain only non-fixed and no-omega parameters
bool <- popparams$method != "FIXED"
freePars <- popparams[bool, ]

# Number of starts to perform
nRuns <- args[2]
nRuns <- as.numeric(nRuns)

# Create an empty results object
currRun <- 0
lsEstimates <- createEmptyResults(scenario)

times <- list(
    saem = NULL,
    cond = NULL,
    mode = NULL,
    se = NULL,
    ll = NULL,
    run = NULL
)

# Get "parameter bounds" (see readme for more info)
parBounds <- checkParameterBounds(paste0(projectInitDir, "parameter_bounds_", model, ".csv"))

# Run multistart
tryCatch({
    for (j in 1:nRuns) {
        currRun <- currRun + 1
        # currRun <- (i-1)*nRunsPerRound+j
        summary <- performRun(
            nRuns,
            currRun,
            freePars,
            popparams,
            scenario,
            lsEstimates,
            times,
            parBounds
            )
        lsEstimates <- summary$lsEstimates
        times <- summary$times
    }
    # Save the results every `nRunsPerRound` runs.
    saveResults(saveDir, lsEstimates, times)
}, error = function(cond) {
    cat("ERROR :", conditionMessage(cond), "\n")
}
)

# Finish execution (write final time to file)
writeLines(paste0(as.character(now()), ": The multistart optimization has finished at."))
