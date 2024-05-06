# Load R environment
renv::autoload()

# The script calculates the number of iterations and simulations for each run

# Libraries
library(ggplot2)
library(reshape2)
library(stringr)

# Define paths
relativeBase <- "../.."

# Which project to work on (will be used to define the project path later)
projectNames <- c(
    "froehlich_simple",
    "froehlich_detailed"
)
smallOrLarge <- c(
    "simple",
    "detailed"
)

# Cell numbers
cellNumbers <- c(
    50,
    100,
    200,
    500,
    1000,
    5000,
    10000
)

for (j in seq_along(projectNames)) {
    # Select project name
    projectName <- projectNames[j]
    # Keep tally over all runs for all cell numbers for each model
    totalIterations <- 0
    totalSimulations <- 0
    for (i in seq_along(cellNumbers)) {
        cellNumber <- cellNumbers[i]
        model <- str_glue("froehlich/{smallOrLarge[j]}/synthetic_{cellNumbers[i]}")
        # Model base path
        projectBase <- str_glue("{relativeBase}/projects/{model}")
        analysisBase <- str_glue("{relativeBase}/model_analysis/projects/{model}")
        # Parameter estimation results path
        resultsPath <- str_glue(
            "{projectBase}/monolix_files/{projectName}/assessment"
        )

        # Load convergence information
        walltimes <- read.csv(
            str_glue("{resultsPath}/runtimes.csv")
        )

        # Load likelihoods and sort convergence data frame by likelihoods
        likelihoods <- read.csv(str_glue("{resultsPath}/log_likelihood.csv"))
        orderedLl <- likelihoods[order(likelihoods$OFV), ]
        orderedLl$rel_OFV <- orderedLl$OFV - orderedLl$OFV[1]
        # Sort by "orderedLl"
        walltimes <- walltimes[match(orderedLl$run, walltimes$run), ]

        # Calculate average walltimes
        averageSaem <- sum(walltimes$saem)/nrow(walltimes)
        averageSe <- sum(walltimes$se)/nrow(walltimes)
        averageLl <- sum(walltimes$ll)/nrow(walltimes)

        # Summarize
        writeLines(
            str_glue(
                "Model: {model}\n",
                "Number of runs: {nrow(walltimes)}\n",
                "Average walltime SAEM: {averageSaem/60} min\n",
                "Average walltime standard errors: {averageSe/60} min\n",
                "Average walltime likelihood: {averageLl/60} min\n",
                "Total walltime all runs (SAEM + SE + LL): {(sum(walltimes$saem) + sum(walltimes$se) + sum(walltimes$ll))/60} min\n\n",
            )
        )

    }
}