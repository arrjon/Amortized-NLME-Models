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
        convergence <- read.csv(
            str_glue("{resultsPath}/convergence_indicators.csv")
        )

        # Load likelihoods and sort convergence data frame by likelihoods
        likelihoods <- read.csv(str_glue("{resultsPath}/log_likelihood.csv"))
        orderedLl <- likelihoods[order(likelihoods$OFV), ]
        orderedLl$rel_OFV <- orderedLl$OFV - orderedLl$OFV[1]
        # Sort by "orderedLl"
        convergence <- convergence[match(orderedLl$run, convergence$run), ]

        # Select only up to 100 rows (sometimes there are less)
        convergence <- convergence[1:min(100, nrow(convergence)), ]

        # Calculate number of iterations per run
        # (sum up exploratory and smoothing iterations)
        convergence$totalIterations <- convergence$exploratoryNumber + convergence$smoothingNumber
        # Calculate the total number of iterations
        # (assuming 1 simulation per cell per iteration)
        convergence$totalSimulations <- convergence$totalIterations * cellNumber

        # Calculate some statistics
        averageExploratoryIterations <- sum(convergence$exploratoryNumber)/nrow(convergence)
        averageSmoothingIterations <- sum(convergence$smoothingNumber)/nrow(convergence)
        averageTotalIterations <- sum(convergence$totalIterations)/nrow(convergence)
        iterationsOverAllRuns <- sum(convergence$totalIterations)
        simulationOverAllRuns <- sum(convergence$totalSimulations)

        writeLines(
            str_glue(
                "Model: {model}\n",
                "Number of runs: {nrow(convergence)}\n",
                "Sum of all iterations: {iterationsOverAllRuns}\n",
                "Sum of all simulations: {simulationOverAllRuns}\n",
                "Average number of exploratory iterations: {averageExploratoryIterations}\n",
                "Average number of smoothing iterations: {averageSmoothingIterations}\n",
                "Average number of total iterations: {averageTotalIterations}\n\n"
            )
        )
        totalIterations <- totalIterations + iterationsOverAllRuns
        totalSimulations <- totalSimulations + simulationOverAllRuns
    }
    writeLines("\n")
    writeLines("Total number of simulations and iterations:")
    writeLines(
        str_glue(
            "Model: {projectName}\n",
            "Total number of iterations: {totalIterations}\n",
            "Total number of simulations: {totalSimulations}\n"
        )
    )
    writeLines("\n\n\n")
}