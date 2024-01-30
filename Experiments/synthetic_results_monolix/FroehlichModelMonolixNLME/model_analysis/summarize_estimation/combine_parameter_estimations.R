# Set working directory
renv::autoload()

# Tidyverse
library(stringr)
library(reshape2)
library(dplyr)

# Monolix projects
# Read projects from `available_projects_and_models.txt`
models <- readLines("../available_projects_and_models.txt")
writeLines("This is all available models:")
writeLines(
  str_glue("{as.character(c(1:length(models)))}: {models}")
)
# Let user select model
model <- readline("Which model would you like to summarize? ")
model <- models[as.numeric(model)]

# Define paths
relativeBase <- "../.." # base of git repository
projectBase <- str_glue("{relativeBase}/projects/{model}")
analysisBase <- str_glue("{relativeBase}/model_analysis/projects/{model}")

# Create analysis directory if it doesn't exist
if (!dir.exists(str_glue("{analysisBase}"))) {
  dir.create(str_glue("{analysisBase}"))
  dir.create(str_glue("{analysisBase}/plots"))
  dir.create(str_glue("{analysisBase}/other"))
}

# Select monolix project name
# Can also be read from file
projectNames <- readLines("../monolix_project_names.txt")
writeLines("This is all available projects:")
writeLines(
  str_glue("{as.character(c(1:length(projectNames)))}: {projectNames}")
)
# Let user select project
projectName <- readline("What is the name of the project (precedes `.mlxtran`)?: ")
projectName <- projectNames[as.numeric(projectName)]

# Results
resultDir <- str_glue("{projectBase}/monolix_files/{projectName}/assessment")

# List all folders in the `resultDir`
allMultistarts <- list.dirs(resultDir, full.names = FALSE)
allMultistarts <- allMultistarts[2:length(allMultistarts)]
writeLines("This is all the multistarts that were run:")
writeLines(
  str_glue("{as.character(c(1:length(allMultistarts)))}: {allMultistarts}")
)
writeLines(str_glue("The results will be summarized in `{resultDir}`."))

# Ask user whether this is correct
correct <- readline("Is this correct? (y/N) ")
if (correct != "y") {
  stop("Please correct the `resultDir` and `allMultistarts` variables.")
}

# Gather all result objects
# Start with first multistart result:
multistart <- allMultistarts[1]
population_parameters <- read.csv(
  str_glue("{resultDir}/{multistart}/population_parameters.csv"),
  stringsAsFactors = FALSE
)
standard_errors <- read.csv(
  str_glue("{resultDir}/{multistart}/standard_errors.csv"),
  stringsAsFactors = FALSE
)
conv_indicators <- read.csv(
  str_glue("{resultDir}/{multistart}/convergence_indicators.csv"),
  stringsAsFactors = FALSE
)
individual_parameters <- read.csv(
  str_glue("{resultDir}/{multistart}/individual_parameters.csv"),
  stringsAsFactors = FALSE
)
likelihoods <- read.csv(
  str_glue("{resultDir}/{multistart}/log_likelihood.csv"),
  stringsAsFactors = FALSE
)
runtimes <- read.csv(
  str_glue("{resultDir}/{multistart}/runtimes.csv"),
  stringsAsFactors = FALSE
)

# Get pair of multistart and best objective function value
bestObjective <- c(multistart, min(likelihoods$OFV))
bestRun <- likelihoods[likelihoods$OFV == bestObjective[2], "run"]

# simulated_parameters <- read.csv(
#   str_glue("{resultDir}/{multistart}/simulated_parameters.csv"),
#   stringsAsFactors = FALSE
# )

if (length(allMultistarts) > 1) {
  for (multistart in allMultistarts[2:length(allMultistarts)]) {
    # Population parameters
    population_parameters_2 <- read.csv(
      str_glue("{resultDir}/{multistart}/population_parameters.csv"),
      stringsAsFactors = FALSE
    )
    # Merge columns
    population_parameters <- merge(
      population_parameters,
      population_parameters_2,
      by = "X",
      all = TRUE
    )

    # Standard errors
    standard_errors_2 <- read.csv(
      str_glue("{resultDir}/{multistart}/standard_errors.csv"),
      stringsAsFactors = FALSE
    )
    # Merge columns
    standard_errors <- merge(
      standard_errors,
      standard_errors_2,
      by = "X",
      all = TRUE
    )

    # Convergence indicators
    conv_indicators_2 <- read.csv(
      str_glue("{resultDir}/{multistart}/convergence_indicators.csv"),
      stringsAsFactors = FALSE
    )
    # Append rows
    conv_indicators <- rbind(conv_indicators, conv_indicators_2)

    # Individual parameters
    individual_parameters_2 <- read.csv(
      str_glue("{resultDir}/{multistart}/individual_parameters.csv"),
      stringsAsFactors = FALSE
    )
    # Append rows
    individual_parameters <- rbind(
      individual_parameters,
      individual_parameters_2
    )

    # Runtimes
    runtimes_2 <- read.csv(
      str_glue("{resultDir}/{multistart}/runtimes.csv"),
      stringsAsFactors = FALSE
    )
    # Append rows
    runtimes <- rbind(runtimes, runtimes_2)

    # Likelihoods
    likelihoods_2 <- read.csv(
      str_glue("{resultDir}/{multistart}/log_likelihood.csv"),
      stringsAsFactors = FALSE
    )
    # Append rows
    likelihoods <- rbind(likelihoods, likelihoods_2)

    # Check if better objective function value was achieved in this multistart
    if (min(likelihoods_2$OFV) < bestObjective[2]) {
      bestObjective <- c(multistart, min(likelihoods_2$OFV))
      bestRun <- likelihoods_2[likelihoods_2$OFV == bestObjective[[2]], "run"]
    }
  }
}
# Check if simulated_parameters exists
if (file.exists(str_glue("{resultDir}/{bestObjective[1]}/simulated_parameters.csv"))) {
  simulated_parameters <- read.csv(
    str_glue("{resultDir}/{bestObjective[1]}/simulated_parameters.csv"),
    stringsAsFactors = FALSE
  )
  simulated_parameters <- simulated_parameters[
    simulated_parameters$run == bestRun,
  ]
} else {
  simulated_parameters <- NULL
}

# # Simulated parameters should be treated separately, because the file can
# # become super large
# # I'll probably just save the best start or the top 10/20 starts
# # Simulated parameters
# simulated_parameters <- read.csv(
#   str_glue("{resultDir}/{multistart}/simulated_parameters.csv"),
#   stringsAsFactors = FALSE
# )
# simulated_parameters$multistart <- multistart
# allResults$simulated_parameters <- c(
#   allResults$simulated_parameters,
#   list(simulated_parameters)
# )

# Save new dataframes with summarized results
write.csv(
  population_parameters,
  str_glue("{resultDir}/population_parameters.csv"),
  row.names = FALSE
)
write.csv(
  standard_errors,
  str_glue("{resultDir}/standard_errors.csv"),
  row.names = FALSE
)
write.csv(
  conv_indicators,
  str_glue("{resultDir}/convergence_indicators.csv"),
  row.names = FALSE
)
write.csv(
  individual_parameters,
  str_glue("{resultDir}/individual_parameters.csv"),
  row.names = FALSE
)
write.csv(
  likelihoods,
  str_glue("{resultDir}/log_likelihood.csv"),
  row.names = FALSE
)
write.csv(
  runtimes,
  str_glue("{resultDir}/runtimes.csv"),
  row.names = FALSE
)

if (file.exists(str_glue("{resultDir}/{bestObjective[1]}/simulated_parameters.csv"))) {
  write.csv(
    simulated_parameters,
    str_glue("{resultDir}/simulated_parameters.csv"),
    row.names = FALSE
  )
}

# Ask user whether they want to remove the multistart directories
removeMultistarts <- readline(
  "Do you want to remove the multistart directories? (y/N) "
)
if (removeMultistarts == "y") {
  for (multistart in allMultistarts) {
    dir.remove(str_glue("{resultDir}/{multistart}"))
  }
}