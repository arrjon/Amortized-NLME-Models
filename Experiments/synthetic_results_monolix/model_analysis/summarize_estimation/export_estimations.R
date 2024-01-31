# Load R environment
renv::autoload()

# Load libraries
library(dplyr)

# Read available projects and models from `available_projects_and_models.txt`
models <- readLines("../available_projects_and_models.txt")
models <- trimws(models)

# Let user choose model
cat("Choose model:\n")
for (i in seq_along(models)) {
  cat(i, ": ", models[i], "\n")
}
modelNb <- readline("Which model would you like to export? (Press Enter to continue)")
model <- models[as.integer(modelNb)]

# Select monolix project name
# Can also be read from file
projectNames <- readLines("../monolix_project_names.txt")
projectNames <- trimws(projectNames)

# cat("This is all available projects:\n")
for (i in seq_along(projectNames)) {
  cat(i, ": ", projectNames[i], "\n")
}
# Let user select project
projectIndex <- readline("What is the name of the project (precedes `.mlxtran`)?: ")
projectName <- trimws(projectNames[as.integer(projectIndex)])

# Set paths
relativeBase <- "../.."
projectFolder <- file.path(relativeBase, "projects", model, "monolix_files")
resultsFolder <- file.path(projectFolder, projectName, "assessment")
modelFolder <- "../../models/antimony"

# Folder to save plots
analysisBase <- file.path(relativeBase, "model_analysis", "projects", model)

# Load likelihood data
likelihoods <- read.csv(file.path(resultsFolder, "log_likelihood.csv"))
likelihoods <- likelihoods[order(likelihoods$OFV), ]
orderedRunIds <- likelihoods$run

# Load walltime data
walltimes <- read.csv(file.path(resultsFolder, "runtimes.csv"))
# Sort by `ordererRunIds`
walltimes$run <- factor(walltimes$run, levels = orderedRunIds)
walltimes <- walltimes[order(walltimes$run), ]

# Population parameters
populationParameters <- read.csv(file.path(resultsFolder, "population_parameters.csv"))
# Transpose and set column names
populationParameters <- t(populationParameters)
colnames(populationParameters) <- populationParameters[1, ]
populationParameters <- populationParameters[-1, ]
# Move index to column
populationParameters <- data.frame(run = rownames(populationParameters), populationParameters)
# Recreate index from 0 to nrow(populationParameters)
rownames(populationParameters) <- c(1:nrow(populationParameters))
# Replace "run" in `run` column with empty string
populationParameters$run <- gsub("run_", "", populationParameters$run)
# Sort by `ordererRunIds`
populationParameters$run <- factor(populationParameters$run, levels = orderedRunIds)
populationParameters <- populationParameters[order(populationParameters$run), ]

# Standard errors
standardErrors <- read.csv(file.path(resultsFolder, "standard_errors.csv"))
# Replace "se_run_" in column names with "run_"
colnames(standardErrors) <- gsub("se_run_", "run_", colnames(standardErrors))
# Sort columns by `ordererRunIds`
colLevels <- factor(colnames(standardErrors), levels = c("X", paste0("run_", orderedRunIds)))
standardErrors <- standardErrors[, order(colLevels)]

# Save all data frames to `analysisBase`/other
model <- tail(strsplit(model, "/")[[1]], 1)
otherFolder <- file.path(analysisBase, "other")
write.csv(likelihoods, file.path(otherFolder, paste0(model, "_likelihoods.csv")), row.names = FALSE)
write.csv(walltimes, file.path(otherFolder, paste0(model, "_walltimes.csv")), row.names = FALSE)
write.csv(populationParameters, file.path(otherFolder, paste0(model, "_population_parameters.csv")), row.names = FALSE)
write.csv(standardErrors, file.path(otherFolder, paste0(model, "_standard_errors.csv")), row.names = FALSE)

# Zip all exported files and make archive name equal to last bit of `model`
# Save archive in `analysisBase`/other
archiveName <- model
archiveName <- paste0(otherFolder, "/", archiveName, ".zip")
zipCommand <- paste("zip", archiveName, paste0(otherFolder, "/", model, "*.csv"))
system(zipCommand)
