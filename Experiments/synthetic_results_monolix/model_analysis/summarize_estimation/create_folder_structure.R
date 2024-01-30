# The working directory should be set to the folder of this script
renv::autoload()

# Define paths
relativeBase <- "../.." # base of git repository

# Tidyverse
library(stringr)

# Monolix projects
# Read projects from `available_projects_and_models.txt`
models <- readLines("../available_projects_and_models.txt")
writeLines("This is all available models:")
writeLines(
  str_glue("{as.character(c(1:length(models)))}: {models}")
)
# Wait for user to check models and press enter
readline("Press enter to continue.")
for (i in 1:length(models)) {
  model <- models[i]
  # Define paths
  projectBase <- str_glue("{relativeBase}/projects/{model}")
  analysisBase <- str_glue("{relativeBase}/model_analysis/projects/{model}")

  # Create analysis directory if it doesn't exist
  if (!dir.exists(str_glue("{analysisBase}")) && dir.exists(str_glue("{projectBase}"))) {
    dir.create(str_glue("{analysisBase}"), recursive = TRUE)
    # dir.create(str_glue("{analysisBase}/plots"))
    dir.create(str_glue("{analysisBase}/other"))
  } else {
    writeLines("The analysis directory already exists or the Monolix project does not.")
  }
}