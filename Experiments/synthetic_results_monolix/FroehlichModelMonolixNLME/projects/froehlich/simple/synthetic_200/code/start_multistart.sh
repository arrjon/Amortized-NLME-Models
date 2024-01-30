#!/bin/bash
# Activate Monolix license (if necessary)
#/Monolix/lib/licenseActivate -k 
# The current time will be the unique ID for the number of multistarts in `multistart_monolix.R`
currentTime=$(date +%s_%4N)
assessmentPath=$(find ../ -type d -name assessment)
mkdir -pv "$assessmentPath/$currentTime"
# Run multistart script
# The first argument is the unique multistart ID and the second the number of starts that should be performed
Rscript ./multistart_monolix.R $(echo $currentTime) 1
