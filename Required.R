options(warn = -1)

# Changing default 256MB JVM memory limit before loading Java packages:
options(java.parameters = "-Xmx16g")

suppressWarnings(suppressMessages({
    packages = c("backports",
                 "devtools",
                 "reticulate",
                 "keras",
                 "tensorflow",
                 "doParallel",
                 "RODBC",
                 "Hmisc",
                 "caret",
                 "lubridate",
                 "clusterSim",
                 "fastcluster",
                 "ClusterR",
                 "kohonen",
                 "RWeka",
                 "dtwclust",
                 "speccalt",
                 "Rssa",
                 "EMD",
                 "praznik",
                 "e1071",
                 "metaheuristicOpt")
    for (package in packages) {
        if (!require(package, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)) {
            install.packages(package)

            if (package == "lubridate") {
                devtools::install_github("tidyverse/lubridate")
            } else if (package == "tensorflow") {
                devtools::install_github("rstudio/tensorflow", force = TRUE)
                tensorflow::install_tensorflow()
            } else if (package == "keras") {
                devtools::install_github("rstudio/keras", force = TRUE)
                keras::install_keras()
            }

            library(package, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
        }
    }
}))

# In case of "ModuleNotFoundError: No module named 'rpytools'" error copy rpytools folder
# from <user>\Documents\R\win-library\<version>\reticulate\python
#   to <user>\AppData\Local\r-miniconda\envs\r-reticulate.
# Also, try reinstating tensorflow and keras using the commands above.

# Workaround for broken VMD package that cannot be installed from CRAN:
source("https://raw.githubusercontent.com/igormanojlovic/TimeSeriesML/main/VMD.R")

# Disabling TensorFlow info messages:
py_run_string("import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';")