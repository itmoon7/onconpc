#!/bin/bash

# Check if the correct conda environment is already activated
if [ "$CONDA_DEFAULT_ENV" != "onconpc_conda_env" ]; then
    conda activate onconpc_conda_env
fi

# Install R packages
R --vanilla <<EOF
options(repos = c(CRAN = "https://cloud.r-project.org/"))  # Set a CRAN mirror

# Install BiocManager if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# Set Bioconductor version to 3.18
BiocManager::install(version = "3.18")

# Install BSgenome and related packages
BiocManager::install("BSgenome")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
BiocManager::install("GenomeInfoDb")

# Install the package directly from the URL
install.packages("https://cran.r-project.org/src/contrib/Archive/deconstructSigs/deconstructSigs_1.8.0.tar.gz", repos = NULL, type = "source")
EOF