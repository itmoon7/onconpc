#!/bin/bash

# Activate Conda environment
conda activate onconpc_conda_env

# Install R packages
R --vanilla <<EOF
options(repos = c(CRAN = "https://cloud.r-project.org/"))  # Set a CRAN mirror
install.packages("reshape2")
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
BiocManager::install("BSgenome")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
BiocManager::install("GenomeInfoDb")
BiocManager::install("deconstructSigs")
EOF

