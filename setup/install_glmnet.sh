#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -o logs/install_glmnet.out
#SBATCH -e logs/install_glmnet.err

module load R/4.4.3-fasrc01

R_LIB="$HOME/R/library"
mkdir -p "$R_LIB"

R_LIBS_USER="$R_LIB" R --no-save -e "
.libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths()))
install.packages('glmnet', lib=Sys.getenv('R_LIBS_USER'),
                 repos='https://cloud.r-project.org',
                 INSTALL_opts='--no-multiarch')
cat('glmnet installed:', is.element('glmnet', installed.packages()[,1]), '\n')
cat('R library path:', Sys.getenv('R_LIBS_USER'), '\n')
"
