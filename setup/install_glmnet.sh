#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -o logs/install_glmnet.out
#SBATCH -e logs/install_glmnet.err

module load R/4.4.3-fasrc01

R_LIBS_USER=/n/home11/jinhopark/R/library R --no-save -e "
.libPaths(c('/n/home11/jinhopark/R/library', .libPaths()))
install.packages('glmnet', lib='/n/home11/jinhopark/R/library',
                 repos='https://cloud.r-project.org',
                 INSTALL_opts='--no-multiarch')
cat('glmnet installed:', is.element('glmnet', installed.packages()[,1]), '\n')
"
