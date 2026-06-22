#!/bin/bash

#SBATCH --job-name=did_ols
#SBATCH -o logs/ols_out.out
#SBATCH -e logs/ols_err.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinhopark@college.harvard.edu
#SBATCH -t 0-04:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --partition=shared

module load python/3.10.13-fasrc01
module load Anaconda3/2022.05
source activate riesz
cd /n/home11/jinhopark/NestedRiesz_DiD/DiD
python simulation_cluster_only_OLS.py
