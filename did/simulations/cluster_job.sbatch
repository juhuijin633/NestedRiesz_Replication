#!/bin/bash 

#SBATCH --job-name=advriesz_gcv_test
#SBATCH --array=0-8
#SBATCH -o logs/out_%a.out
#SBATCH -e logs/err_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinhopark@college.harvard.edu
#SBATCH -t 4-00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition=intermediate

module load python/3.10.13-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz
python simulation_cluster.py