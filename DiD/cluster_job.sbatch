#!/bin/bash 

#SBATCH --job-name=advriesz_gcv_test
#SBATCH -o out.out
#SBATCH -e err.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvinlob@fas.harvard.edu
#SBATCH -t 24-00:00  
#SBATCH --cpus-per-task=1 
#SBATCH --mem=100GB
#SBATCH --partition=intermediate

module load python/3.10.16-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate NestedRiesz
python simulation_cluster.py