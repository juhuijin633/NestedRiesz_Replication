#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -t 02:00:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --array=0-4499
#SBATCH -o logs/job_nonATE2_%a.out
#SBATCH -e logs/job_nonATE2_%a.err

module load python/3.10.9-fasrc01
module load R/4.4.3-fasrc01

source ~/.bashrc
conda activate riesz

export R_LIBS_USER=/n/home11/jinhopark/R/library

cd ~/NestedRiesz/Time\ varying\ treatment

# -----------------------------------------------------------------------
# Part 2: configs 4-6  (3 configs x 3 N x 500 = 4500 jobs)
#
# Config 4 / local 0 (linear, truncated_adv, lb=0.1, ub=0.9):
#   0    - 499  : N=500
#   500  - 999  : N=1000
#   1000 - 1499 : N=2000
#
# Config 5 / local 1 (linear, truncated_adv, lb=0.3, ub=0.7):
#   1500 - 1999 : N=500
#   2000 - 2499 : N=1000
#   2500 - 2999 : N=2000
#
# Config 6 / local 2 (linear, logistic, no bounds):
#   3000 - 3499 : N=500
#   3500 - 3999 : N=1000
#   4000 - 4499 : N=2000
# -----------------------------------------------------------------------

N_VALUES=(500 1000 2000)
TMAX=500

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / (${#N_VALUES[@]} * TMAX) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#N_VALUES[@]} * TMAX) ))
N_IDX=$(( REMAINDER / TMAX ))
ITER=$(( REMAINDER % TMAX ))

N_VAL=${N_VALUES[$N_IDX]}
PYTHON=/n/home11/jinhopark/.conda/envs/riesz/bin/python

if [ $CONFIG_IDX -eq 0 ]; then
    $PYTHON run_sim.py $ITER $N_VAL truncated_adv 0.1 0.9
elif [ $CONFIG_IDX -eq 1 ]; then
    $PYTHON run_sim.py $ITER $N_VAL truncated_adv 0.3 0.7
else
    $PYTHON run_sim.py $ITER $N_VAL logistic
fi
