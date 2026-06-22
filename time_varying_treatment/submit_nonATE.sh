#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -t 02:00:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --array=0-10499
#SBATCH -o logs/job_nonATE_%a.out
#SBATCH -e logs/job_nonATE_%a.err

module load python/3.10.9-fasrc01
module load R/4.4.3-fasrc01

source ~/.bashrc
conda activate riesz

export R_LIBS_USER=/n/home11/jinhopark/R/library

cd ~/NestedRiesz/Time\ varying\ treatment

# -----------------------------------------------------------------------
# Index mapping:
# 7 configs x 3 N values x 500 iterations = 10500 total jobs
#
# Config 0 (linear,    truncated_logistic, lb=0.1, ub=0.9):
#   0    - 499  : N=500
#   500  - 999  : N=1000
#   1000 - 1499 : N=2000
#
# Config 1 (linear,    truncated_logistic, lb=0.3, ub=0.7):
#   1500 - 1999 : N=500
#   2000 - 2499 : N=1000
#   2500 - 2999 : N=2000
#
# Config 2 (nonlinear, truncated_adv,      lb=0.1, ub=0.9):
#   3000 - 3499 : N=500
#   3500 - 3999 : N=1000
#   4000 - 4499 : N=2000
#
# Config 3 (nonlinear, truncated_adv,      lb=0.3, ub=0.7):
#   4500 - 4999 : N=500
#   5000 - 5499 : N=1000
#   5500 - 5999 : N=2000
#
# Config 4 (linear,    truncated_adv,      lb=0.1, ub=0.9):
#   6000 - 6499 : N=500
#   6500 - 6999 : N=1000
#   7000 - 7499 : N=2000
#
# Config 5 (linear,    truncated_adv,      lb=0.3, ub=0.7):
#   7500 - 7999 : N=500
#   8000 - 8499 : N=1000
#   8500 - 8999 : N=2000
#
# Config 6 (linear,    logistic,           no bounds):
#   9000 - 9499 : N=500
#   9500 - 9999 : N=1000
#   10000-10499 : N=2000
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
    $PYTHON run_sim.py $ITER $N_VAL truncated_logistic 0.1 0.9
elif [ $CONFIG_IDX -eq 1 ]; then
    $PYTHON run_sim.py $ITER $N_VAL truncated_logistic 0.3 0.7
elif [ $CONFIG_IDX -eq 2 ]; then
    $PYTHON run_sim_nonlinear.py $ITER $N_VAL truncated_adv 0.1 0.9
elif [ $CONFIG_IDX -eq 3 ]; then
    $PYTHON run_sim_nonlinear.py $ITER $N_VAL truncated_adv 0.3 0.7
elif [ $CONFIG_IDX -eq 4 ]; then
    $PYTHON run_sim.py $ITER $N_VAL truncated_adv 0.1 0.9
elif [ $CONFIG_IDX -eq 5 ]; then
    $PYTHON run_sim.py $ITER $N_VAL truncated_adv 0.3 0.7
else
    $PYTHON run_sim.py $ITER $N_VAL logistic
fi
