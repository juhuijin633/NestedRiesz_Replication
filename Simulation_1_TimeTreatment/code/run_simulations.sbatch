#!/bin/bash
#SBATCH --job-name=tvt_sim
#SBATCH --array=0-5999
#SBATCH -o ../results/logs/out_psi11_%a.out
#SBATCH -e ../results/logs/err_psi11_%a.err
#SBATCH -t 02:00:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --partition=serial_requeue

# One array task = one MC replication for E[Y(1,1)].
# Index mapping matches time_varying_treatment/submit.sh:
#   4 DGPs × 3 N × 500 iterations = 6000 jobs
#
#   DGP 0 (linear,    truncated_logistic): tasks 0–1499
#   DGP 1 (nonlinear, truncated_adv):      tasks 1500–2999
#   DGP 2 (linear,    truncated_adv):      tasks 3000–4499
#   DGP 3 (linear,    logistic):          tasks 4500–5999

set -euo pipefail

module load python/3.10.9-fasrc01
module load R/4.4.3-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz

export R_LIBS_USER="${R_LIBS_USER:-/n/home11/jinhopark/R/library}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python 1_run_simulation.py
