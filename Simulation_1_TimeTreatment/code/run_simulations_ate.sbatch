#!/bin/bash
#SBATCH --job-name=tvt_ate
#SBATCH --array=0-5999
#SBATCH -o ../results/logs/out_ate_%a.out
#SBATCH -e ../results/logs/err_ate_%a.err
#SBATCH -t 04:00:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --partition=serial_requeue

# One array task = one MC replication for ATE = E[Y(1,1)] - E[Y(0,0)].
# Same index mapping as run_simulations.sbatch (matches submit_ate.sh).

set -euo pipefail

module load python/3.10.9-fasrc01
module load R/4.4.3-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz

export R_LIBS_USER="${R_LIBS_USER:-/n/home11/jinhopark/R/library}"
export SIM_TARGET=ate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python 1_run_simulation.py
