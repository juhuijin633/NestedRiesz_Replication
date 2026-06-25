#!/bin/bash
#SBATCH --job-name=tvt_smoke
#SBATCH --partition=serial_requeue
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=../results/logs/smoke_test_%j.out
#SBATCH --error=../results/logs/smoke_test_%j.err

set -eo pipefail

module load python/3.10.9-fasrc01
module load R/4.4.3-fasrc01

eval "$(conda shell.bash hook)"
conda activate riesz

export R_LIBS_USER="$HOME/R/library"
mkdir -p "$R_LIBS_USER"

cd "$SLURM_SUBMIT_DIR"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "Running from $(pwd)"
echo "Python path: $(which python)"
python --version
ls -la

python smoke_test.py
