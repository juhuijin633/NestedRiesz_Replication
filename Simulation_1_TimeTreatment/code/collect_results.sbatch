#!/bin/bash
#SBATCH --job-name=tvt_collect
#SBATCH -o ../results/logs/collect.out
#SBATCH -e ../results/logs/collect.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=serial_requeue

set -eo pipefail

module load python/3.10.9-fasrc01
eval "$(conda shell.bash hook)"
conda activate riesz

cd "$SLURM_SUBMIT_DIR"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "Runing from $(pwd)"
ls -la

python 2_collect_results.py --force
