#!/bin/bash
#SBATCH --job-name=did_sim
#SBATCH --array=0-8
#SBATCH -o ../results/logs/out_%a.out
#SBATCH -e ../results/logs/err_%a.err
#SBATCH -t 4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition=intermediate

# One array task = one (N, propensity) job with 100 Monte Carlo replications inside.
#
#   task_id // 3  →  N ∈ {500, 1000, 2000}
#   task_id % 3  →  propensity ∈ {logistic, truncated_logistic, truncated_step}
#
# Matches the original did/simulations/cluster_job.sbatch + simulation_cluster.py.

set -euo pipefail

module load python/3.10.13-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python 1_run_simulation.py
