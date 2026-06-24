#!/bin/bash
#SBATCH --job-name=tvt_collect
#SBATCH -o ../results/logs/collect.out
#SBATCH -e ../results/logs/collect.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=serial_requeue

set -euo pipefail

module load python/3.10.9-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python 2_collect_results.py --force
