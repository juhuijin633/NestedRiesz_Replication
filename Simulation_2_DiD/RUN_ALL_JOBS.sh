#!/bin/bash
# Submit all 9 simulation array jobs, then a collect job when they finish.
#
# Usage (from Simulation_2_DiD/ on the cluster):
#   bash RUN_ALL_JOBS.sh
#
# Optional: pass extra sbatch flags, e.g.  bash RUN_ALL_JOBS.sh --partition=shared

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$PROJECT_ROOT/code"
mkdir -p "$PROJECT_ROOT/results/logs"

cd "$CODE_DIR"

SIM_JOB_ID=$(sbatch "$@" run_simulations.sbatch | awk '{print $NF}')
echo "Submitted simulation array: job $SIM_JOB_ID  (tasks 0-8)"

COLLECT_JOB_ID=$(sbatch --dependency=afterok:"$SIM_JOB_ID" collect_results.sbatch | awk '{print $NF}')
echo "Submitted collect job: $COLLECT_JOB_ID  (runs after all array tasks succeed)"
echo ""
echo "When complete, see:"
echo "  $PROJECT_ROOT/results/summary.csv"
echo "  $PROJECT_ROOT/results/table_logistic.csv"
echo "  $PROJECT_ROOT/results/table_truncated_logistic.csv"
echo "  $PROJECT_ROOT/results/table_truncated_step.csv"
