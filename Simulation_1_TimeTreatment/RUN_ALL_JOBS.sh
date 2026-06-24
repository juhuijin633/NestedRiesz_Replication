#!/bin/bash
# Submit E[Y(1,1)] and ATE simulation arrays, then collect when both finish.
#
# Usage (from Simulation_1_TimeTreatment/ on the cluster):
#   bash RUN_ALL_JOBS.sh
#
# Optional: pass extra sbatch flags, e.g.  bash RUN_ALL_JOBS.sh --partition=shared

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$PROJECT_ROOT/code"
mkdir -p "$PROJECT_ROOT/results/logs"

cd "$CODE_DIR"

PSI11_JOB_ID=$(sbatch "$@" run_simulations.sbatch | awk '{print $NF}')
echo "Submitted E[Y(1,1)] array: job $PSI11_JOB_ID  (tasks 0-5999)"

ATE_JOB_ID=$(sbatch "$@" run_simulations_ate.sbatch | awk '{print $NF}')
echo "Submitted ATE array: job $ATE_JOB_ID  (tasks 0-5999)"

COLLECT_JOB_ID=$(
  sbatch --dependency=afterok:"$PSI11_JOB_ID":"$ATE_JOB_ID" collect_results.sbatch | awk '{print $NF}'
)
echo "Submitted collect job: $COLLECT_JOB_ID  (runs after both arrays succeed)"
echo ""
echo "When complete, see:"
echo "  $PROJECT_ROOT/results/summary.csv"
echo "  $PROJECT_ROOT/results/table_psi11.csv"
echo "  $PROJECT_ROOT/results/table_ate.csv"
