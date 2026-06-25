#!/bin/bash
# Submit E[Y(1,1)] simulation array, then collect when finished.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
mkdir -p "$PROJECT_ROOT/results/logs"

cd "$PROJECT_ROOT/code"

PSI11_JOB_ID=$(sbatch "$@" run_simulations.sbatch | awk '{print $NF}')
echo "Submitted E[Y(1,1)] array: job $PSI11_JOB_ID  (tasks 0-5999)"

COLLECT_JOB_ID=$(
  sbatch --dependency=afterok:"$PSI11_JOB_ID" collect_results.sbatch | awk '{print $NF}'
)
echo "Submitted collect: job $COLLECT_JOB_ID  (after $PSI11_JOB_ID completes)"

echo
echo "Results will appear under:"
echo "  $PROJECT_ROOT/results/summary.csv"
echo "  $PROJECT_ROOT/results/table_psi11.csv"
