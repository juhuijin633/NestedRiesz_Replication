#!/bin/bash
# Submit E[Y(1,1)] simulation array, then collect when finished.
#
# Run on a LOGIN NODE (boslogin), NOT via sbatch:
#   bash RUN_ALL_JOBS.sh
#
# Optional flags forwarded to BOTH sbatch files:
#   bash RUN_ALL_JOBS.sh --partition=shared

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run on a login node with bash, not as a batch job." >&2
  echo "  cd Simulation_1_TimeTreatment && bash RUN_ALL_JOBS.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
mkdir -p "$PROJECT_ROOT/results/logs"

cd "$PROJECT_ROOT/code"

SBATCH_EXTRA=("$@")

PSI11_JOB_ID=$(sbatch "${SBATCH_EXTRA[@]}" run_simulations.sbatch | awk '{print $NF}')
echo "Submitted E[Y(1,1)] array: job $PSI11_JOB_ID  (tasks 0-5999)"

COLLECT_JOB_ID=$(
  sbatch "${SBATCH_EXTRA[@]}" --dependency=afterok:"$PSI11_JOB_ID" collect_results.sbatch | awk '{print $NF}'
)
echo "Submitted collect: job $COLLECT_JOB_ID  (after $PSI11_JOB_ID completes)"

echo
echo "Results will appear under:"
echo "  $PROJECT_ROOT/results/summary.csv"
