#!/bin/bash
# Submit all 9 simulation array jobs, then a collect job when they finish.
#
# Run from Simulation_2_DiD/ on a LOGIN NODE (boslogin), NOT via sbatch:
#
#   bash RUN_ALL_JOBS.sh
#
# Do NOT run:  sbatch RUN_ALL_JOBS.sh
#
# Extra sbatch flags apply to BOTH the simulation array and the collect job, e.g.:
#   bash RUN_ALL_JOBS.sh --partition=shared
#   bash RUN_ALL_JOBS.sh --partition=serial_requeue -t 72:00:00
#
# FASRC notes:
#   - shared / serial_requeue: max ~3 days → default -t 72:00:00 in run_simulations.sbatch
#   - intermediate: requires >3 days → use -t 100:00:00 (or 4-00:00:00) with --partition=intermediate

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run this script on a login node with bash, not as a batch job." >&2
  echo "  cd Simulation_2_DiD && bash RUN_ALL_JOBS.sh" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$PROJECT_ROOT/code"
mkdir -p "$PROJECT_ROOT/results/logs"

cd "$CODE_DIR"

SBATCH_EXTRA=("$@")

SIM_JOB_ID=$(sbatch "${SBATCH_EXTRA[@]}" run_simulations.sbatch | awk '{print $NF}')
echo "Submitted simulation array: job $SIM_JOB_ID  (tasks 0-8)"

COLLECT_JOB_ID=$(
  sbatch "${SBATCH_EXTRA[@]}" --dependency=afterok:"$SIM_JOB_ID" collect_results.sbatch | awk '{print $NF}'
)
echo "Submitted collect job: $COLLECT_JOB_ID  (runs after all array tasks succeed)"
echo ""
echo "When complete, see:"
echo "  $PROJECT_ROOT/results/summary.csv"
echo "  $PROJECT_ROOT/results/table_logistic.csv"
echo "  $PROJECT_ROOT/results/table_truncated_logistic.csv"
echo "  $PROJECT_ROOT/results/table_truncated_step.csv"
