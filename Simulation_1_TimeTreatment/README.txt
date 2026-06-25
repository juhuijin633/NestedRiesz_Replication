Simulation_1_TimeTreatment
============================

Monte Carlo simulation for time-varying treatment estimators (E[Y(1,1)] only).

Layout
------
  RUN_ALL_JOBS.sh              cluster entry point (submit from project root)
  code/
    RUN.py                     local entry point
    run_simulations.sbatch     SLURM array 0-5999 → E[Y(1,1)] replications
    collect_results.sbatch     SLURM collect → 2_collect_results.py
    smoke_test.py              one replication pre-flight (cluster)
    1_run_simulation.py        one (config, N, iteration) job per invocation
    2_collect_results.py       aggregate .pt → CSV tables
    utils/generate_dgp.py, hyperparams.py, dynamicRiesz*, dynamicRieszBradic.py, Bradic.R
  results/
    intermediate/N_{500,1000,2000}/{config_id}/result_{t}.pt
    summary.csv, table_psi11.csv

Cluster
-------
  cd Simulation_1_TimeTreatment/code
  module load python/3.10.9-fasrc01
  module load R/4.4.3-fasrc01
  conda activate riesz

  Pre-flight (one replication):
    python smoke_test.py

  Submit full array:
    sbatch run_simulations.sbatch

  Each array task runs one MC replication (matches time_varying_treatment/submit.sh):
    4 DGPs × 3 N × 500 iterations = 6000 tasks.

Collect
-------
  python 2_collect_results.py --force

Seeds: torch.manual_seed(123 + t) before each replication (matches original scripts).

Methods: Oracle, Bradic, LASSO-LASSO, RF-RF, Net-Net.

Configs (array order):
  0  linear + truncated_logistic
  1  nonlinear + truncated_adv
  2  linear + truncated_adv
  3  linear + logistic
