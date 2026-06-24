Simulation_2_DiD
================

Monte Carlo simulation for DiD estimators (Tables D.6–D.8).

Layout
------
  RUN_ALL_JOBS.sh            cluster entry point (submit from project root)
  code/
    RUN.py                   local entry point
    run_simulations.sbatch   SLURM array 0-8 → 1_run_simulation.py
    collect_results.sbatch   SLURM collect → 2_collect_results.py
    1_run_simulation.py      one (N, propensity) job per invocation
    2_collect_results.py     aggregate .pt → CSV tables
    utils/generate_dgp.py, hyperparams.py, estimateDiD_*, dynamicRiesz*
  results/
    intermediate/N_{500,1000,2000}/{propensity}_*.pt
    summary.csv, table_*.csv

Cluster
-------
  cd Simulation_2_DiD
  bash RUN_ALL_JOBS.sh

Local
-----
  cd Simulation_2_DiD/code
  python RUN.py

Single setting (cluster task or local)
--------------------------------------
  python 1_run_simulation.py --N 500 --model logistic

Seeds: DGP seed=123; replication t uses torch.manual_seed(t) and seed=t.
