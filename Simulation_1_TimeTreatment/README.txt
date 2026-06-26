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
    utils/simulation_config.py design constants + paper method labels
    utils/generate_dgp.py, hyperparams.py, dynamicRiesz*, Bradic.R
  results/
    intermediate/N_{500,1000,2000}/{config_id}/result_{t}.pt
    summary.csv

Cluster (FASRC)
---------------
  cd Simulation_1_TimeTreatment
  bash RUN_ALL_JOBS.sh

  # or with explicit partition:
  bash RUN_ALL_JOBS.sh --partition=shared

  Do NOT: sbatch RUN_ALL_JOBS.sh

  Pre-flight (one replication):
    cd code && python smoke_test.py

Collect
-------
  python 2_collect_results.py --force

Seeds: torch.manual_seed(123 + t) before each replication (matches run_sim.py).

Method labels in summary.csv (paper tables):
  Oracle, Manual-Lasso, Auto-Lasso, Auto-RF, Auto-NN
  (upstream scripts used: Oracle, Bradic, LASSO-LASSO, RF-RF, Net-Net)

Coverage / CI: theta ± 1.96 * (pred_sig / sqrt(N)) for all methods — matches
collect_sim_results.py (pred_sig is influence-function SD; Oracle uses SD of psi_i).

Configs (array order, 6000 tasks = 4 × 3 N × 500 reps):
  0  linear + truncated_logistic [0.1, 0.9]   ← Table F.3 Panel A
  1  nonlinear + truncated_adv [0.1, 0.9]
  2  linear + truncated_adv [0.1, 0.9]
  3  linear + logistic

Table F.3 Panel B ([0.3, 0.7] truncation) is in upstream submit_nonATE.sh;
not in the default 6000-job array (add a config in simulation_config.py to extend).
