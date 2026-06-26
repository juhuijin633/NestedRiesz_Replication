# NestedRiesz: Replication Package

This repository contains all code needed to reproduce the simulation studies and empirical applications in the paper. The replication is organized into four exercises: two Monte Carlo simulation studies and two empirical applications. The simulations take too long to run on a single machine and were run on a SLURM-based HPC cluster; the applications are self-contained Jupyter notebooks that can be run on a laptop.

---

## Repository Structure

```
NestedRiesz_DiD/
├── time_varying_treatment/   # Exercise 1: time-varying treatment simulations
├── did/
│   ├── simulations/          # Exercise 2: DiD simulations
│   └── application/          # Exercise 4: DiD empirical application (minimum wage)
├── surrogate_application/    # Exercise 3: surrogate/NNIPV empirical application
├── DiD/                      # core DiD estimator library (shared utilities)
├── NNIPV/                    # NNIPV estimator and pre-computed results
├── environment.yml           # conda environment specification
└── clean_requirements.txt    # pinned pip dependencies
```

---

## Environment Setup

All four exercises share a single `conda` environment named `riesz`, built on Python 3.10:

```bash
conda env create -f did/environment.yml
conda activate riesz
```

The neural network estimator will use a GPU automatically if one is available and fall back to CPU otherwise. The LASSO estimator calls R through `rpy2`, so R (version ≥ 4.2) must be installed along with the `glmnet` package. A helper script installs `glmnet` into the right library path:

```bash
bash time_varying_treatment/install_glmnet.sh
```

Alternatively, run `install.packages("glmnet")` from an R session. If R or `glmnet` is absent, only the LASSO method will fail; RF and the neural network will still run.

**Replicability (this replication package):** use `conda activate riesz` and verify pins in `setup/clean_requirements.txt`. Auto-NN runs on CPU (not GPU) with single-thread BLAS and fixed DataLoader shuffle seeds so repeated runs agree. See `setup/REPRODUCIBILITY.txt` for details and verify commands.

---

## Exercise 1: Time-Varying Treatment Simulations

**Location:** `time_varying_treatment/`

This exercise evaluates estimators for E[Y(1,1)] and the ATE under time-varying treatment. The simulation crosses two outcome DGPs (linear and nonlinear) with several propensity score specifications (logistic, truncated-logistic, and truncated-adversarial), running 500 Monte Carlo iterations at each of three sample sizes (N = 500, 1000, 2000). The estimators compared are an oracle, the Bradic et al. baseline, and three versions of the proposed nested Riesz estimator — using LASSO, random forests (RF), and a neural network (Net) to fit the nuisance functions.

We ran these on a SLURM cluster, submitting up to 10,500 array jobs. Each job requests 4 CPUs, 16 GB of RAM, and up to 2 hours of wall time, with Python 3.10 and R 4.4 loaded as modules.

For the main E[Y(1,1)] simulations (4 DGPs × 3 sample sizes × 500 iterations = 6,000 jobs), submit `submit.sh` from the `time_varying_treatment/` directory. For ATE simulations use `submit_ate.sh`, and for additional propensity configurations used in the appendix use `submit_nonATE.sh` (or `submit_nonATE_part1.sh` / `submit_nonATE_part2.sh` to split across two batches).

```bash
cd time_varying_treatment
sbatch submit.sh
```

Each array job maps its task ID to a DGP, sample size, and iteration index, then dispatches to either `run_sim.py` (linear-outcome DGPs) or `run_sim_nonlinear.py` (nonlinear-outcome DGP):

```bash
python run_sim.py <iteration> <N> <func_name> [lower] [upper]
# for example:
python run_sim.py 42 1000 truncated_logistic 0.1 0.9
```

Results are written to `results/N{N}/{func_name}/result_{t}.pt` as PyTorch files containing point estimates, standard deviations, and the true parameter value. Jobs that find an existing output file exit immediately, so the array can be safely resubmitted after cluster failures without duplicating work.

Once all jobs finish, run `collect_sim_results.py` from the same directory to load all result files and print simulation tables (bias, RMSE, coverage, interval length) to stdout:

```bash
python collect_sim_results.py > summary_tables.txt
```

---

## Exercise 2: DiD Simulations

**Location:** `did/simulations/`

This exercise evaluates difference-in-differences estimators in a panel setting where covariates evolve between periods. The DGP features a binary treatment, pre- and post-period covariate vectors, and a nonlinearly-determined outcome. Three propensity score specifications are considered (logistic, truncated-logistic, and truncated-adversarial) with 100 Monte Carlo iterations at each of three sample sizes (N = 500, 1000, 2000). The estimators compared are OLS, a linear DiD estimator, and three Riesz-based machine learning estimators (LASSO, RF, and Net).

As with Exercise 1, these simulations were run on a SLURM cluster. The main job array has 9 tasks (3 propensity models × 3 sample sizes), each requesting 1 CPU, 8 GB of RAM, and up to 4 days of wall time.

```bash
cd did/simulations
sbatch cluster_job.sbatch
```

This runs `simulation_cluster.py` for each task, with the SLURM task ID selecting the propensity model and sample size. Results are saved as `.pt` files in `results/`. Robustness checks with alternative DGP coefficients, a low-rank outcome structure, and a nonlinear outcome can be run via `simulation_cluster_only_OLS_different_coef.py`, `simulation_cluster_only_OLS_low_rank.py`, and `simulation_cluster_only_OLS_nonlin_y_final.py`, submitted analogously.

Once all jobs complete, run `collect_results.py` to print the summary table and save it to `results/summary.csv`. For the OLS robustness check results, run `collect_ols_results.py` after those scripts finish.

```bash
python collect_results.py
python collect_ols_results.py  # for robustness check results
```

---

## Exercise 3: Surrogate Application

**Location:** `surrogate_application/`

This exercise applies the proposed surrogate estimator to evaluate the effect of a job training program on quarterly earnings and employment. The application combines an experimental sample — used only as a benchmark — with an observational sample, and estimates the average treatment effect using intermediate outcome surrogates. Estimators based on LASSO, random forests, and neural networks are compared against a difference-in-means benchmark from the experimental data, a naive observational estimate, and several NNIPV baselines from Athey et al.

To reproduce, open `surrogate_application/application_tables.ipynb` in Jupyter and run all cells. The notebook reads observational data from `data/others_data.csv` and experimental data from `data/river_data.csv`, fits each estimator, and produces the figures from the paper. Pre-computed NNIPV results are read from `NNIPV/results_nnpiv_earnings.pkl` and `NNIPV/results_nnpiv.pkl`, which must be present; they are generated by `NNIPV/gains_app.ipynb`.

---

## Exercise 4: DiD Application (Minimum Wage)

**Location:** `did/application/`

This exercise applies the dynamic DiD estimator to a county-level panel to estimate the effect of minimum wage increases on teenage employment. The estimand is the ATT for counties that raised their minimum wage, measured at each post-treatment year from 2004 through 2007. The estimators compared include a standard OLS first-difference regression, the Caetano linear estimator, and the proposed Riesz-based estimators (LASSO, RF, Net), benchmarked against the Sant'Anna–Zhao doubly-robust DiD estimator and an RF-based static estimator.

Open `did/application/final_application.ipynb` in Jupyter and run all cells. The notebook handles all data loading, estimation, and plotting. Results for each year are cached to `results/min_wage_cache/results_{year}.csv` on first run and loaded from cache on subsequent runs. To force recomputation for a given year, pass `force_recompute=True` to `plot_att_estimates()`. Each year takes several minutes due to cross-fitting; no GPU is needed.

---

## Dependencies

The full list of pinned versions is in `clean_requirements.txt` and `did/environment.yml`. Key packages:

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.7.0 |
| econml | 0.15.1 |
| scikit-learn | 1.5.2 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| statsmodels | 0.14.4 |
| rpy2 | 3.6.0 |
| R (external) | ≥ 4.2 |
| glmnet (R package) | any recent version |

Cluster logs are written to `time_varying_treatment/logs/` and `did/simulations/logs/`.
