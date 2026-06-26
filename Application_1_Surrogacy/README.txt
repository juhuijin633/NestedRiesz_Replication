Application_1_Surrogacy
=======================

DATA: Raw data quarterly.mat is drawn from Athey, et al. 2025. This data is then processed into experimental data (Riverside site) and observational data (remaining sites). 

CODE: 

1 | 1_clean_data.py: raw quarterly.mat --> data/processed/obs_data.csv and exp_data.csv 
2 | 2_calc_auto_estimates.py: Auto-Lasso, Auto-RF, Auto-NN 
3 | 3_calc_manual_estimates.py: Manual-Lasso, Manual-RF, Manual-NN estimates using NNIPV package
4 | 4_tables_figs.py: combines all estimators and generates final figures 

utils -- scripts called to compute estimates
RUN.py is a meta-script that runs (1)-(2)-(3)-(4) in order. 

REPRODUCIBILITY
---------------
  conda activate riesz   # pins in setup/clean_requirements.txt (48 packages)
  utils/seeding.py: full RNG + single-thread BLAS + deterministic cuDNN
  Auto-NN trains on CPU; DataLoader uses DATALOADER_SEED (hyperparams.py)
  Recompute: python 2_calc_auto_estimates.py --force

RESULTS: 
results/intermediate/ -- per-estimator and combined estimate CSVs as they are computed
results/earn_estimates.csv, earn_figure.png, employ_estimates.csv, employ_figure.png -- final tables and figures

Paper Table F.9 / upstream q6.csv are reference snapshots; fresh runs on this
pinned env should agree with each other when using --force, not necessarily with
historical paper digits.
