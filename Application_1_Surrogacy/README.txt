
DATA: Raw data quarterly.mat is drawn from Athey, et al. 2025. This data is then processed into experimental data (Riverside site) and observational data (remaining sites). 

CODE: 

1 | 1_clean_data.py: raw quarterly.mat --> data/processed/obs_data.csv and exp_data.csv 
2 | 2_calc_auto_estimates.py: Auto-Lasso, Auto-RF, Auto-NN 
3 | 3_calc_manual_estimates.py: Manual-Lasso, Manual-RF, Manual-NN estimates using NNIPV package
4 | 4_tables_figs.py: combines all estimators and generates final figures 

utils -- scripts called to compute estimates
RUN.py is a meta-script that runs (1)-(2)-(3)-(4) in order. 

RESULTS: 
results/intermediate/ -- per-estimator and combined estimate CSVs as they are computed
results/earn_estimates.csv, earn_figure.png, employ_estimates.csv, employ_figure.png -- final tables and figures
