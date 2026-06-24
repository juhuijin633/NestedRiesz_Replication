

DATA: Raw data quarterly.mat is drawn from Athey, et al. 2025. This data is then processed into experimental data (Riverside site) and observational data (remaining sites). 

CODE: 

1 | clean_data.py: raw quarterly.mat --> obs_data.csv and exp_data.csv 
2 | calc_auto_estimates.py: Auto-Lasso, Auto-RF, Auto-NN 
3 | calc_manual_estimates.py: Manual-Lasso, Manual-RF, Manual-NN estimates using NNIPV package
4 | tables_figs.py: combines all estimators and generates final figures 

utils -- scripts called to compute estimates
run.py is a meta-script that runs (1)-(2)-(3)-(4) in order. 

RESULTS: 
. | earnings_estimates.csv
. | earnings_figure.png 
. | employment_estimates.csv
. | employment_figure.png
