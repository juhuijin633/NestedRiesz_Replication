"""Learner hyperparameters for DiD simulations.

Canonical values from did/simulations/simulation_cluster.py.
Per-replication randomness is NOT set here — it comes from seed=t passed
at call time (see 1_run_simulation.py).
"""

from __future__ import annotations

import copy

import torch

FOLDS = 5

_lasso_common = {
    "lambda_val": 0,
    "beta_start": None,
    "D_LB": 0,
    "D_add": 0.2,
    "c1": "CV",
    "c2": 0.1,
    "tol": 1e-5,
    "max_iter": 100,
    "b_degree": 1,
    "control": {"maxIter": 1000, "optTol": 1e-5, "zeroThreshold": 1e-6},
}

lasso_a_settings = copy.deepcopy(_lasso_common)
lasso_f_settings = copy.deepcopy(_lasso_common)

lasso_cv_settings = {
    "b_degree": 1,
    "cv_folds": FOLDS,
    "random_state": 42,
}

_rf_common = {
    "l2": 0,
    "n_estimators": 100,
    "criterion": "mse",
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "min_weight_fraction_leaf": 0.0,
    "min_var_fraction_leaf": None,
    "min_var_leaf_on_val": False,
    "max_features": "auto",
    "min_impurity_decrease": 0.0,
    "max_samples": 0.45,
    "min_balancedness_tol": 0.45,
    "honest": True,
    "inference": True,
    "fit_intercept": True,
    "subforest_size": 4,
    "n_jobs": 1,
    "random_state": None,  # set to replication seed t at fit time
    "verbose": 0,
    "warm_start": False,
}

rf_a_settings = {**_rf_common, "poly_degree": 0}
rf_f_settings = {**_rf_common, "poly_degree": 1}

_net_common = {
    "test_split": 0,
    "learner_lr": 1e-4,
    "learner_l2": 1e-3,
    "learner_l1": 0,
    "n_epochs": 100,
    "earlystop_rounds": 20,
    "earlystop_delta": 1e-3,
    "bs": 64,
    "optimizer": "adam",
    "warm_start": False,
    "logger": None,
    "model_dir": ".",
    "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    "n_hidden": 100,
    "drop_prob": 0,
    "degree": 2,
    "interaction_only": True,
    "n_common": 200,
    "act_func": "elu",
}

net_a_settings = copy.deepcopy(_net_common)
net_f_settings = copy.deepcopy(_net_common)
