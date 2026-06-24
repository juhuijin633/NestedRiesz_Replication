"""Single source of truth for Application 2 learner hyperparameters and run config.

All estimate scripts and utils modules import from here — do not duplicate
these dicts elsewhere in Application_2_DiD.
"""

from __future__ import annotations

import copy

import torch

FOLDS = 5
SEED = 0  # final_application.ipynb dynamic_riesz_results default

lasso_cv_settings = {
    "b_degree": 1,
    "cv_folds": FOLDS,
    "random_state": 42,
}

_lasso = {
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

lasso_a_settings = copy.deepcopy(_lasso)
lasso_f_settings = copy.deepcopy(_lasso)

_rf = {
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
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
}

rf_a_settings = {**_rf, "poly_degree": 0}
rf_f_settings = {**_rf, "poly_degree": 1}

_net = {
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

net_a_settings = copy.deepcopy(_net)
net_f_settings = copy.deepcopy(_net)
