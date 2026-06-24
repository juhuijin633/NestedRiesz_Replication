"""Automatic (Dynamic Riesz) estimators.

Writes per-estimator CSVs to results/intermediate/ after each run:
    auto_{outcome}_{estimator}.csv   e.g. auto_earn_Auto_Lasso.csv

Writes combined CSV per outcome when all estimators finish:
    results/intermediate/auto_{outcome}_estimates.csv

Replication notes (surrogate_application/application_fit_final.py):
  - AUTO_SEED=0, FOLDS=5 from utils/hyperparams.py
  - KFold random_state=42
  - Auto-Lasso / Auto-RF: estimateDynamicRiesz(..., subsetting=False)
  - Auto-NN: estimateDynamicRiesz_subsetting_net(...)  (separate code path)
  - Estimator order: Net, Lasso, RF
  - seed_everything(AUTO_SEED) before each computed auto method (matches
    application_fit_final.py torch.manual_seed(0) before each fit).
  - This differs from manual estimates — see 3_calc_manual_estimates.py.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.dynamicRieszFunctions import (
    estimateDynamicRiesz,
    estimateDynamicRiesz_subsetting_net,
)
from utils.hyperparams import (
    AUTO_SEED,
    FOLDS,
    lasso_a_settings,
    lasso_f_settings,
    net_a_settings,
    net_f_settings,
    rf_a_settings,
    rf_f_settings,
)

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data" / "processed"
INTERMEDIATE_DIR = APP_DIR / "results" / "intermediate"

PRETREAT_VARS = (
    [f"paid{i}" for i in range(1, 5)]
    + [f"tcpp{i}" for i in range(1, 11)]
    + [f"tcprn{i}" for i in range(1, 11)]
)
COVARIATES = [
    "xsexf", "xhsdip", "xchld05", "single",
    "grd1720", "grade16", "grd1315", "grade12", "grde911", "white",
    "hisp", "black", "age",
] + PRETREAT_VARS

QUARTER = 6
Z_SCORE = 1.96

OUTCOME_LABELS = {"earn": "earnings", "employ": "employment"}

# Order matches application_fit_final.py (Net first, then Lasso, then RF).
AUTO_ESTIMATORS = [
    ("Auto-NN", "net"),
    ("Auto-Lasso", "lasso"),
    ("Auto-RF", "rf"),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sample(application: str, quarter: int) -> dict:
    """Stack observational (G=1) and experimental (G=0) units; mask D on obs, Y on exp."""
    exp = pd.read_csv(DATA_DIR / "exp_data.csv")
    obs = pd.read_csv(DATA_DIR / "obs_data.csv")

    if application == "employ":
        s_cols = (
            [f"employ{i}" for i in range(1, quarter + 1)]
            + [f"aid{i}" for i in range(1, quarter + 1)]
            + [f"earn{i}" for i in range(1, quarter + 1)]
        )
    else:
        s_cols = (
            [f"earn{i}" for i in range(1, quarter + 1)]
            + [f"aid{i}" for i in range(1, quarter + 1)]
        )

    y = pd.concat([obs[f"Y_{application}"], exp[f"Y_{application}"]], ignore_index=True)
    x = pd.concat([obs[COVARIATES], exp[COVARIATES]], ignore_index=True)
    s = pd.concat([obs[s_cols], exp[s_cols]], ignore_index=True)
    d = pd.concat([obs["e"], exp["e"]], ignore_index=True)
    g = pd.concat([pd.Series(np.ones(len(obs))), pd.Series(np.zeros(len(exp)))], ignore_index=True)

    Y = torch.tensor(y.values, dtype=torch.float64).view(-1, 1)
    X = torch.tensor(x.values, dtype=torch.float64)
    S = torch.tensor(s.values, dtype=torch.float64)
    D = torch.tensor(d.values, dtype=torch.float64).view(-1, 1)
    G = torch.tensor(g.values, dtype=torch.float64).view(-1, 1)

    Y_est = Y.clone()
    D_est = D.clone()
    D_est[G.bool()] = 0
    Y_est[(1 - G).bool()] = 0

    return {"Y_est": Y_est, "X": X, "S": S, "D_est": D_est, "G": G, "n": len(y)}


def _run_estimator(method: str, sample: dict) -> tuple[float, float]:
    y, g, x, d, s = sample["Y_est"], sample["G"], sample["X"], sample["D_est"], sample["S"]
    seed_everything(AUTO_SEED)

    if method == "net":
        att, std, _ = estimateDynamicRiesz_subsetting_net(
            y, g, x, d, s, FOLDS,
            method_a="Net", net_a_settings=copy.deepcopy(net_a_settings),
            method_f="Net", net_f_settings=copy.deepcopy(net_f_settings),
            seed=AUTO_SEED,
        )
    elif method == "lasso":
        att, std, _ = estimateDynamicRiesz(
            y, g, x, d, s, FOLDS,
            method_a="LASSO", lasso_a_settings=copy.deepcopy(lasso_a_settings),
            method_f="LASSO", lasso_f_settings=copy.deepcopy(lasso_f_settings),
            seed=AUTO_SEED, subsetting=False,
        )
    elif method == "rf":
        att, std, _ = estimateDynamicRiesz(
            y, g, x, d, s, FOLDS,
            method_a="RF", rf_a_settings=copy.deepcopy(rf_a_settings),
            method_f="RF", rf_f_settings=copy.deepcopy(rf_f_settings),
            seed=AUTO_SEED, subsetting=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(att.item()), float(std.item())


def run_application(application: str, quarter: int, force: bool = False) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_sample(application, quarter)
    rows = []
    label = OUTCOME_LABELS[application]

    for est_label, method in AUTO_ESTIMATORS:
        slug = est_label.replace("-", "_")
        intermediate = INTERMEDIATE_DIR / f"auto_{application}_{slug}.csv"

        if intermediate.exists() and not force:
            rows.append(pd.read_csv(intermediate).iloc[0].to_dict())
            print(f"[{label}] {est_label} Complete.", flush=True)
            continue

        point, sigma = _run_estimator(method, sample)
        se = sigma / (sample["n"] ** 0.5)
        row = {
            "outcome": application,
            "quarter": quarter,
            "estimator": est_label,
            "group": "auto",
            "point_estimate": point,
            "se": se,
            "ci_lower": point - Z_SCORE * se,
            "ci_upper": point + Z_SCORE * se,
        }
        pd.DataFrame([row]).to_csv(intermediate, index=False)
        rows.append(row)
        print(f"[{label}] {est_label} Complete.", flush=True)

    out = INTERMEDIATE_DIR / f"auto_{application}_estimates.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--applications", nargs="+", default=["earn", "employ"])
    parser.add_argument("--force", action="store_true", help="Recompute even if cached CSVs exist.")
    args = parser.parse_args()
    for app in args.applications:
        run_application(app, QUARTER, force=args.force)


if __name__ == "__main__":
    main()
