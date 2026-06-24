"""Automatic (Dynamic Riesz) estimators.

Writes one CSV per outcome to data/processed/:
    auto_{outcome}_estimates.csv

Each file holds long-run ATT point estimates and 95% CIs from the three
Dynamic Riesz estimators used in the figures (Auto-Lasso, Auto-RF, Auto-NN),
using q quarters of surrogate outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.dynamicRieszFunctions import estimateDynamicRiesz

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data" / "processed"

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

FOLDS = 5
SEED = 0
QUARTER = 6  # quarters of surrogate outcomes (earn1..earn6, etc.)
Z_SCORE = 1.96

OUTCOME_LABELS = {"earn": "earnings", "employ": "employment"}

AUTO_ESTIMATORS = [
    ("Auto-Lasso", "LASSO", "LASSO", False),
    ("Auto-RF", "RF", "RF", False),
    ("Auto-NN", "Net", "Net", True),
]


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


def run_application(application: str, quarter: int) -> None:
    sample = load_sample(application, quarter)
    rows = []

    label = OUTCOME_LABELS[application]

    for est_label, method_a, method_f, subsetting in AUTO_ESTIMATORS:
        torch.manual_seed(SEED)
        att, std, _ = estimateDynamicRiesz(
            sample["Y_est"], sample["G"], sample["X"], sample["D_est"], sample["S"],
            FOLDS, method_a=method_a, method_f=method_f, seed=SEED, subsetting=subsetting,
        )
        point = float(att.item())
        se = float(std.item()) / (sample["n"] ** 0.5)
        rows.append({
            "outcome": application,
            "quarter": quarter,
            "estimator": est_label,
            "group": "auto",
            "point_estimate": point,
            "se": se,
            "ci_lower": point - Z_SCORE * se,
            "ci_upper": point + Z_SCORE * se,
        })
        print(f"[{label}] {est_label} Complete.", flush=True)

    out = DATA_DIR / f"auto_{application}_estimates.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--applications", nargs="+", default=["earn", "employ"])
    args = parser.parse_args()
    for app in args.applications:
        run_application(app, QUARTER)


if __name__ == "__main__":
    main()
