"""Manual (NNPIV / DML) estimators.

Writes per-estimator CSVs to results/intermediate/ after each run:
    manual_{outcome}_{estimator}.csv   e.g. manual_earn_Manual_Lasso.csv

Replication policy (trimmed pipeline — three estimators only):
  - Does NOT replay the full gains_app.ipynb sequence (TSLS, RKHS, sparse, etc.).
  - seed_everything(MANUAL_SEED) once per outcome (earn/employ), matching gains_app.ipynb.
  - MANUAL_SEED=123 matches gains_app.ipynb.
  - Re-run with --force after code/data fixes; cached per-estimator CSVs are not invalidated
    automatically when only one outcome is recomputed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from nnpiv.ensemble import EnsembleIV
from nnpiv.neuralnet.agmm import AGMM
from nnpiv.semiparametrics import DML_longterm
from nnpiv.tsls import regtsls
from utils.hyperparams import FOLDS, MANUAL_SEED

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

QUARTER = 6  # quarters of surrogate outcomes (earn1..earn6, etc.)
Z_SCORE = 1.96
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

OUTCOME_LABELS = {"earn": "earnings", "employ": "employment"}

DML_KWARGS = dict(
    longterm_model="surrogacy",
    n_folds=FOLDS,
    n_rep=1,
    random_seed=MANUAL_SEED,
    CHIM=False,
    prop_score=LogisticRegression(max_iter=2000),
)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sample(application: str, quarter: int) -> dict:
    """Stack observational (G=1) and experimental (G=0) units; mask D on obs, Y on exp.

    Y, D, G must be (n, 1) column vectors — gains_app.ipynb used torch tensors with
    .view(-1, 1). With 1D (n,) arrays, nnpiv propensity IPW terms broadcast to (n, n)
    and estimates become wrong (often opposite sign).
    """
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

    y_est = y.copy()
    d_est = d.copy()
    y_est[g == 0] = 0
    d_est[g == 1] = 0

    return {
        "Y": y_est.to_numpy(dtype=float).reshape(-1, 1),
        "D": d_est.to_numpy(dtype=float).reshape(-1, 1),
        "X": x.to_numpy(dtype=float),
        "S": s.to_numpy(dtype=float),
        "G": g.to_numpy(dtype=float).reshape(-1, 1),
    }


def fit_dml(sample: dict, model1, **extra) -> tuple[float, float, float]:
    dml = DML_longterm(
        sample["Y"], sample["D"], sample["S"], sample["G"],
        X1=sample["X"], model1=model1, **DML_KWARGS, **extra,
    )
    theta, _var, ci = dml.dml()
    return float(theta), float(ci[0]), float(ci[1])


def fit_manual_lasso(sample: dict) -> tuple[float, float, float]:
    return fit_dml(
        sample,
        model1=[regtsls(), regtsls()],
        nn_1=[False, False],
    )


def fit_manual_rf(sample: dict) -> tuple[float, float, float]:
    rf = EnsembleIV(n_iter=200, max_abs_value=2)
    return fit_dml(
        sample,
        model1=[rf, rf],
        nn_1=[False, False],
    )


def fit_manual_nn(sample: dict) -> tuple[float, float, float]:
    p, n_hidden = 0, 100
    fitargs = dict(
        n_epochs=100, bs=64, learner_lr=1e-4, adversary_lr=1e-4,
        learner_l2=1e-3, adversary_l2=1e-3, adversary_norm_reg=1e-1, device=DEVICE,
    )

    def learner(n_in):
        return nn.Sequential(
            nn.Dropout(p=p), nn.Linear(n_in, n_hidden), nn.LeakyReLU(),
            nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
            nn.Dropout(p=p), nn.Linear(n_hidden, 1),
        )

    def adversary(n_in):
        return nn.Sequential(
            nn.Dropout(p=p), nn.Linear(n_in, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
            nn.Dropout(p=p), nn.Linear(n_hidden, 1),
        )

    m1_dim = sample["S"].shape[1] + sample["X"].shape[1]
    m2_dim = sample["X"].shape[1]
    return fit_dml(
        sample,
        model1=[AGMM(learner(m1_dim), adversary(m1_dim)), AGMM(learner(m2_dim), adversary(m2_dim))],
        nn_1=[True, True],
        fitargs1=[fitargs, fitargs],
    )


MANUAL_ESTIMATORS = [
    ("Manual-Lasso", fit_manual_lasso),
    ("Manual-RF", fit_manual_rf),
    ("Manual-NN", fit_manual_nn),
]


def run_application(application: str, quarter: int, force: bool = False) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_sample(application, quarter)
    rows = []
    label = OUTCOME_LABELS[application]

    with threadpool_limits(1):
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        seed_everything(MANUAL_SEED)

        for est_label, fit_fn in MANUAL_ESTIMATORS:
            slug = est_label.replace("-", "_")
            intermediate = INTERMEDIATE_DIR / f"manual_{application}_{slug}.csv"

            if intermediate.exists() and not force:
                rows.append(pd.read_csv(intermediate).iloc[0].to_dict())
                print(f"[{label}] {est_label} Complete.", flush=True)
                continue

            point, ci_lo, ci_hi = fit_fn(sample)
            se = (ci_hi - ci_lo) / (2 * Z_SCORE)
            row = {
                "outcome": application,
                "quarter": quarter,
                "estimator": est_label,
                "group": "manual",
                "point_estimate": point,
                "se": se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            }
            pd.DataFrame([row]).to_csv(intermediate, index=False)
            rows.append(row)
            print(f"[{label}] {est_label} Complete.", flush=True)

    out = INTERMEDIATE_DIR / f"manual_{application}_estimates.csv"
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
