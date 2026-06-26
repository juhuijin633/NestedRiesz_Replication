#!/usr/bin/env python3
"""Monte Carlo simulation for DiD estimators (one N × propensity job per invocation)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from utils.generate_dgp import DiD_DGP
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiDLinear import estimateDiDLinear
from utils.estimateDiD_OLS import estimateDiD_OLS
from utils.hyperparams import (
    FOLDS,
    lasso_a_settings,
    lasso_f_settings,
    net_a_settings,
    net_f_settings,
    rf_a_settings,
    rf_f_settings,
)

PROJECT_ROOT = CODE_DIR.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"

# --- Design (matches simulation_cluster.py; do not change seeds) ---
NS = [500, 1000, 2000]
PROPENSITY_NAMES = ["logistic", "truncated_logistic", "truncated_step"]
TMAX = 500
DIM_X, DIM_Z = 3, 2
DGP_SEED = 123
PHI_Y = 1.0
ATT_N = 1_000_000
N_METHODS = 5
# Per replication t=0..TMAX-1: torch.manual_seed(t); all estimators use seed=t

_PROP_LOWER, _PROP_UPPER = 0.30, 0.70


def _logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))


def _truncated_logistic(x):
    return _PROP_LOWER + (_PROP_UPPER - _PROP_LOWER) * _logistic(x)


def _truncated_step(x):
    return _PROP_LOWER + (_PROP_UPPER - _PROP_LOWER) * (x > 0)


PROPENSITY_MODELS = {
    "logistic": _logistic,
    "truncated_logistic": _truncated_logistic,
    "truncated_step": _truncated_step,
}

# Paper table labels (column order in saved .pt files)
METHOD_LABELS = ["OLS", "Auto-Linear", "Auto-Lasso", "Auto-RF", "Auto-NN"]


def _job_dir(n: int) -> Path:
    return INTERMEDIATE_DIR / f"N_{n}"


def _paths(n: int, model_name: str) -> dict[str, Path]:
    d = _job_dir(n)
    return {
        "att": d / f"{model_name}_ATT.pt",
        "theta": d / f"{model_name}_pred_theta.pt",
        "sig": d / f"{model_name}_pred_sig.pt",
    }


def _run_one(n: int, model_name: str, force: bool) -> None:
    paths = _paths(n, model_name)
    _job_dir(n).mkdir(parents=True, exist_ok=True)

    if paths["theta"].exists() and paths["sig"].exists() and not force:
        print(f"[N={n}, {model_name}] Complete (cached).")
        return

    print(f"\nRunning N={n}, propensity={model_name}")
    dgp = DiD_DGP(
        dim_X=DIM_X, dim_Z=DIM_Z, alpha_1=1,
        gamma_1=torch.ones(DIM_Z), gamma_2=torch.ones(DIM_Z),
        g=PROPENSITY_MODELS[model_name], beta_2=2, delta_3=2,
    )

    data = dgp.generate(n=n * TMAX, seed=DGP_SEED)
    X1, X2, Y1, Y2, Z, D = data["X1"], data["X2"], data["Y1"], data["Y2"], data["Z"], data["D"]
    Y2 = Y2 + PHI_Y * (Y1 > 0).float()

    att_calculations = dgp.simulate_ATT(n=ATT_N)
    torch.save(att_calculations, paths["att"])
    att = att_calculations["ATT"]

    pred_theta = torch.zeros(TMAX, N_METHODS)
    pred_sig = torch.zeros(TMAX, N_METHODS)

    for t in tqdm(range(TMAX), desc=f"N={n}, {model_name}"):
        torch.manual_seed(t)
        sl = slice(t * n, (t + 1) * n)
        X1_sub, X2_sub = X1[sl, :], X2[sl, :]
        Y1_sub = Y1[sl].view(-1, 1)
        Y2_sub = Y2[sl].view(-1, 1)
        D_sub = D[sl].view(-1, 1)
        Z_sub = Z[sl, :]

        pred_theta[t, 0], pred_sig[t, 0] = estimateDiD_OLS(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed=t)
        pred_theta[t, 1], pred_sig[t, 1] = estimateDiDLinear(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed=t)
        pred_theta[t, 2], pred_sig[t, 2], *_ = estimateDynamicRiesz(
            Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, FOLDS,
            method_a="LASSO", lasso_a_settings=lasso_a_settings, method_f="LASSO", lasso_f_settings=lasso_f_settings, seed=t,
        )
        pred_theta[t, 3], pred_sig[t, 3], *_ = estimateDynamicRiesz(
            Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, FOLDS,
            method_a="RF", rf_a_settings=rf_a_settings, method_f="RF", rf_f_settings=rf_f_settings, seed=t,
        )
        pred_theta[t, 4], pred_sig[t, 4], *_ = estimateDynamicRiesz(
            Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, FOLDS,
            method_a="Net", net_a_settings=net_a_settings, method_f="Net", net_f_settings=net_f_settings, seed=t,
        )

    torch.save(pred_theta, paths["theta"])
    torch.save(pred_sig, paths["sig"])

    print(f"\nResults: N={n}, model={model_name}")
    print(f"{'Method':<14} {'Bias':>8} {'RMSE':>8} {'Coverage':>10} {'CI Length':>10}")
    print("-" * 54)
    for k, label in enumerate(METHOD_LABELS):
        theta_k, sig_k = pred_theta[:, k], pred_sig[:, k]
        bias = torch.mean(theta_k - att).item()
        rmse = torch.sqrt(torch.mean((theta_k - att) ** 2)).item()
        # OLS: sig_k is HC0 SE. Riesz: sig_k is IF SD → divide by sqrt(n) for CI (see 2_collect_results.py).
        se_k = sig_k if k == 0 else sig_k / (n ** 0.5)
        ci_low, ci_high = theta_k - 1.96 * se_k, theta_k + 1.96 * se_k
        coverage = torch.mean(((ci_low <= att) & (att <= ci_high)).float()).item()
        print(f"{label:<14} {bias:>8.4f} {rmse:>8.4f} {coverage:>10.2f} {torch.mean(2 * 1.96 * se_k).item():>10.4f}")

    print(f"[N={n}, {model_name}] Complete.")


def _resolve_jobs(args) -> list[tuple[int, str]]:
    if args.n is not None and args.model is not None:
        return [(args.n, args.model)]

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        return [(NS[idx // len(PROPENSITY_NAMES)], PROPENSITY_NAMES[idx % len(PROPENSITY_NAMES)])]

    if args.all or (args.n is None and args.model is None):
        return [(n, m) for n in NS for m in PROPENSITY_NAMES]

    raise SystemExit("Specify --N and --model, set SLURM_ARRAY_TASK_ID, or omit args to run all locally.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, choices=NS)
    parser.add_argument("--model", choices=PROPENSITY_NAMES)
    parser.add_argument("--all", action="store_true", help="Run all 9 N × propensity jobs (local default).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("1. Run Simulation")
    for n, model_name in _resolve_jobs(args):
        _run_one(n, model_name, args.force)


if __name__ == "__main__":
    main()
