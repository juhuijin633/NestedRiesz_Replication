#!/usr/bin/env python3
"""Monte Carlo simulation for E[Y(1,1)] (one replication per invocation on cluster)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from utils.dynamicRieszBradic import estimateBradic
from utils.dynamicRieszFunctions import estimateDynamicRiesz_all
from utils.generate_dgp import SimulationData, generate
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

# --- Design (matches time_varying_treatment/submit.sh; do not change seeds) ---
NS = [500, 1000, 2000]
TMAX = 500
DGP_SEED_OFFSET = 123  # torch.manual_seed(123 + t)

CONFIGS = [
    {
        "id": "linear_truncated_logistic",
        "dgp": "linear",
        "func": "truncated_logistic",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Linear DGP + truncated logistic",
    },
    {
        "id": "nonlinear_truncated_adv",
        "dgp": "nonlinear",
        "func": "truncated_adv",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Nonlinear DGP + truncated adversarial",
    },
    {
        "id": "linear_truncated_adv",
        "dgp": "linear",
        "func": "truncated_adv",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Linear DGP + truncated adversarial",
    },
    {
        "id": "linear_logistic",
        "dgp": "linear",
        "func": "logistic",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Linear DGP + logistic",
    },
]


def _result_path(n: int, config_id: str, t: int) -> Path:
    return INTERMEDIATE_DIR / f"N_{n}" / config_id / f"result_{t}.pt"


def _decode_cluster_task(task_id: int) -> tuple[dict, int, int]:
    """Map SLURM array id → (config, N, iteration t). Matches submit.sh order."""
    n_configs = len(CONFIGS)
    block = len(NS) * TMAX
    dgp_idx = task_id // block
    remainder = task_id % block
    n_idx = remainder // TMAX
    t = remainder % TMAX
    if dgp_idx >= n_configs:
        raise ValueError(f"Task id {task_id} out of range for {n_configs} configs")
    return CONFIGS[dgp_idx], NS[n_idx], t


def _oracle_psi11(data: SimulationData) -> tuple[float, float]:
    a1, a2 = data.D[:, 0:1], data.D[:, 1:2]
    rr2 = a1 * a2 / (data.pi1 * data.pi2_1)
    rr1 = a1 / data.pi1
    psi = rr2 * data.Y - (rr1 - 1) * data.mu1_1 - (rr2 - rr1) * data.mu2_1
    theta = torch.mean(psi).item()
    sig = torch.sqrt(torch.mean((psi - torch.mean(psi)) ** 2)).item()
    return theta, sig


def _estimate_psi11(data: SimulationData) -> dict:
    y, x, d = data.Y, data.X, data.D
    x_index = data.X_index
    d_vec = torch.ones(d.shape)

    oracle_theta, oracle_sig = _oracle_psi11(data)

    try:
        bradic_result = estimateBradic(y, x, d, x_index, FOLDS)
        bradic_theta = float(bradic_result[6])
        bradic_sig = float(torch.sqrt(torch.tensor(float(bradic_result[9]))))
    except Exception as exc:
        print(f"  Bradic failed: {exc}")
        bradic_theta = float("nan")
        bradic_sig = float("nan")

    pt, sg, *_ = estimateDynamicRiesz_all(
        y,
        x,
        d,
        d_vec,
        x_index,
        FOLDS,
        lasso_a_settings=lasso_a_settings,
        lasso_f_settings=lasso_f_settings,
        rf_a_settings=rf_a_settings,
        rf_f_settings=rf_f_settings,
        net_a_settings=net_a_settings,
        net_f_settings=net_f_settings,
    )

    return {
        "theta_true": data.theta_true,
        "oracle_theta": oracle_theta,
        "oracle_sig": oracle_sig,
        "bradic_theta": bradic_theta,
        "bradic_sig": bradic_sig,
        "pred_theta": pt,
        "pred_sig": sg,
    }


def _run_one(config: dict, n: int, t: int, force: bool) -> None:
    out_path = _result_path(n, config["id"], t)
    if out_path.exists() and not force:
        print(f"[{config['id']}, N={n}, t={t}] Cached.")
        return

    torch.manual_seed(DGP_SEED_OFFSET + t)
    data = generate(config["dgp"], n, config["func"], config["lower"], config["upper"])

    t0 = time.time()
    payload = _estimate_psi11(data)
    payload["iteration"] = t
    payload["N"] = n
    payload["func_name"] = config["func"]
    payload["config_id"] = config["id"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[{config['label']}, N={n}] Iteration {t} done in {time.time() - t0:.1f}s")


def _resolve_jobs(args) -> list[tuple[dict, int, int | None]]:
    """Return list of (config, N, t). t=None means all iterations."""
    if args.config is not None and args.n is not None:
        config = next(c for c in CONFIGS if c["id"] == args.config)
        if args.iteration is not None:
            return [(config, args.n, args.iteration)]
        return [(config, args.n, t) for t in range(TMAX)]

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        config, n, t = _decode_cluster_task(int(task_id))
        return [(config, n, t)]

    if args.all or (args.config is None and args.n is None):
        jobs: list[tuple[dict, int, int | None]] = []
        for config in CONFIGS:
            for n in NS:
                for t in range(TMAX):
                    jobs.append((config, n, t))
        return jobs

    raise SystemExit(
        "Specify --config and --N (optional --iteration), set SLURM_ARRAY_TASK_ID, "
        "or use --all for local full sweep."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=[c["id"] for c in CONFIGS])
    parser.add_argument("--N", type=int, choices=NS)
    parser.add_argument("--iteration", type=int, help="Single MC replication (0..499).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configs × N × 500 iterations locally (very slow).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("1. Run Simulation")
    for config, n, t in _resolve_jobs(args):
        assert t is not None
        _run_one(config, n, t, args.force)


if __name__ == "__main__":
    main()
