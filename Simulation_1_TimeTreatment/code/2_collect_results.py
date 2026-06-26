#!/usr/bin/env python3
"""Aggregate per-iteration .pt files into summary.csv."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from utils.simulation_config import CONFIGS, METHOD_LABELS, NS, TMAX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"
RESULTS_DIR = PROJECT_ROOT / "results"


def _job_dir(config_id: str, n: int) -> Path:
    """Resolve result directory (flat layout preferred; legacy psi11/ subdir supported)."""
    flat = INTERMEDIATE_DIR / f"N_{n}" / config_id
    if flat.exists():
        return flat
    return INTERMEDIATE_DIR / "psi11" / f"N_{n}" / config_id


def _load_results(config_id: str, n: int) -> tuple[list[dict], list[int]]:
    results, missing = [], []
    base = _job_dir(config_id, n)
    for t in range(TMAX):
        path = base / f"result_{t}.pt"
        if not path.exists():
            missing.append(t)
            continue
        results.append(torch.load(path, weights_only=False))
    return results, missing


def _compute_metrics(results: list[dict], n: int) -> pd.DataFrame:
    """Bias / RMSE / coverage — same formulas as collect_sim_results.py."""
    t_count = len(results)
    theta_true = results[0]["theta_true"]
    if isinstance(theta_true, torch.Tensor):
        theta_true = theta_true.item()

    pred_theta = torch.zeros(t_count, len(METHOD_LABELS))
    pred_sig = torch.zeros(t_count, len(METHOD_LABELS))

    for i, row in enumerate(results):
        pred_theta[i, 0] = row["oracle_theta"]
        pred_sig[i, 0] = row["oracle_sig"]

        bt = row.get("bradic_theta", float("nan"))
        bs = row.get("bradic_sig", float("nan"))
        pred_theta[i, 1] = bt if not (isinstance(bt, float) and math.isnan(bt)) else float("nan")
        pred_sig[i, 1] = bs if not (isinstance(bs, float) and math.isnan(bs)) else float("nan")

        pred_theta[i, 2:] = row["pred_theta"]
        pred_sig[i, 2:] = row["pred_sig"]

    bias = torch.nanmean(pred_theta - theta_true, 0)
    rmse = torch.sqrt(torch.nanmean((pred_theta - theta_true) ** 2, 0))
    # pred_sig = influence-function SD (Oracle: SD of per-unit psi); SE = sig / sqrt(n).
    se = pred_sig / (n ** 0.5)
    ub = pred_theta + 1.96 * se
    lb = pred_theta - 1.96 * se
    coverage = torch.nanmean(((theta_true >= lb) & (theta_true <= ub)).float(), 0)
    interval_length = torch.nanmean(ub - lb, 0)

    return pd.DataFrame(
        {
            "Method": METHOD_LABELS,
            "Bias": [round(x, 4) for x in bias.tolist()],
            "RMSE": [round(x, 4) for x in rmse.tolist()],
            "Coverage": [round(x, 4) for x in coverage.tolist()],
            "Interval Length": [round(x, 4) for x in interval_length.tolist()],
        }
    )


def collect(verbose: bool = True) -> pd.DataFrame:
    rows = []
    for cfg in CONFIGS:
        for n in NS:
            results, missing = _load_results(cfg["id"], n)
            if not results:
                if verbose:
                    print(f"\n[{cfg['label']}, N={n}] — no results, skipping")
                continue
            if missing and verbose:
                print(f"\nWarning [{cfg['label']}, N={n}]: {len(missing)} missing iterations")

            df = _compute_metrics(results, n)
            theta_true = results[0]["theta_true"]
            if isinstance(theta_true, torch.Tensor):
                theta_true = theta_true.item()

            for _, method_row in df.iterrows():
                rows.append(
                    {
                        "config": cfg["id"],
                        "label": cfg["label"],
                        "N": n,
                        "method": method_row["Method"],
                        "theta_true": round(theta_true, 4),
                        "iterations": len(results),
                        "bias": method_row["Bias"],
                        "rmse": method_row["RMSE"],
                        "coverage": method_row["Coverage"],
                        "ci_length": method_row["Interval Length"],
                    }
                )

            if verbose:
                print(f"\n{'=' * 65}")
                print(cfg["label"])
                print(f"N = {n}  |  tmax = {TMAX}")
                print(f"True E[Y(1,1)] = {theta_true:.4f}  |  Iterations: {len(results)} / {TMAX}")
                print(f"{'=' * 65}")
                print(df.to_string(index=False, float_format="{:.4f}".format))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = RESULTS_DIR / "summary.csv"
    if out_path.exists() and not args.force:
        print("2. Collect Results — summary.csv exists (use --force to overwrite).")
        return

    df = collect()
    if df.empty:
        raise SystemExit("No intermediate results found.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n2. Collect Results")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
