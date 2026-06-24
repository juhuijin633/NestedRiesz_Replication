#!/usr/bin/env python3
"""Aggregate simulation .pt files into summary.csv and per-table CSVs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"
RESULTS_DIR = PROJECT_ROOT / "results"

NS = [500, 1000, 2000]
PROPENSITY_MODELS = ["logistic", "truncated_logistic", "truncated_step"]
METHOD_LABELS = ["OLS", "Auto-Linear", "Auto-Lasso", "Auto-RF", "Auto-NN"]

TABLE_TITLES = {
    "logistic": "Table D.6 Logistic Propensities",
    "truncated_logistic": "Table D.7 Truncated Logistic Propensities",
    "truncated_step": "Table D.8 Truncated Step Propensities",
}


def _load_job(n: int, model_name: str) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    job_dir = INTERMEDIATE_DIR / f"N_{n}"
    theta_path = job_dir / f"{model_name}_pred_theta.pt"
    sig_path = job_dir / f"{model_name}_pred_sig.pt"
    att_path = job_dir / f"{model_name}_ATT.pt"

    if not all(p.exists() for p in (theta_path, sig_path, att_path)):
        return None

    pred_theta = torch.load(theta_path, weights_only=False)
    pred_sig = torch.load(sig_path, weights_only=False)
    att = torch.load(att_path, weights_only=False)["ATT"]
    return pred_theta, pred_sig, att.item()


def collect() -> pd.DataFrame:
    rows = []
    for n in NS:
        for model_name in PROPENSITY_MODELS:
            loaded = _load_job(n, model_name)
            if loaded is None:
                print(f"Missing N={n}, model={model_name} — skipping")
                continue
            pred_theta, pred_sig, att = loaded

            for k, method in enumerate(METHOD_LABELS):
                theta_k, sig_k = pred_theta[:, k], pred_sig[:, k]
                se_k = sig_k / math.sqrt(n)
                ci_low, ci_high = theta_k - 1.96 * se_k, theta_k + 1.96 * se_k
                rows.append({
                    "N": n,
                    "model": model_name,
                    "method": method,
                    "bias": round(torch.mean(theta_k - att).item(), 4),
                    "rmse": round(torch.sqrt(torch.mean((theta_k - att) ** 2)).item(), 4),
                    "coverage": round(torch.mean(((ci_low <= att) & (att <= ci_high)).float()).item(), 4),
                    "ci_length": round(torch.mean(2 * 1.96 * se_k).item(), 4),
                })
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
    print("2. Collect Results")
    print(df.to_string(index=False))
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    for model_name, title in TABLE_TITLES.items():
        sub = df[df["model"] == model_name].drop(columns=["model"])
        table_path = RESULTS_DIR / f"table_{model_name}.csv"
        sub.to_csv(table_path, index=False)
        print(f"  {title} → {table_path.name}")


if __name__ == "__main__":
    main()
