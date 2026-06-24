#!/usr/bin/env python3
"""Build plot tables and figures from cached DiD estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"
RESULTS_DIR = PROJECT_ROOT / "results"

# Plot order and x-axis labels (Static bracket, then Dynamic bracket)
PLOT_ORDER = (
    "static_ols", "static_manual_linear", "static_auto_rf", "static_auto_lasso", "static_auto_nn",
    "dynamic_ols", "dynamic_auto_lasso", "dynamic_auto_rf", "dynamic_auto_nn",
)
DISPLAY_LABELS = {
    "static_ols": "OLS", "static_manual_linear": "Manual-Linear",
    "static_auto_rf": "Auto-RF", "static_auto_lasso": "Auto-Lasso", "static_auto_nn": "Auto-NN",
    "dynamic_ols": "OLS", "dynamic_auto_lasso": "Auto-Lasso",
    "dynamic_auto_rf": "Auto-RF", "dynamic_auto_nn": "Auto-NN",
}
N_STATIC = 5

# Hardcoded static benchmarks (not computed in pipeline)
STATIC_BENCHMARKS = {
    2004: {"static_manual_linear": (-0.0303, 0.0225), "static_auto_rf": (-0.022, 0.019), "static_auto_lasso": (-0.024, 0.020), "static_auto_nn": (-0.022, 0.019)},
    2005: {"static_manual_linear": (-0.0247, 0.0217), "static_auto_rf": (-0.049, 0.020), "static_auto_lasso": (-0.045, 0.021), "static_auto_nn": (-0.046, 0.020)},
    2006: {"static_manual_linear": (-0.0497, 0.0212), "static_auto_rf": (-0.051, 0.020), "static_auto_lasso": (-0.052, 0.021), "static_auto_nn": (-0.052, 0.020)},
    2007: {"static_manual_linear": (-0.0709, 0.0232), "static_auto_rf": (-0.064, 0.023), "static_auto_lasso": (-0.060, 0.025), "static_auto_nn": (-0.064, 0.023)},
}


def _load_computed(year: int) -> pd.DataFrame:
    path = INTERMEDIATE_DIR / f"results_{year}.csv"
    df = pd.read_csv(path)
    if "method_id" not in df.columns:
        legacy = {"OLS YDiff on Z, X1, and D": "static_ols", "Caetano": "dynamic_ols",
                  "LASSO": "dynamic_auto_lasso", "RF": "dynamic_auto_rf", "Net": "dynamic_auto_nn"}
        df["method_id"] = df["method"].map(legacy)
    return df.set_index("method_id")


def build_plot_df(year: int) -> pd.DataFrame:
    computed = _load_computed(year)
    static = pd.concat([
        computed.loc[["static_ols"]].reset_index(),
        pd.DataFrame([{"method_id": k, "ATT": v[0], "SE": v[1]} for k, v in STATIC_BENCHMARKS[year].items()]),
    ])
    static["group"] = "Static"
    dynamic = computed.loc[[m for m in PLOT_ORDER if m.startswith("dynamic_")]].reset_index()
    dynamic["group"] = "Dynamic"
    plot_df = pd.concat([static, dynamic], ignore_index=True).set_index("method_id").reindex(PLOT_ORDER).reset_index()
    plot_df["label"] = plot_df["method_id"].map(DISPLAY_LABELS)
    return plot_df


def _point_color(method_id: str) -> str:
    if method_id in ("static_ols", "dynamic_ols"):
        return "black"
    if method_id == "static_manual_linear":
        return "lightblue"
    return "green"


def plot_year(year: int, plot_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8.5))
    for xi, (_, row) in enumerate(plot_df.iterrows()):
        ax.errorbar(xi, row["ATT"], yerr=1.96 * row["SE"], fmt="o", capsize=5, markersize=9,
                    color=_point_color(row["method_id"]), ecolor="gray", elinewidth=2)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=20)
    ax.set_xlabel("Estimators", fontsize=26, labelpad=45)
    ax.set_ylabel("Effect on Employment", fontsize=26)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylim(-0.13, 0.02)

    trans = ax.get_xaxis_transform()
    for x0, x1, label, y in [(0, N_STATIC - 1, "Static", -0.40), (N_STATIC, len(plot_df) - 1, "Dynamic", -0.40)]:
        ax.plot([x0, x0, x1, x1], [y + 0.04, y, y, y + 0.04], transform=trans, clip_on=False, color="black")
        ax.text((x0 + x1) / 2, y - 0.03, label, transform=trans, ha="center", va="top", fontsize=20, clip_on=False)

    fig.subplots_adjust(bottom=0.40, left=0.18)
    out_path = RESULTS_DIR / f"att_estimates_{year}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2004, 2005, 2006, 2007])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("3. Tables & Figures")
    for year in args.years:
        plot_df = build_plot_df(year)
        plot_df.to_csv(RESULTS_DIR / f"plot_table_{year}.csv", index=False)
        fig_path = plot_year(year, plot_df)
        print(f"   [{year}] Complete — {fig_path.name}")


if __name__ == "__main__":
    main()
