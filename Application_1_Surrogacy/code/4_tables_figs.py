"""Build summary tables and figures.

Reads intermediate estimate CSVs from data/processed/:
    auto_{outcome}_estimates.csv    — Auto-Lasso, Auto-RF, Auto-NN
    manual_{outcome}_estimates.csv  — Manual-Lasso, Manual-RF, Manual-NN

Writes final outputs to results/:
    {outcome}_estimates.csv  — all estimators + benchmarks (combined)
    {outcome}_figure.png
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data" / "processed"
RESULTS_DIR = APP_DIR / "results"

QUARTER = 6  # quarters of surrogate outcomes used in estimates
Z_SCORE = 1.96
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
CI_LINEWIDTH = 4
MARKER_SIZE = 14

COLOR_BASELINE = "black"
COLOR_MANUAL = "skyblue"
COLOR_AUTO = "green"

MANUAL_ORDER = ["Manual-Lasso", "Manual-RF", "Manual-NN"]
AUTO_ORDER = ["Auto-Lasso", "Auto-RF", "Auto-NN"]

OUTCOME_LABELS = {"earn": "earnings", "employ": "employment"}

BENCHMARKS = {
    "earn": {"Benchmark": (248.0, 31.5), "Observational": (327.1, 36.6)},
    "employ": {"Benchmark": (0.063, 0.006), "Observational": (0.117, 0.010)},
}
YLABELS = {
    "earn": "Effect on Earnings",
    "employ": "Effect on Employment",
}
PLOT_OPTIONS = {
    "earn": {"exclude_manual_nn_from_scale": False, "yticks": None},
    "employ": {"exclude_manual_nn_from_scale": True, "yticks": [0, 0.04, 0.08, 0.12]},
}


def build_summary(application: str, quarter: int) -> pd.DataFrame:
    auto = pd.read_csv(DATA_DIR / f"auto_{application}_estimates.csv")
    manual = pd.read_csv(DATA_DIR / f"manual_{application}_estimates.csv")
    rows = []

    for name, (point, se) in BENCHMARKS[application].items():
        rows.append({
            "outcome": application,
            "quarter": quarter,
            "estimator": name,
            "group": "baseline",
            "point_estimate": point,
            "se": se,
            "ci_lower": point - Z_SCORE * se,
            "ci_upper": point + Z_SCORE * se,
        })

    return pd.concat([pd.DataFrame(rows), manual, auto], ignore_index=True)


def plot_summary(
    summary: pd.DataFrame,
    *,
    ylabel: str,
    output_path: Path,
    exclude_manual_nn_from_scale: bool = False,
    yticks: list[float] | None = None,
) -> None:
    baseline = summary[summary["group"] == "baseline"]
    manual = summary[summary["group"] == "manual"].set_index("estimator").loc[MANUAL_ORDER]
    auto = summary[summary["group"] == "auto"].set_index("estimator").loc[AUTO_ORDER]

    fig, ax = plt.subplots(figsize=(14, 6))

    for _, row in baseline.iterrows():
        point, lo, hi = row["point_estimate"], row["ci_lower"], row["ci_upper"]
        ax.errorbar(
            x=[row["estimator"]], y=[point],
            yerr=[[point - lo], [hi - point]],
            fmt="o", capsize=5, markersize=MARKER_SIZE,
            color=COLOR_BASELINE, ecolor="gray", elinewidth=CI_LINEWIDTH,
        )

    ax.errorbar(
        x=manual.index.tolist(), y=manual["point_estimate"].to_numpy(),
        yerr=[
            manual["point_estimate"].to_numpy() - manual["ci_lower"].to_numpy(),
            manual["ci_upper"].to_numpy() - manual["point_estimate"].to_numpy(),
        ],
        fmt="o", capsize=5, markersize=MARKER_SIZE,
        color=COLOR_MANUAL, ecolor="gray", elinewidth=CI_LINEWIDTH,
    )

    ax.errorbar(
        x=auto.index.tolist(), y=auto["point_estimate"].to_numpy(),
        yerr=[
            auto["point_estimate"].to_numpy() - auto["ci_lower"].to_numpy(),
            auto["ci_upper"].to_numpy() - auto["point_estimate"].to_numpy(),
        ],
        fmt="o", capsize=5, markersize=MARKER_SIZE,
        color=COLOR_AUTO, ecolor="gray", elinewidth=CI_LINEWIDTH,
    )

    ax.set_xlabel("Estimators", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, rotation=30)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(True, linestyle="--", alpha=0.6)

    if exclude_manual_nn_from_scale:
        manual_mask = manual.index != "Manual-NN"
        scale_lower = np.concatenate([
            baseline["ci_lower"].to_numpy(),
            manual.loc[manual_mask, "ci_lower"].to_numpy(),
            auto["ci_lower"].to_numpy(),
        ])
        scale_upper = np.concatenate([
            baseline["ci_upper"].to_numpy(),
            manual.loc[manual_mask, "ci_upper"].to_numpy(),
            auto["ci_upper"].to_numpy(),
        ])
    else:
        scale_lower = summary["ci_lower"].to_numpy()
        scale_upper = summary["ci_upper"].to_numpy()

    pad = 0.05 * (scale_upper.max() - scale_lower.min())
    ax.set_ylim(0, scale_upper.max() + pad)

    if yticks is not None:
        ax.set_yticks(yticks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_outcome(application: str, quarter: int) -> None:
    summary = build_summary(application, quarter)
    csv_path = RESULTS_DIR / f"{application}_estimates.csv"
    fig_path = RESULTS_DIR / f"{application}_figure.png"
    plot_opts = PLOT_OPTIONS[application]
    summary.to_csv(csv_path, index=False)
    plot_summary(
        summary,
        ylabel=YLABELS[application],
        output_path=fig_path,
        exclude_manual_nn_from_scale=plot_opts["exclude_manual_nn_from_scale"],
        yticks=plot_opts["yticks"],
    )
    print(f"[{OUTCOME_LABELS[application]}] Tables and Figures Complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--applications", nargs="+", default=["earn", "employ"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for app in args.applications:
        run_outcome(app, QUARTER)


if __name__ == "__main__":
    main()
