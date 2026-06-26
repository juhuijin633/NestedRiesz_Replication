"""Single source of truth for Simulation_1_TimeTreatment design and table labels.

Array layout matches time_varying_treatment/submit.sh (6000 tasks).
Method labels match paper tables (e.g. Table F.3): Oracle, Manual-Lasso, Auto-*.
Internal estimator code still uses Bradic / LASSO / RF / Net — only CSV output is relabeled.
"""

from __future__ import annotations

NS = [500, 1000, 2000]
TMAX = 500
DGP_SEED_OFFSET = 123  # torch.manual_seed(123 + t) before each DGP draw

# Column order in aggregated metrics (matches upstream collect_sim_results.py index).
METHOD_LABELS = [
    "Oracle",
    "Manual-Lasso",  # upstream: Bradic (estimateBradic / D-DRL)
    "Auto-Lasso",    # upstream: LASSO-LASSO
    "Auto-RF",       # upstream: RF-RF
    "Auto-NN",       # upstream: Net-Net
]

# Legacy names in upstream .pt files / collect_sim_results.py (for cross-reference only).
UPSTREAM_METHOD_LABELS = ["Oracle", "Bradic", "LASSO-LASSO", "RF-RF", "Net-Net"]

CONFIGS = [
    {
        "id": "linear_truncated_logistic",
        "dgp": "linear",
        "func": "truncated_logistic",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Linear DGP + truncated logistic [0.1, 0.9]",
    },
    {
        "id": "nonlinear_truncated_adv",
        "dgp": "nonlinear",
        "func": "truncated_adv",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Nonlinear DGP + truncated adversarial [0.1, 0.9]",
    },
    {
        "id": "linear_truncated_adv",
        "dgp": "linear",
        "func": "truncated_adv",
        "lower": 0.10,
        "upper": 0.90,
        "label": "Linear DGP + truncated adversarial [0.1, 0.9]",
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
