#!/usr/bin/env python3
"""Compute DiD estimates for each effect year."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiD_OLS import estimateDiD_OLS
from utils.hyperparams import (
    FOLDS,
    SEED,
    lasso_a_settings,
    lasso_f_settings,
    net_a_settings,
    net_f_settings,
    rf_a_settings,
    rf_f_settings,
)
from utils.seeding import configure_runtime
from utils.load_minwage_data import MinWageData

configure_runtime()

PROJECT_ROOT = CODE_DIR.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"
PRE_YEAR, TREATMENT_YEAR = 2003, 2004
EFFECT_YEARS = (2004, 2005, 2006, 2007)

STATIC_OLS = "static_ols"
DYNAMIC_OLS = "dynamic_ols"
DYNAMIC_AUTO_LASSO = "dynamic_auto_lasso"
DYNAMIC_AUTO_RF = "dynamic_auto_rf"
DYNAMIC_AUTO_NN = "dynamic_auto_nn"
COMPUTED_METHOD_IDS = (STATIC_OLS, DYNAMIC_OLS, DYNAMIC_AUTO_LASSO, DYNAMIC_AUTO_RF, DYNAMIC_AUTO_NN)

LEGACY_NAME_TO_ID = {
    "OLS YDiff on Z, X1, and D": STATIC_OLS,
    "Caetano": DYNAMIC_OLS,
    "LASSO": DYNAMIC_AUTO_LASSO,
    "RF": DYNAMIC_AUTO_RF,
    "Net": DYNAMIC_AUTO_NN,
}


def _as_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    if hasattr(x, "item"):
        return float(x.item())
    if isinstance(x, str) and x.startswith("tensor("):
        m = re.search(r"tensor\(([-+0-9.eE]+)", x)
        if m:
            return float(m.group(1))
    return float(x)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method_id" not in out.columns:
        out["method_id"] = out["method"].map(LEGACY_NAME_TO_ID)
    out["ATT"] = out["ATT"].map(_as_float)
    out["SE"] = out["SE"].map(_as_float)
    return out[["method_id", "ATT", "SE"]]


def _method_path(year: int, method_id: str) -> Path:
    return INTERMEDIATE_DIR / f"{year}_{method_id}.csv"


def _combined_path(year: int) -> Path:
    return INTERMEDIATE_DIR / f"results_{year}.csv"


def _save_row(method_id: str, att: float, se: float, path: Path) -> None:
    pd.DataFrame([{"method_id": method_id, "ATT": _as_float(att), "SE": _as_float(se)}]).to_csv(path, index=False)


def _compute_static_ols(Y1, Y2, D, Z, X1) -> tuple[float, float]:
    Z_df, X1_df = pd.DataFrame(Z.numpy()), pd.DataFrame(X1.numpy())
    X_cov = sm.add_constant(pd.concat([pd.DataFrame(D.numpy(), columns=["D"]), Z_df, X1_df], axis=1))
    ols = sm.OLS(np.asarray(Y2) - np.asarray(Y1), X_cov).fit(cov_type="HC1")
    return float(ols.params["D"]), float(ols.bse["D"])


def _compute_all(data: dict) -> dict[str, tuple[float, float]]:
    """Run all estimators in one pass (matches notebook dynamic_riesz_results order)."""
    Y1, Y2, Z, D, X1, X2 = data["Y1"], data["Y2"], data["Z"], data["D"], data["X1"], data["X2"]
    n = len(Y1)
    out: dict[str, tuple[float, float]] = {}

    out[STATIC_OLS] = _compute_static_ols(Y1, Y2, D, Z, X1)

    att, se = estimateDiD_OLS(Y1, Y2, D, Z, X1, X2, seed=SEED)
    out[DYNAMIC_OLS] = (_as_float(att), _as_float(se))

    for method_id, key in (
        (DYNAMIC_AUTO_LASSO, "LASSO"),
        (DYNAMIC_AUTO_RF, "RF"),
        (DYNAMIC_AUTO_NN, "Net"),
    ):
        kw: dict = {"seed": SEED}
        if key == "LASSO":
            kw["lasso_a_settings"] = copy.deepcopy(lasso_a_settings)
            kw["lasso_f_settings"] = copy.deepcopy(lasso_f_settings)
        elif key == "RF":
            kw["rf_a_settings"] = copy.deepcopy(rf_a_settings)
            kw["rf_f_settings"] = copy.deepcopy(rf_f_settings)
        else:
            kw["net_a_settings"] = copy.deepcopy(net_a_settings)
            kw["net_f_settings"] = copy.deepcopy(net_f_settings)
        att, std, *_ = estimateDynamicRiesz(
            Y1, Y2, D, Z, X1, X2, FOLDS, method_a=key, method_f=key, **kw
        )
        out[method_id] = (_as_float(att), _as_float(std / np.sqrt(n)))

    return out


def _year_cached(year: int) -> bool:
    return all(_method_path(year, m).exists() for m in COMPUTED_METHOD_IDS)


def _migrate_legacy(year: int) -> None:
    combined = pd.read_csv(_combined_path(year))
    if "method_id" in combined.columns:
        for method_id in COMPUTED_METHOD_IDS:
            row = combined[combined["method_id"] == method_id]
            if not row.empty:
                row = _normalize(row)
                row.to_csv(_method_path(year, method_id), index=False)
        return
    for _, row in combined.iterrows():
        mid = LEGACY_NAME_TO_ID[row["method"]]
        _save_row(mid, row["ATT"], row["SE"], _method_path(year, mid))


def run_year(year: int, force: bool) -> None:
    if _year_cached(year) and not force:
        # Re-normalize in case old files contain tensor(...) strings
        for method_id in COMPUTED_METHOD_IDS:
            path = _method_path(year, method_id)
            df = _normalize(pd.read_csv(path))
            df.to_csv(path, index=False)
        _normalize(pd.concat([pd.read_csv(_method_path(year, m)) for m in COMPUTED_METHOD_IDS])).to_csv(
            _combined_path(year), index=False
        )
        print(f"[{year}] Estimates Complete (cached).")
        return

    data = MinWageData().get_panel(PRE_YEAR, year, treatment_year=TREATMENT_YEAR, baseline_2001=True)
    results = _compute_all(data)

    for method_id, (att, se) in results.items():
        _save_row(method_id, att, se, _method_path(year, method_id))
        print(f"   [{year}] {method_id} Complete.")

    combined = _normalize(pd.concat([pd.read_csv(_method_path(year, m)) for m in COMPUTED_METHOD_IDS]))
    combined.to_csv(_combined_path(year), index=False)
    print(f"[{year}] Estimates Complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute all methods for each year.")
    parser.add_argument("--years", nargs="+", type=int, default=list(EFFECT_YEARS))
    args = parser.parse_args()

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    print("2. Calc Estimates")
    for year in args.years:
        if _combined_path(year).exists() and not _year_cached(year):
            _migrate_legacy(year)
        run_year(year, args.force)


if __name__ == "__main__":
    main()
