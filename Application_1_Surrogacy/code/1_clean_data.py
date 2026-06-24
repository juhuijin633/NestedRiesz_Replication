"""
Build analysis-ready experimental and observational CSVs from quarterly.mat.

Input:
    data/raw/quarterly.mat

Output:
    data/processed/exp_data.csv   — Riverside (experimental) sample
    data/processed/obs_data.csv   — other counties (observational) sample

Python port of data/data_manipulation.R.

Run from Application_1_Surrogacy/:
    python code/clean_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

CODE_DIR = Path(__file__).resolve().parent
APP_DIR = CODE_DIR.parent
RAW_MAT = APP_DIR / "data" / "raw" / "quarterly.mat"
PROCESSED_DIR = APP_DIR / "data" / "processed"
EXP_DATA_CSV = PROCESSED_DIR / "exp_data.csv"
OBS_DATA_CSV = PROCESSED_DIR / "obs_data.csv"

EMPLOYMENT_COLS = [f"employ{i}" for i in range(1, 37)]
EARN_COLS = [f"earn{i}" for i in range(1, 37)]
AID_COLS = [f"aid{i}" for i in range(1, 37)]
PRETREAT_VARS = (
    [f"paid{i}" for i in range(1, 5)]
    + [f"tcpp{i}" for i in range(1, 11)]
    + [f"tcprn{i}" for i in range(1, 11)]
)
COVARIATES = [
    "xsexf",
    "xhsdip",
    "xchld05",
    "single",
    "grd1720",
    "grade16",
    "grd1315",
    "grade12",
    "grde911",
    "white",
    "hisp",
    "black",
    "age",
    *PRETREAT_VARS,
]
OUTPUT_COLS = [
    "e",
    *COVARIATES,
    *EMPLOYMENT_COLS,
    *EARN_COLS,
    "Y_employ",
    "Y_earn",
    *AID_COLS,
]


def load_quarterly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    mat = scipy.io.loadmat(path)
    columns = {
        key: np.asarray(value).reshape(-1)
        for key, value in mat.items()
        if not key.startswith("__")
    }
    df = pd.DataFrame(columns)

    rename_map = {}
    for col in df.columns:
        if col.startswith("ptcedd"):
            rename_map[col] = col.replace("ptcedd", "employ", 1)
        elif col.startswith("tcedd"):
            rename_map[col] = col.replace("tcedd", "earn", 1)
    df = df.rename(columns=rename_map)

    return df


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Y_employ"] = out[EMPLOYMENT_COLS].mean(axis=1)
    out["Y_earn"] = out[EARN_COLS].mean(axis=1)
    return out[OUTPUT_COLS]


def split_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "river" not in df.columns:
        raise KeyError("Expected 'river' column in quarterly.mat")

    exp_df = build_analysis_frame(df.loc[df["river"] == 1].copy())
    obs_df = build_analysis_frame(df.loc[df["river"] == 0].copy())
    return exp_df.reset_index(drop=True), obs_df.reset_index(drop=True)


def write_outputs(
    exp_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    *,
    exp_path: Path,
    obs_path: Path,
) -> None:
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(exp_path, index=False)
    obs_df.to_csv(obs_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean surrogate application data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_MAT,
        help=f"Path to quarterly.mat (default: {RAW_MAT.relative_to(APP_DIR)})",
    )
    parser.add_argument(
        "--exp-output",
        type=Path,
        default=EXP_DATA_CSV,
        help=f"Experimental CSV output (default: {EXP_DATA_CSV.relative_to(APP_DIR)})",
    )
    parser.add_argument(
        "--obs-output",
        type=Path,
        default=OBS_DATA_CSV,
        help=f"Observational CSV output (default: {OBS_DATA_CSV.relative_to(APP_DIR)})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quarterly = load_quarterly(args.input)
    exp_df, obs_df = split_samples(quarterly)
    write_outputs(exp_df, obs_df, exp_path=args.exp_output, obs_path=args.obs_output)

    print("Wrote exp_data.csv", flush=True)
    print("Wrote obs_data.csv", flush=True)


if __name__ == "__main__":
    main()
