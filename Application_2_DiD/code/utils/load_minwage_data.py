"""Load min-wage data and build DiD tensors (Y1, Y2, D, Z, X1, X2) for one year pair."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

DATA_URL = "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/minwage_data.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "minwage_data.csv"


class MinWageData:
    """Min-wage panel from Causal AI Book / Card-Dickson-Katz application."""

    def __init__(self) -> None:
        if RAW_DATA_PATH.exists():
            self.data = pd.read_csv(RAW_DATA_PATH, index_col=0)
        else:
            self.data = pd.read_csv(DATA_URL, index_col=0)

    def get_panel(
        self,
        pre_treatment_year: int,
        effect_year: int,
        treatment_year: int | None = None,
        baseline_2001: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return pre/post outcomes, treatment, instruments Z, and covariates X1/X2."""
        if treatment_year is None:
            treatment_year = effect_year

        g_filter = (
            (self.data["G"] == 0)
            | (self.data["G"] > effect_year)
            | (self.data["G"] == treatment_year)
        )
        g_filtered = self.data[g_filter]
        filtered = g_filtered[
            (g_filtered["year"] == pre_treatment_year) | (g_filtered["year"] == effect_year)
        ]

        Y1 = torch.tensor(
            filtered[filtered["year"] == pre_treatment_year]["lemp"].values,
            dtype=torch.float32,
        ).view(-1, 1)
        Y2 = torch.tensor(
            filtered[filtered["year"] == effect_year]["lemp"].values,
            dtype=torch.float32,
        ).view(-1, 1)

        drop_cols = [
            "lemp", "treated", "state_name", "FIPS", "quarter", "censusdiv",
            "region", "ever_treated", "id", "state_mw", "fed_mw", "G",
            "countyreal", "emp0A01_BS", "pop", "annual_avg_pay",
        ]
        X = filtered.drop(columns=drop_cols).copy()
        X1 = torch.tensor(
            X[X["year"] == pre_treatment_year].drop(columns=["year"]).values,
            dtype=torch.float32,
        )
        X2 = torch.tensor(
            X[X["year"] == effect_year].drop(columns=["year"]).values,
            dtype=torch.float32,
        )

        post = filtered[filtered["year"] == effect_year]
        D = torch.tensor(
            (post["G"] == treatment_year).astype(float).values.reshape(-1, 1),
            dtype=torch.float32,
        )

        Z = post[["region"]].reset_index(drop=True)
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        Z = pd.DataFrame(encoder.fit_transform(Z), columns=encoder.get_feature_names_out())
        if baseline_2001:
            baselines = (
                g_filtered[g_filtered["year"] == 2001][["lemp", "lpop", "lavg_pay"]]
                .reset_index(drop=True)
                .add_prefix("2001_")
            )
            assert len(Z) == len(baselines)
            Z = Z.merge(baselines, left_index=True, right_index=True)

        return {"X1": X1, "X2": X2, "D": D, "Y1": Y1, "Y2": Y2, "Z": torch.tensor(Z.values, dtype=torch.float32)}
