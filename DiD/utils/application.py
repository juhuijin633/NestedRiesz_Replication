import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

class application_data():
    def __init__(self):
        self.data = pd.read_csv("https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/minwage_data.csv", index_col=0)

    def get_data(self, pre_treatment_year, post_treatment_year, treatment_year = None):

        if treatment_year is None:
            treatment_year = post_treatment_year
        # Filter for the relevant years
        filtered_data = self.data[(self.data["year"] == pre_treatment_year) | (self.data["year"] == post_treatment_year)]
        # exclude the units that are already treated in the pre-treatment year and the units that are treated between the treatment year and post-treatment years
        filtered_data = filtered_data[(filtered_data.G == 0) | (filtered_data.G > post_treatment_year) | (filtered_data.G == treatment_year)] 

        # Outcome variables
        Y1 = torch.tensor(
            filtered_data[filtered_data["year"] == pre_treatment_year]["lemp"].values, dtype=torch.float32
        ).view(-1, 1)
        Y2 = torch.tensor(
            filtered_data[filtered_data["year"] == post_treatment_year]["lemp"].values, dtype=torch.float32
        ).view(-1, 1)

        # Covariates 
        X = filtered_data.drop(columns=[
            "lemp", "treated", "state_name", "FIPS", "quarter", "censusdiv",
            "region", "ever_treated", "id", "state_mw", "fed_mw", "G", "countyreal"
        ]).copy()

        X1 = X[X["year"] == pre_treatment_year].drop(columns=["year"]).reset_index(drop=True)
        X2 = X[X["year"] == post_treatment_year].drop(columns=["year"]).reset_index(drop=True)
        print("Changing covariates: ", X1.columns.to_list())    
        X1 = torch.tensor(X1.values, dtype=torch.float32)
        X2 = torch.tensor(X2.values, dtype=torch.float32)

        post_treatment_data = filtered_data[filtered_data["year"] == post_treatment_year]
        D = torch.tensor(
            (post_treatment_data["G"] == treatment_year).astype(float).values.reshape(-1, 1),
            dtype=torch.float32
        )


        # Z variables
        Z = filtered_data[filtered_data["year"] == post_treatment_year][["region", "censusdiv"]].reset_index(drop=True)
        # One-hot encoding for categorical variables
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        Z_encoded = encoder.fit_transform(Z)
        Z = pd.DataFrame(Z_encoded, columns=encoder.get_feature_names_out())
        print("Z variables: ", Z.columns.to_list())
        Z = torch.tensor(Z.values, dtype=torch.float32)


        return {
            "X1": X1,
            "X2": X2,
            "D": D,
            "Y1": Y1,
            "Y2": Y2,
            "Z": Z
        }
