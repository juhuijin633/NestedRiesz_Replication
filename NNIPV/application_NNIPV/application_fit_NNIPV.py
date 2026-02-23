
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np



river_data = pd.read_csv("data/river_data.csv") # experimental
other_data = pd.read_csv("data/others_data.csv") # observational

# Define pretreatment variables
pretreat_vars = (
    [f"paid{i}" for i in range(1, 5)] +         # 4 lagged values for aid
    [f"tcpp{i}" for i in range(1, 11)] +        # 10 lagged values for employment
    [f"tcprn{i}" for i in range(1, 11)]         # 10 lagged values for earnings
)

# List of used covariates
covariates = [
    "xsexf", "xhsdip", "xchld05", "single",
    "grd1720", "grade16", "grd1315", "grade12", "grde911", "white",
    "hisp", "black", "age"
] + pretreat_vars



def create_dataset(quarters, application):
    if application not in ["earn", "employ"]:
        raise ValueError
    Y_obsorved = other_data[f"Y_{application}"]
    Y_experimental = river_data[f"Y_{application}"] # not used in fitting, only to evaluate
    if application == "employ":
        s_colums = (
            [f"{application}{i}" for i in range(1, quarters+ 1)] +
            [f"aid{i}" for i in range(1, quarters+ 1)] +
            [f"earn{i}" for i in range(1, quarters + 1)]
        )
    elif application == "earn":
        s_colums = (
            [f"{application}{i}" for i in range(1, quarters+ 1)] +
            [f"aid{i}" for i in range(1, quarters + 1)]
        )

    S_obs = other_data[s_colums]
    S_exp = river_data[s_colums]
    D_exp = river_data["e"]
    D_obs = other_data["e"]
    X_obs = other_data[covariates]
    X_exp = river_data[covariates]
    Y_all = pd.concat([Y_obsorved, Y_experimental], axis=0).reset_index(drop=True)
    X_all = pd.concat([X_obs, X_exp], axis=0).reset_index(drop=True)
    S_all = pd.concat([S_obs, S_exp], axis=0).reset_index(drop=True)
    D_all = pd.concat([D_obs, D_exp], axis=0).reset_index(drop=True)
    G_all = pd.concat([
        pd.Series(np.ones(len(D_obs))),
        pd.Series(np.zeros(len(D_exp)))
    ], axis=0).reset_index(drop=True)

    Y_all_torch = torch.tensor(Y_all.values, dtype=torch.float64).view(-1, 1)
    X_all_torch = torch.tensor(X_all.values, dtype=torch.float64)
    S_all_torch = torch.tensor(S_all.values, dtype=torch.float64)
    D_all_torch = torch.tensor(D_all.values, dtype=torch.float64).view(-1, 1)
    G_all_torch = torch.tensor(G_all.values, dtype=torch.float64).view(-1, 1)
    return {"Y_obsorved": Y_obsorved,"Y_experimental": Y_experimental, "S_obs": S_obs, "S_exp":S_exp,
            "D_exp": D_exp, "D_ops":D_obs, "X_obs": X_obs, "X_exp": X_exp, "Y_all" : Y_all_torch, "X_all": X_all_torch, "S_all": S_all_torch, "D_all": D_all_torch, "G_all": G_all_torch,
            "names_x":covariates, "names_z":s_colums }



q = 6
application = "employ"
ds = create_dataset(q, application)
Y_all = ds["Y_all"]
X_all = ds["X_all"]
S_all = ds["S_all"]
D_all = ds["D_all"]
G_all = ds["G_all"]

D_changed = D_all.clone()
Y_changed = Y_all.clone()

###################
D_changed[G_all.bool()] = 0
Y_changed[(1 - G_all).bool()] = 0



# Fit on Y_changed, D_changed, X_all, G_all, S_all


