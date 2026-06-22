import torch
import pandas as pd
import os

Ns = [500, 1000, 2000]
propensity_models = ['logistic', 'truncated_logistic', 'truncated_adv']

rows = []

for N in Ns:
    for model_name in propensity_models:
        file_suffix = f"N:{N}_{model_name}"

        theta_path = f"results/{file_suffix}_pred_theta_OLS.pt"
        sig_path   = f"results/{file_suffix}_pred_sig_OLS.pt"
        att_path   = f"results/{file_suffix}_ATT_calculations.pt"

        if not all(os.path.exists(p) for p in [theta_path, sig_path, att_path]):
            print(f"Missing results for N={N}, model={model_name} — skipping")
            continue

        theta = torch.load(theta_path, weights_only=False).squeeze()
        sig   = torch.load(sig_path,   weights_only=False).squeeze()
        ATT   = torch.load(att_path,   weights_only=False)["ATT"]

        bias     = (theta - ATT).mean().item()
        rmse     = ((theta - ATT) ** 2).mean().sqrt().item()
        ci_low   = theta - 1.96 * sig
        ci_high  = theta + 1.96 * sig
        coverage = ((ci_low <= ATT) & (ATT <= ci_high)).float().mean().item()
        interval_length = (2 * 1.96 * sig).mean().item()

        rows.append({
            "N":               N,
            "model":           model_name,
            "true_ATT":        round(ATT.item(), 4),
            "mean_est":        round(theta.mean().item(), 4),
            "bias":            round(bias, 4),
            "rmse":            round(rmse, 4),
            "coverage":        round(coverage, 4),
            "interval_length": round(interval_length, 4),
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results/summary_ols.csv", index=False)
print("\nSaved to results/summary_ols.csv")
