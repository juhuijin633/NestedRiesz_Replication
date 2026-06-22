import torch
import pandas as pd
import os
import math

Ns = [500, 1000, 2000]
propensity_models = ['logistic', 'truncated_logistic', 'truncated_adv']
method_names = ['OLS', 'Linear_old', 'LASSO', 'RF', 'Net']

rows = []

for N in Ns:
    for model_name in propensity_models:
        file_suffix = f"final_N:{N}_{model_name}"

        theta_path = f"results/{file_suffix}_pred_theta.pt"
        sig_path = f"results/{file_suffix}_pred_sig.pt"
        att_path = f"results/{file_suffix}_ATT_calculations.pt"

        if not all(os.path.exists(p) for p in [theta_path, sig_path, att_path]):
            print(f"Missing results for N={N}, model={model_name} — skipping")
            continue

        pred_theta = torch.load(theta_path)
        pred_sig = torch.load(sig_path)
        ATT = torch.load(att_path)["ATT"]

        for k, name in enumerate(method_names):
            theta_k = pred_theta[:, k]
            sig_k = pred_sig[:, k]

            # sig_k is the SD of the influence function, not the SE
            se_k = sig_k / math.sqrt(N)

            bias = torch.mean(theta_k - ATT).item()
            rmse = torch.sqrt(torch.mean((theta_k - ATT) ** 2)).item()
            ci_low = theta_k - 1.96 * se_k
            ci_high = theta_k + 1.96 * se_k
            coverage = torch.mean(((ci_low <= ATT) & (ATT <= ci_high)).float()).item()
            interval_length = torch.mean(2 * 1.96 * se_k).item()
            rows.append({
                "N": N,
                "model": model_name,
                "method": name,
                "bias": round(bias, 4),
                "rmse": round(rmse, 4),
                "coverage": round(coverage, 4),
                "interval_length": round(interval_length, 4),
            })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results/summary.csv", index=False)
print("\nSaved to results/summary.csv")