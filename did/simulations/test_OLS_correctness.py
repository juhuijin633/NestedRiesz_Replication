import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Quick correctness check for estimateDiD_OLS.
Uses reduced N/tmax/ATT-sim size to run fast locally.
Reports bias, RMSE, and 95% CI coverage vs true ATT.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.estimateDiD_OLS import estimateDiD_OLS
from utils.dgp import DiD_DGP

# --- Reduced parameters for quick check ---
Ns     = [500, 1000]
tmax   = 20          # was 100
ATT_n  = 500_000     # was 100_000_000
seed   = 123
dimX   = 3
dimZ   = 2
lower, upper = 0.30, 0.70


def logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))

def truncated_logistic(x):
    return lower + (upper - lower) * logistic(x)

def truncated_adv(x):
    return lower + (upper - lower) * (x > 0)

propensity_models = {
    'logistic':           logistic,
    'truncated_logistic': truncated_logistic,
    'truncated_adv':      truncated_adv,
}

print(f"{'Model':<22} {'N':>6}  {'True ATT':>10}  {'Mean est':>10}  "
      f"{'Bias':>8}  {'RMSE':>8}  {'Cover 95%':>10}")
print("-" * 85)

for model_name, prop_func in propensity_models.items():
    dgp = DiD_DGP(
        dim_X=dimX, dim_Z=dimZ,
        alpha_1=1,
        gamma_1=torch.ones(dimZ),
        gamma_2=torch.ones(dimZ),
        g=prop_func,
        beta_2=2,
        delta_3=2,
    )
    true_ATT = dgp.simulate_ATT(n=ATT_n)["ATT"].item()

    for N in Ns:
        data = dgp.generate(n=N * tmax, seed=seed)
        X1, X2 = data['X1'], data['X2']
        Y1, Y2 = data['Y1'], data['Y2']
        Z,  D  = data['Z'],  data['D']

        thetas = torch.zeros(tmax)
        sigs   = torch.zeros(tmax)

        for t in range(tmax):
            torch.manual_seed(t)
            sl = slice(t * N, (t + 1) * N)
            theta_t, sig_t = estimateDiD_OLS(
                Y1[sl].view(-1, 1), Y2[sl].view(-1, 1),
                D[sl].view(-1, 1),
                Z[sl],
                X1[sl], X2[sl],
                seed=t,
            )
            thetas[t] = theta_t
            sigs[t]   = sig_t

        bias     = (thetas - true_ATT).mean().item()
        rmse     = ((thetas - true_ATT) ** 2).mean().sqrt().item()
        # 95% CI: theta ± 1.96 * sigma
        covered  = ((thetas - 1.96 * sigs <= true_ATT) &
                    (true_ATT <= thetas + 1.96 * sigs)).float().mean().item()

        print(f"{model_name:<22} {N:>6}  {true_ATT:>10.4f}  {thetas.mean().item():>10.4f}  "
              f"{bias:>8.4f}  {rmse:>8.4f}  {covered:>10.3f}")
