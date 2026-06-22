import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Quick smoke test: imports all utils and runs one iteration of each estimator
with tiny N so you can catch errors before submitting to SLURM.

Run from the DiD/ directory:
    python smoke_test.py
"""
import torch
import traceback

N = 200
dimX = 3
dimZ = 2
folds = 2

print("--- Imports ---")
from utils.dgp import DiD_DGP
from utils.estimateDiD_OLS import estimateDiD_OLS
from utils.estimateDiDLinear import estimateDiDLinear
from utils.dynamicRieszFunctions import estimateDynamicRiesz
print("OK")

print("\n--- Generating data ---")
def logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))

dgp = DiD_DGP(dim_X=dimX, dim_Z=dimZ,
               alpha_1=1,
               gamma_1=torch.ones(dimZ),
               gamma_2=torch.ones(dimZ),
               g=logistic,
               beta_2=2,
               delta_3=2)
data = dgp.generate(n=N, seed=0)
X1, X2 = data['X1'], data['X2']
Y1, Y2 = data['Y1'], data['Y2']
Z, D = data['Z'], data['D']
Y1 = Y1.view(-1, 1)
Y2 = Y2.view(-1, 1)
D = D.view(-1, 1)
print(f"X1: {X1.shape}, Y1: {Y1.shape}, D: {D.shape}, Z: {Z.shape}")

estimators = {
    "OLS":        lambda: estimateDiD_OLS(Y1, Y2, D, Z, X1, X2, seed=0),
    "Linear_old": lambda: estimateDiDLinear(Y1, Y2, D, Z, X1, X2, seed=0),
    "LASSO":      lambda: estimateDynamicRiesz(Y1, Y2, D, Z, X1, X2, folds,
                      method_a="LASSO", method_f="LASSO")[:2],
    "RF":         lambda: estimateDynamicRiesz(Y1, Y2, D, Z, X1, X2, folds,
                      method_a="RF", method_f="RF")[:2],
}

print("\n--- Running estimators ---")
all_ok = True
for name, fn in estimators.items():
    try:
        theta, sigma = fn()
        print(f"  {name:<12} theta={theta.item():.4f}  sigma={sigma.item():.4f}  OK")
    except Exception:
        print(f"  {name:<12} FAILED")
        traceback.print_exc()
        all_ok = False

print("\n" + ("All checks passed." if all_ok else "Some checks FAILED — see above."))
