import sys
import os
import torch
import time
from utils.dynamicRieszFunctions import estimateDynamicRiesz_all
from utils.dynamicRieszBradic import estimateBradic

# -----------------------------------------------------------------------
# Arguments: iteration index, N, and treatment probability function name
# Usage: python run_sim_ate.py <t> <N> <func_name>
# -----------------------------------------------------------------------
t         = int(sys.argv[1])
N         = int(sys.argv[2])
func_name = sys.argv[3]
if os.path.exists(f'results_ate/N{N}/{func_name}/result_{t}.pt'):
    print(f"Skipping iteration {t}, already done")
    sys.exit(0)
torch.manual_seed(123 + t)

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
dimX1 = 5
dimX2 = 5
folds = 4

# -----------------------------------------------------------------------
# Estimator settings
# -----------------------------------------------------------------------
lasso_a_settings = {
    'estimate'   : True,
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB'       : 0,
    'D_add'      : 0.2,
    'c1'         : 0.1,
    'c2'         : 0.1,
    'tol'        : 1e-5,
    'max_iter'   : 100,
    'b_degree'   : 1,
    'control'    : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}
lasso_f_settings = {
    'estimate'   : True,
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB'       : 0,
    'D_add'      : 0.2,
    'c1'         : 0.1,
    'c2'         : 0.1,
    'tol'        : 1e-5,
    'max_iter'   : 100,
    'b_degree'   : 1,
    'control'    : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}
rf_a_settings = {
    'estimate'                : True,
    'poly_degree'             : 0,
    'l2'                      : 0,
    'n_estimators'            : 100,
    'criterion'               : "mse",
    'max_depth'               : None,
    'min_samples_split'       : 10,
    'min_samples_leaf'        : 5,
    'min_weight_fraction_leaf': 0.,
    'min_var_fraction_leaf'   : None,
    'min_var_leaf_on_val'     : False,
    'max_features'            : "auto",
    'min_impurity_decrease'   : 0.,
    'max_samples'             : .45,
    'min_balancedness_tol'    : .45,
    'honest'                  : True,
    'inference'               : True,
    'fit_intercept'           : True,
    'subforest_size'          : 4,
    'n_jobs'                  : -1,
    'random_state'            : None,
    'verbose'                 : 0,
    'warm_start'              : False
}
rf_f_settings = {
    'estimate'                : True,
    'poly_degree'             : 1,
    'l2'                      : 0,
    'n_estimators'            : 100,
    'criterion'               : "mse",
    'max_depth'               : None,
    'min_samples_split'       : 10,
    'min_samples_leaf'        : 5,
    'min_weight_fraction_leaf': 0.,
    'min_var_fraction_leaf'   : None,
    'min_var_leaf_on_val'     : False,
    'max_features'            : "auto",
    'min_impurity_decrease'   : 0.,
    'max_samples'             : .45,
    'min_balancedness_tol'    : .45,
    'honest'                  : True,
    'inference'               : True,
    'fit_intercept'           : True,
    'subforest_size'          : 4,
    'n_jobs'                  : -1,
    'random_state'            : None,
    'verbose'                 : 0,
    'warm_start'              : False
}
net_a_settings = {
    'estimate'         : True,
    'test_split'       : 0,
    'learner_lr'       : 1e-4,
    'learner_l2'       : 1e-3,
    'learner_l1'       : 0,
    'n_epochs'         : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta'  : 1e-3,
    'bs'               : 64,
    'optimizer'        : 'adam',
    'warm_start'       : False,
    'logger'           : None,
    'model_dir'        : '.',
    'device'           : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden'         : 100,
    'drop_prob'        : 0,
    'degree'           : 2,
    'interaction_only' : True,
    'n_common'         : 200,
    'act_func'         : 'elu'
}
net_f_settings = {
    'estimate'         : True,
    'test_split'       : 0,
    'learner_lr'       : 1e-4,
    'learner_l2'       : 1e-3,
    'learner_l1'       : 0,
    'n_epochs'         : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta'  : 1e-3,
    'bs'               : 64,
    'optimizer'        : 'adam',
    'warm_start'       : False,
    'logger'           : None,
    'model_dir'        : '.',
    'device'           : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden'         : 100,
    'drop_prob'        : 0,
    'degree'           : 2,
    'interaction_only' : True,
    'n_common'         : 200,
    'act_func'         : 'elu'
}

# -----------------------------------------------------------------------
# Treatment probability functions
# -----------------------------------------------------------------------
lower = 0.10
upper = 0.90

def logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))

def truncated_logistic(x):
    return lower + (upper - lower) * logistic(x)

def func_link(x):
    return (x > 0) * torch.abs(x) / (torch.abs(x) + 1) + (x < 0) / (torch.abs(x) + 1)

def truncated_link(x):
    return lower + (upper - lower) * func_link(x)

def truncated_adv(x):
    return lower + (upper - lower) * (x > 0).float()

def double_nonlinear(x):
    return lower + (upper - lower) * ((x < -0.5) + (x < 0.5)).float()

func_map = {
    'logistic'          : logistic,
    'truncated_logistic': truncated_logistic,
    'truncated_link'    : truncated_link,
    'truncated_adv'     : truncated_adv,
    'double_nonlinear'  : double_nonlinear,
}

treatment_probability_func = func_map[func_name]

# -----------------------------------------------------------------------
# DGP: coefficients
# -----------------------------------------------------------------------
beta_pi1_0    = 0
beta_pi1_S1   = torch.tensor([1, 1, 1]      + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_0_0  = 0
beta_pi2_0_S1 = torch.tensor([0.5, 0, -0.5] + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_0_S2 = torch.tensor([0.5, 0, 0.5]  + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_1_0  = 0
beta_pi2_1_S1 = torch.tensor([1, 1, 0]      + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_1_S2 = torch.tensor([1, -1, 0]     + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g0_0     = 1
beta_g0_S1    = torch.tensor([1, 1, -1]     + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g0_S2    = torch.tensor([1, 1, 1]      + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g1_0     = -1
beta_g1_S1    = torch.tensor([-1, 1, -1]    + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g1_S2    = torch.tensor([-1, -1, 1]    + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)

# -----------------------------------------------------------------------
# DGP: generate one dataset of size N
# -----------------------------------------------------------------------
S1    = torch.randn(N, dimX1)
pi1   = treatment_probability_func(beta_pi1_0 + S1 @ beta_pi1_S1).reshape(-1, 1)
A1    = torch.bernoulli(pi1).int().reshape(-1, 1)

delta1 = torch.randn(N, 1)
delta2 = torch.randn(N, dimX2)

S2   = S1 + A1 * (1 + delta1) + delta2
S2_1 = S1 + 1 + delta1 + delta2
S2_0 = S1 + delta2

pi2_0 = treatment_probability_func(beta_pi2_0_0 + S1 @ beta_pi2_0_S1 + S2_0 @ beta_pi2_0_S2)
pi2_1 = treatment_probability_func(beta_pi2_1_0 + S1 @ beta_pi2_1_S1 + S2_1 @ beta_pi2_1_S2)
pi2   = (1 - A1) * pi2_0 + A1 * pi2_1
A2    = torch.bernoulli(pi2).int()

g = (
    (A1 + A2 == 0).float() * (beta_g0_0 + S1 @ beta_g0_S1 + S2 @ beta_g0_S2) +
    (A1 * A2 == 1).float() * (beta_g1_0 + S1 @ beta_g1_S1 + S2 @ beta_g1_S2)
)
Y = g + torch.randn(N, 1)

# -----------------------------------------------------------------------
# True nuisances for E[Y(1,1)] (oracle)
# -----------------------------------------------------------------------
mu2_1 = beta_g1_0 + S1 @ beta_g1_S1 + S2_1 @ beta_g1_S2
mu1_1 = beta_g1_0 + S1 @ (beta_g1_S1 + beta_g1_S2) + beta_g1_S2.sum()
theta1 = beta_g1_0 + beta_g1_S2.sum()

# -----------------------------------------------------------------------
# True nuisances for E[Y(0,0)] (oracle)
# E[S2_0 | S1] = S1  =>  mu1_0 = beta_g0_0 + S1 @ (beta_g0_S1 + beta_g0_S2)
# E[Y(0,0)] = beta_g0_0  (since E[S1]=0, E[S2_0]=0)
# -----------------------------------------------------------------------
mu2_0 = beta_g0_0 + S1 @ beta_g0_S1 + S2_0 @ beta_g0_S2
mu1_0 = beta_g0_0 + S1 @ (beta_g0_S1 + beta_g0_S2)
theta0 = beta_g0_0

ate_true = theta1 - theta0

# -----------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------
X       = torch.hstack((S1, S2))
X_index = torch.tensor([S1.shape[1] - 1, S1.shape[1] + S2.shape[1] - 1])
D       = torch.hstack((A1, A2))
d_ones  = torch.ones(D.shape)
d_zeros = torch.zeros(D.shape)

time0 = time.time()

# Oracle for E[Y(1,1)]
RR2_11 = A1 * A2 / (pi1 * pi2_1)
RR1_11 = A1 / pi1
psi_11 = RR2_11 * Y - (RR1_11 - 1) * mu1_1 - (RR2_11 - RR1_11) * mu2_1

# Oracle for E[Y(0,0)]
RR2_00 = (1 - A1) * (1 - A2) / ((1 - pi1) * (1 - pi2_0))
RR1_00 = (1 - A1) / (1 - pi1)
psi_00 = RR2_00 * Y - (RR1_00 - 1) * mu1_0 - (RR2_00 - RR1_00) * mu2_0

# Oracle ATE
psi_ate_oracle   = psi_11 - psi_00
oracle_ate       = torch.mean(psi_ate_oracle).item()
oracle_ate_sig   = torch.sqrt(torch.mean((psi_ate_oracle - torch.mean(psi_ate_oracle))**2)).item()

# Bradic (estimates E[Y(1,1)] only; ATE not available)
try:
    bradic_result    = estimateBradic(Y, X, D, X_index, folds)
    bradic_theta_11  = float(bradic_result[6])
    bradic_sig_11    = float(torch.sqrt(torch.tensor(float(bradic_result[9]))))
except Exception as e:
    print(f"  Bradic failed: {e}")
    bradic_theta_11 = float('nan')
    bradic_sig_11   = float('nan')

# LASSO-LASSO / RF-RF / Net-Net: estimate E[Y(1,1)] and E[Y(0,0)]
_, _, _, _, _, _, fr_11 = estimateDynamicRiesz_all(
    Y, X, D, d_ones, X_index, folds,
    lasso_a_settings=lasso_a_settings,
    lasso_f_settings=lasso_f_settings,
    rf_a_settings=rf_a_settings,
    rf_f_settings=rf_f_settings,
    net_a_settings=net_a_settings,
    net_f_settings=net_f_settings,
    return_fold_results=True
)
_, _, _, _, _, _, fr_00 = estimateDynamicRiesz_all(
    Y, X, D, d_zeros, X_index, folds,
    lasso_a_settings=lasso_a_settings,
    lasso_f_settings=lasso_f_settings,
    rf_a_settings=rf_a_settings,
    rf_f_settings=rf_f_settings,
    net_a_settings=net_a_settings,
    net_f_settings=net_f_settings,
    return_fold_results=True
)

fr_ate  = fr_11 - fr_00                                   # (N, 3) ATE influence functions
pt_ate  = torch.mean(fr_ate, 0)                           # (3,)
sg_ate  = torch.sqrt(torch.mean((fr_ate - pt_ate)**2, 0)) # (3,)

print(f"[N={N}, {func_name}, ATE] Iteration {t} done in {time.time() - time0:.1f}s")

# -----------------------------------------------------------------------
# Save results (results_ate/N{N}/{func_name}/result_{t}.pt)
# -----------------------------------------------------------------------
os.makedirs(f'results_ate/N{N}/{func_name}', exist_ok=True)
torch.save({
    'iteration'      : t,
    'N'              : N,
    'func_name'      : func_name,
    'ate_true'       : ate_true.item(),
    'oracle_theta'   : oracle_ate,
    'oracle_sig'     : oracle_ate_sig,
    'bradic_theta'   : float('nan'),   # Bradic ATE not available
    'bradic_sig'     : float('nan'),
    'pred_theta'     : pt_ate,         # 3-element: [LASSO-LASSO, RF-RF, Net-Net]
    'pred_sig'       : sg_ate,
}, f'results_ate/N{N}/{func_name}/result_{t}.pt')
