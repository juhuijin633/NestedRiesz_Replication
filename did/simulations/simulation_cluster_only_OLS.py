import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Parameters

from utils.dynamicRieszFunctions import estimateDynamicRiesz_all
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiDLinear import estimateDiDLinear
from utils.estimateDiD_OLS import estimateDiD_OLS
import torch
import pandas as pd
import time
from torch.distributions import Normal
from utils.dgp import DiD_DGP
from tqdm import tqdm

Ns = [500, 1000, 2000]
tmax = 100
dimX = 3
dimZ = 2
seed = 123 # this seed is for the DGP
folds = 5
# Bounds (only for truncated distributions)
lower = 0.30
upper = 0.70


lasso_cv_settings = {
    'b_degree' : 1,
    'cv_folds' : folds,
    'random_state' : 42
}

lasso_a_settings = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

lasso_f_settings = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' :  "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

rf_a_settings = {
    'poly_degree' : 0,
    'l2' : 0,
    'n_estimators' : 100,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 4,
    'n_jobs' : 1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}
rf_f_settings = {
    'poly_degree' : 1, # 1 or 2?
    'l2' : 0,
    'n_estimators' : 100,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 4,
    'n_jobs' : 1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}

net_a_settings = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}

net_f_settings = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}






def logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))


def truncated_logistic(x):
    return lower + (upper - lower) * logistic(x)


def truncated_adv(x):
    return lower + (upper - lower) * (x > 0)


propensity_models = {
    'logistic': logistic,
    'truncated_logistic': truncated_logistic,
    'truncated_adv': truncated_adv
}

for model_name, prop_func in propensity_models.items():
    print(f"\nSimulating ATT for propensity model: {model_name}")
    dgp = DiD_DGP(dim_X=dimX, dim_Z=dimZ,
                    alpha_1 = 1,
                    gamma_1=torch.ones(dimZ),
                    gamma_2=torch.ones(dimZ),
                    g = prop_func,
                    beta_2=2,
                    delta_3= 2
                    )
    ATT_calculations = dgp.simulate_ATT(n=100000000)
    for N in Ns:
        torch.save(ATT_calculations, f"results/N:{N}_{model_name}_ATT_calculations.pt")

for N in tqdm(Ns):
    for model_name, prop_func in propensity_models.items():
        print(f"\nRunning with propensity model: {model_name}")
        file_suffix = f"N:{N}_{model_name}"
        dgp = DiD_DGP(dim_X=dimX, dim_Z=dimZ,
                        alpha_1 = 1,
                    gamma_1=torch.ones(dimZ),
                        gamma_2=torch.ones(dimZ),
                        g = prop_func,
                        beta_2=2,
                        delta_3= 2
                        )

        data = dgp.generate(n=N*tmax, seed=seed)
        X1, X2 = data['X1'], data['X2']
        Y1, Y2 = data['Y1'], data['Y2']
        Z, D = data['Z'], data['D']

        pred_theta = torch.zeros(tmax, 1)
        pred_sig = torch.zeros(tmax, 1)


        for t in tqdm(range(tmax)):
            torch.manual_seed(t)
            X1_sub = X1[t*N:(t+1)*N, :]
            X2_sub = X2[t*N:(t+1)*N, :]
            Y1_sub = Y1[t*N:(t+1)*N].view(-1, 1)
            Y2_sub = Y2[t*N:(t+1)*N].view(-1, 1)
            D_sub = D[t*N:(t+1)*N].view(-1, 1)
            Z_sub = Z[t*N:(t+1)*N, :]

            pred_theta[t], pred_sig[t] = estimateDiD_OLS(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed= t)


                

        torch.save(pred_theta, f"results/{file_suffix}_pred_theta_OLS.pt")
        torch.save(pred_sig, f"results/{file_suffix}_pred_sig_OLS.pt")



