# Parameters

from utils.dynamicRieszFunctions import estimateDynamicRiesz_all
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiDLinear import estimateDiDLinear
import torch
import pandas as pd
import time
from torch.distributions import Normal
from utils.dgp import DiD_DGP
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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

        ATT_calculations = dgp.simulate_ATT(n=100000000)
        torch.save(ATT_calculations, f"results/{file_suffix}_ATT_calculations.pt")

        pred_theta = torch.zeros(tmax, 5)
        pred_sig = torch.zeros(tmax, 5)


        for t in tqdm(range(tmax)):
            torch.manual_seed(t)
            X1_sub = X1[t*N:(t+1)*N, :]
            X2_sub = X2[t*N:(t+1)*N, :]
            Y1_sub = Y1[t*N:(t+1)*N].view(-1, 1)
            Y2_sub = Y2[t*N:(t+1)*N].view(-1, 1)
            D_sub = D[t*N:(t+1)*N].view(-1, 1)
            Z_sub = Z[t*N:(t+1)*N, :]

            pred_theta[t, 1], pred_sig[t, 1] = estimateDiDLinear(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed = t)

            pred_theta[t,2], pred_sig[t,2], *_ = estimateDynamicRiesz( Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, folds,
                                                                            method_a = "LASSO", lasso_a_settings = lasso_a_settings,
                                                                                method_f = "LASSO", lasso_f_settings = lasso_f_settings,
                                                                                seed = t)
            pred_theta[t, 3], pred_sig[t, 3],*_ = estimateDynamicRiesz(
                Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, folds,
                method_a="RF", rf_a_settings=rf_a_settings,
                method_f="RF", rf_f_settings=rf_f_settings,
                seed = t)
            pred_theta[t,4], pred_sig[t,4], *_ = estimateDynamicRiesz(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, folds,
                                                                            method_a = "Net", net_a_settings = net_a_settings,
                                                                                method_f = "Net", net_f_settings = net_f_settings, seed=t)
                

        torch.save(pred_theta, f"results/{file_suffix}_pred_theta.pt")
        torch.save(pred_sig, f"results/{file_suffix}_pred_sig.pt")


        rmse_linear = torch.sqrt(torch.mean((pred_theta[:, 1] - ATT_calculations["ATT"])**2))
        sd_linear = torch.mean(pred_sig[:, 1])

        rmse_lasso = torch.sqrt(torch.mean((pred_theta[:, 2] - ATT_calculations["ATT"])**2))
        sd_lasso = torch.mean(pred_sig[:, 2])

        rmse_rf = torch.sqrt(torch.mean((pred_theta[:, 3] - ATT_calculations["ATT"])**2))
        sd_rf = torch.mean(pred_sig[:, 3])

        rmse_net = torch.sqrt(torch.mean((pred_theta[:, 4] - ATT_calculations["ATT"])**2))
        sd_net = torch.mean(pred_sig[:, 4])

        # Print summary
        print(f"Results for {model_name} with n: {N}:")
        print(f"  RMSE Linear:       {rmse_linear:.4f}, SD: {sd_linear:.4f}")
        print(f"  RMSE LASSO Riesz:  {rmse_lasso:.4f}, SD: {sd_lasso:.4f}")
        print(f"  RMSE RF Riesz:     {rmse_rf:.4f}, SD: {sd_rf:.4f}")
        print(f"  RMSE Net Riesz:    {rmse_net:.4f}, SD: {sd_net:.4f}")
