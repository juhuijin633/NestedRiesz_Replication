import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Parameters
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiD_OLS import estimateDiD_OLS
from utils.estimateDiDLinear import estimateDiDLinear
import torch
import pandas as pd
import time
from torch.distributions import Normal

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class DiD_DGP:
    def __init__(self, dim_X=3, dim_Z = 3, beta_1=2.0, beta_2=1.5, c_1=1.0, delta_1 = 1 , delta_2 =1 , delta_3=1 , alpha_1 =1, gamma_1 = None, gamma_2= None, g  = None, gamma_10 = 0,gamma_11=0 ):
        self.dim_X = dim_X
        self.dim_Z = dim_Z
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha_1 = alpha_1
        self.c_1 = c_1
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_3 = delta_3
        self.gamma_1 = gamma_1 # dimension dim_Z
        self.gamma_2 = gamma_2 # dimension dim_Z
        self.gamma_10 = gamma_10
        self.gamma_11 = gamma_11
        if gamma_1 is None: # if not specified, Z does not affect the propensity score
            self.gamma_1 = torch.zeros(dim_Z)
        if gamma_2 is None: # if not specified, Z does not affect the outcome
            self.gamma_2 = torch.zeros(dim_Z)
        self.g = g # this function specifies the propensity model
        self.ATT = None
       

    def generate(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        X1 = torch.randn(n, self.dim_X)
        X11 = X1[:, 0]
        eta = torch.randn(n)
        eps_x = torch.randn(n, self.dim_X)
        Z = torch.randn(n, self.dim_Z)
        Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + torch.randn(n) 
        prob_D = self.g(self.delta_2* X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta) # propensity score
        D = torch.bernoulli(prob_D)
        I_x11 = (X11 > 0)
        X2 = ((D*self.gamma_11 + (1-D)*self.gamma_10)*I_x11).unsqueeze(1)  +X1 + torch.ones(n, self.dim_X) *  (D *(1 + eta)).unsqueeze(1)  + eps_x
        X21 = X2[:, 0]
        Y2 = Y1 + self.c_1 * D.squeeze() + self.beta_2 * (X21 > 0).float() + self.delta_3* D.squeeze() * X21 + torch.randn(n)
        return {
            "X1": X1,
            "X2": X2,
            "Y1": Y1,
            "Y2": Y2,
            "D": D,
            "Z": Z,
        }    
    def simulate_ATT(self, n = 1000000):
        X1 = torch.randn(n, self.dim_X)
        X11 = X1[:, 0]
        I_x11 = (X11 > 0)

        eta = torch.randn(n)
        eps_x = torch.randn(n, self.dim_X)
        Z = torch.randn(n, self.dim_Z)
        Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + torch.randn(n) 
        prob_D = self.g(self.delta_2* X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta) 
        D = torch.bernoulli(prob_D)
        X2_0 =  (self.gamma_10*I_x11).unsqueeze(1) + X1 + eps_x 
        X2_1 =  (self.gamma_11*I_x11).unsqueeze(1)  + X1 + torch.ones(n, self.dim_X) *  (1 + eta).unsqueeze(1)  + eps_x
        X21_0 = X2_0[:, 0]
        X21_1 = X2_1[:, 0]
        # Calculate ATT
        E_X1D1 = torch.mean(X11[D == 1])  # E[X11 | D = 1]
        E_ETAD1 = torch.mean(eta[D == 1]) # E[ETA | D = 1]
        I_X21D1 = (X21_1 >0).float()
        I_X20D1 = (X21_0 > 0).float()
        E_X21D1 = torch.mean(I_X21D1[D == 1]) # P(X21(1) > 0 | D = 1)
        E_X20D1 = torch.mean(I_X20D1[D == 1])  # P(X21(0) > 0 | D = 1)
        E_X1D1_P = torch.mean(I_x11[D == 1].float()) # P(X11>0| D =1 )
        ATT = self.c_1 + self.beta_2 * E_X21D1 - self.beta_2 * E_X20D1 + self.delta_3 * E_X1D1 + self.delta_3* E_ETAD1 + self.delta_3*1 +self.delta_3*self.gamma_11*E_X1D1_P
        del X1, X11, eta, eps_x, Z, Y1, prob_D, D, X2_0, X2_1, X21_0, X21_1, I_X21D1, I_X20D1
        print(ATT)
        return {"ATT": ATT, 
                "E_X1D1": E_X1D1, "E_ETAD1": E_ETAD1, 
                "E_X21D1": E_X21D1, "E_X20D1": E_X20D1,}
                
                

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
                        delta_3= 2 ,
                        gamma_10=.5, gamma_11=1
                        )

        data = dgp.generate(n=N*tmax, seed=seed)
        X1, X2 = data['X1'], data['X2']
        Y1, Y2 = data['Y1'], data['Y2']
        Z, D = data['Z'], data['D']

        ATT_calculations = dgp.simulate_ATT(n=100000000)
        torch.save(ATT_calculations, f"results_new_dgp/{file_suffix}_ATT_calculations.pt")

        pred_theta = torch.zeros(tmax, 4)
        pred_sig = torch.zeros(tmax, 4)


        for t in tqdm(range(tmax)):
            torch.manual_seed(t)
            X1_sub = X1[t*N:(t+1)*N, :]
            X2_sub = X2[t*N:(t+1)*N, :]
            Y1_sub = Y1[t*N:(t+1)*N].view(-1, 1)
            Y2_sub = Y2[t*N:(t+1)*N].view(-1, 1)
            D_sub = D[t*N:(t+1)*N].view(-1, 1)
            Z_sub = Z[t*N:(t+1)*N, :]

            pred_theta[t,0], pred_sig[t,0] = estimateDiD_OLS(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed= t)

            pred_theta[t,1], pred_sig[t,1]  = estimateDiDLinear(Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed= t)

            pred_theta[t,2], pred_sig[t,2],*_ = estimateDynamicRiesz( Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, folds,
                                                                            method_a = "LASSO", lasso_a_settings = lasso_a_settings,
                                                                                method_f = "LASSO", lasso_f_settings = lasso_f_settings,
                                                                                seed = t)
            pred_theta[t,3], pred_sig[t,3],*_ = estimateDynamicRiesz( Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, folds,
                                                                            method_a = "RF", lasso_a_settings = lasso_a_settings,
                                                                                method_f = "RF", lasso_f_settings = lasso_f_settings,
                                                                                seed = t)

                

        torch.save(pred_theta, f"results_new_dgp/{file_suffix}_pred_theta_OLS.pt")
        torch.save(pred_sig, f"results_new_dgp/{file_suffix}_pred_sig_OLS.pt")



