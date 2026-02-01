# Parameters
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiD_OLSv2 import estimateDiD_OLS
from utils.estimateDiDLinear import estimateDiDLinear
import torch
import pandas as pd
import time
from torch.distributions import Normal

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class DiD_DGP:
    def __init__(self, dim_X=3, dim_Z = 3, beta_1=2.0, beta_20 = 1, beta_21 = 2, c_1=1.0, delta_1 = 1 , delta_2 =1 , delta_3=1 , alpha_1 =1, gamma_1 = None, gamma_2= None, g  = None, gamma_10 = 0,gamma_11=0 ):
        self.dim_X = dim_X
        self.dim_Z = dim_Z
        self.beta_1 = beta_1
        self.beta_20 = beta_20
        self.beta_21 = beta_21
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
        Y2 = Y1 + self.c_1 * D.squeeze() + (D*self.beta_21 +(1-D)*self.beta_20) * (X21 > 0).float() + self.delta_3* D.squeeze() * X21 + torch.randn(n)
        return {
            "X1": X1,
            "X2": X2,
            "Y1": Y1,
            "Y2": Y2,
            "D": D,
            "Z": Z,
        }   
    def simulate_att_population(self, n=int(5e6), batch_size=int(5e5), seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)

        #
        diff_sum_treated = 0.0
        n_treated = 0


        ones_template = None  


        num_full = n // batch_size
        remainder = n - num_full * batch_size

        def run_batch(m):
            nonlocal diff_sum_treated, n_treated, ones_template


            X1 = torch.randn(m, self.dim_X)
            X11 = X1[:, 0]
            Z  = torch.randn(m, self.dim_Z)
            eta = torch.randn(m)
            eps_x  = torch.randn(m, self.dim_X)
            eps_y1 = torch.randn(m)
            eps_y2 = torch.randn(m)   # reused for both potentials


            Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + eps_y1


            score = self.delta_2 * X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta
            p = self.g(score)
            D = torch.bernoulli(p)

            # === Build X2(d) under do(D=d) with same shocks
            I_x11 = (X11 > 0).float()
            if ones_template is None:
                ones_template = torch.ones(1, self.dim_X)

            def X2_given(d):
                d = float(d)
                shift = (d * (1 + eta)).unsqueeze(1)
                gate  = ((d * self.gamma_11 + (1 - d) * self.gamma_10) * I_x11).unsqueeze(1)
                return gate + X1 + ones_template.expand(m, -1) * shift + eps_x

            X2_0 = X2_given(0.0)
            X2_1 = X2_given(1.0)
            X21_0 = X2_0[:, 0]
            X21_1 = X2_1[:, 0]

            # === Potential outcomes with identical eps_y2
            Y2_0 = Y1 + self.beta_20 * (X21_0 > 0).float()  
            Y2_1 = Y1 + self.c_1 + self.beta_21 * (X21_1 > 0).float() + self.delta_3 * 1.0 * X21_1 

            # === ATT accumulator
            treated = (D == 1)
            if treated.any():
                diff_sum_treated += (Y2_1 - Y2_0)[treated].sum().item()
                n_treated += treated.sum().item()

        # Run full batches
        for _ in tqdm(range(num_full), desc = "Running diferent batches to simualte the true ATT"):
            run_batch(batch_size)

        # Run remainder if needed
        if remainder > 0:
            run_batch(remainder)

        # Compute ATT
        att = diff_sum_treated / n_treated

        # Store and return
        self.ATT = att

        return att
                 
  
                
                

Ns = [500, 1000, 2000]
tmax = 100
dimX = 3
dimZ = 2
seed = 123 # this seed is for the DGP
folds = 5
# Bounds (only for truncated distributions)
lower = 0.30
upper = 0.70

folder = "results_dgp_new_coef"
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
                        delta_3= 2 ,
                        gamma_10=0, gamma_11=0, # remaining in the old model
                        beta_20 = 1, beta_21 = 2, # using new heterogenity
                        )

        data = dgp.generate(n=N*tmax, seed=seed)
        X1, X2 = data['X1'], data['X2']
        Y1, Y2 = data['Y1'], data['Y2']
        Z, D = data['Z'], data['D']

        ATT_calculations = dgp.simulate_att_population(n = 1000000000)
        torch.save(ATT_calculations, f"{folder}/{file_suffix}_ATT_calculations.pt")

        pred_theta = torch.zeros(tmax, 4)
        pred_sig = torch.zeros(tmax, 4)


        for t in range(tmax):
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

                

        torch.save(pred_theta, f"{folder}/{file_suffix}_pred_theta_OLS.pt")
        torch.save(pred_sig, f"{folder}/{file_suffix}_pred_sig_OLS.pt")



