import utils.dynamicRieszLASSO
import utils.dynamicRieszRF
import utils.dynamicRieszNet
import torch
import numpy as np
from sklearn.model_selection import KFold
import time

lasso_cv_settings_global = {
    'b_degree' : 1,
    'cv_folds' : 5,
    'random_state' : 42
}
lasso_a_settings_global = {
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
lasso_f_settings_global = {
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
rf_a_settings_global = {
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
    'n_jobs' : -1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}
rf_f_settings_global = {
    'poly_degree' : 1,
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
    'n_jobs' : -1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}
net_a_settings_global = {
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
net_f_settings_global = {
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

class Trainer: 
    def __init__(self, Y, X, D, d, X_index,
                  method_a = 'LASSO', method_f = "LASSO",
                    lasso_a_settings = lasso_a_settings_global,
                      lasso_f_settings = lasso_f_settings_global,
                        lasso_cv_settings = lasso_cv_settings_global,
                            rf_a_settings = rf_a_settings_global,
                                rf_f_settings = rf_f_settings_global,
                                    net_a_settings = net_a_settings_global,
                                        net_f_settings = net_f_settings_global):
                
        """
        Dictionary mapping function 

        Y (nx1) : Vector of outcomes
        X (nx(p1...pT)) : Full matrix of covariates
        D (nxT) : Full matrix of treatment assignments 
        d (nxT) : Full matrix of counterfactual assignments 
        
        X_index (T,1) : Shows the index where each new set of covariates starts. E.g. (X1,X1,X1,X2,X2,X3,X3,X3) -> [0,3,5]. 
        """
        self.Y = Y
        self.X = X
        self.D = D
        self.d = d
        self.X_index = X_index
        self.T = D.shape[1]

        # initalise trainers for f functions
        if method_f == 'LASSO':
            self.learners_f = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings) for _ in range(self.T)]
        if method_f == 'LASSO_CV':
            self.learners_f = [utils.dynamicRieszLASSO.Learner_f_LASSO_cv(lasso_cv_settings = lasso_cv_settings) for _ in range(self.T)]
        if method_f == 'RF':
            self.learners_f = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings = rf_f_settings) for _ in range(self.T)]
        if method_f == 'Net':
            self.learners_f = [utils.dynamicRieszNet.Learner_f_Net(net_f_settings=net_f_settings) for _ in range(self.T)]
        
        # initalise trainers for a functions
        if method_a == "LASSO":
            self.learners_a = [utils.dynamicRieszLASSO.Learner_a_LASSO(lasso_a_settings = lasso_a_settings) for _ in range(self.T)]
        if method_a == "RF":
            self.learners_a = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings = rf_a_settings) for _ in range(self.T)]
        if method_a == "Net":
            self.learners_a = [utils.dynamicRieszNet.Learner_a_Net(net_a_settings = net_a_settings) for _ in range(self.T)]

    def train(self):
        # Loop backwards over f_t functions
        for t in range(self.T-1,-1,-1):
            if t == self.T-1:
                f_hat = self.Y
            else:
                f_hat = self.learners_f[t+1].predict(self.X[:, :(self.X_index[t+1]+1)], self.d[:, :(t+1+1)])
            self.learners_f[t].fit(f_hat, self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)])
        # Loop forwards over a_t functions
        for t in range(self.T):
            if t == 0:
                a_prev = torch.ones(self.Y.shape)
            else:
                a_prev = self.learners_a[t-1].predict(self.X[:, :(self.X_index[t-1]+1)], self.D[:, :t])
            self.learners_a[t].fit(self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)], self.d[:, :(t+1)], a_prev)

def thetahat(learners_f, learners_a, Y, X, D, d, X_index):
    """
    This function gives the estimate for theta. 
    
    trainer (Trainer class) : the trainer from which the a_t and f_t functions will be used
    Y : Outcome data
    X: Covariates
    D : Treatment 
    d : Counterfactual treatment
    X_index : Denotes where period t covariates start

    Can delete the RR
    """
        
    T = D.shape[1]
    RR1 = torch.zeros(X.shape[0],1)
    RR2 = torch.zeros(X.shape[0],1)
    f1_hat = torch.zeros(X.shape[0],1)
    f2_hat = torch.zeros(X.shape[0],1)

    theta = (learners_f[0].predict(X[:, :(X_index[0]+1)], d[:,:1]))    
    for t in range(T):
        if t == (T-1):
            # a_T (X_T, D_T) * ( Y - f_T (X_T, D_T) )
            theta += learners_a[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)]) * (Y - learners_f[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)] ))
            # theta += learners_a[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)]) * (Y - learners_f[t].predict(X[:, :(X_index[t]+1)], d[:,:(t+1)] ))
            RR2 = learners_a[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)])
            f2_hat = learners_f[t].predict(X[:, :(X_index[t]+1)], d[:,:(t+1)])
        else:
            # a_t (X_t, D_t) * ( f_t+1 (X_t+1, d_(t+1)) - f_T (X_T, D_T) )
            theta += learners_a[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)]) * (learners_f[t+1].predict(X[:, :(X_index[t+1]+1)], d[:,:(t+1+1)]) - learners_f[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)] ))
            # theta += learners_a[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)]) * (learners_f[t+1].predict(X[:, :(X_index[t+1]+1)], d[:,:(t+1+1)]) - learners_f[t].predict(X[:, :(X_index[t]+1)], d[:,:(t+1)] ))
            RR1 = learners_a[t].predict(X[:, :(X_index[t]+1)], D[:,:(t+1)])
            f1_hat = learners_f[t].predict(X[:, :(X_index[t]+1)], d[:,:(t+1)])

    return theta, RR1, RR2, f1_hat, f2_hat

def estimateDynamicRiesz(Y, X, D, d, X_index, folds,
                          method_a = "LASSO", method_f = "LASSO",
                            lasso_a_settings = lasso_a_settings_global,
                                lasso_f_settings = lasso_f_settings_global,
                                    lasso_cv_settings = lasso_cv_settings_global,
                                        rf_a_settings = rf_a_settings_global,
                                            rf_f_settings = rf_f_settings_global,
                                                net_a_settings = net_a_settings_global,
                                                    net_f_settings = net_f_settings_global):
        
    fold_results = torch.zeros(Y.shape)
    RR1 = torch.zeros(Y.shape)
    RR2 = torch.zeros(Y.shape)
    f1 = torch.zeros(Y.shape)
    f2 = torch.zeros(Y.shape)

    # Iterate through folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(Y)):

        # Splitting data
        X_train, X_test = X[train_index], X[test_index]
        D_train, D_test = D[train_index], D[test_index]
        d_train, d_test = d[train_index], d[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Train data for fold
        trainer = Trainer(Y_train, X_train, D_train, d_train, X_index, 
                          method_a, method_f,
                            lasso_a_settings = lasso_a_settings,
                                lasso_f_settings = lasso_f_settings,
                                    lasso_cv_settings = lasso_cv_settings,
                                        rf_a_settings = rf_a_settings,
                                            rf_f_settings = rf_f_settings,
                                                net_a_settings = net_a_settings,
                                                    net_f_settings = net_f_settings)
        trainer.train()

        # Predict theta for every observation in the test data
        fold_results[test_index], RR1[test_index], RR2[test_index], f1[test_index], f2[test_index] = thetahat(trainer.learners_f, trainer.learners_a, Y_test, X_test, D_test, d_test, X_index)   
        
    point = torch.mean(fold_results)
    sigma2 = torch.mean( (fold_results - point) ** 2 )
    sigma = torch.sqrt(sigma2)
    
    return point, sigma, RR1, RR2, f1, f2

class Trainer_all: 
    def __init__(self, Y, X, D, d, X_index,
                    lasso_a_settings = lasso_a_settings_global,
                      lasso_f_settings = lasso_f_settings_global,
                            rf_a_settings = rf_a_settings_global,
                                rf_f_settings = rf_f_settings_global,
                                    net_a_settings = net_a_settings_global,
                                        net_f_settings = net_f_settings_global):
                
        """
        Dictionary mapping function 

        Y (nx1) : Vector of outcomes
        X (nx(p1...pT)) : Full matrix of covariates
        D (nxT) : Full matrix of treatment assignments 
        d (nxT) : Full matrix of counterfactual assignments 
        
        X_index (T,1) : Shows the index where each new set of covariates starts. E.g. (X1,X1,X1,X2,X2,X3,X3,X3) -> [0,3,5]. 
        """
        self.Y = Y
        self.X = X
        self.D = D
        self.d = d
        self.X_index = X_index
        self.T = D.shape[1]

        # initalise trainers for f functions
        self.learners_f_LASSO = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings) for _ in range(self.T)]
        self.learners_f_RF = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings = rf_f_settings) for _ in range(self.T)]
        self.learners_f_Net = [utils.dynamicRieszNet.Learner_f_Net(net_f_settings=net_f_settings) for _ in range(self.T)]

        rf_f_settings0 = rf_f_settings_global.copy()
        rf_f_settings0['poly_degree'] = 0
        self.learners_f_RF0 = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings = rf_f_settings0) for _ in range(self.T)]

        # initalise trainers for a functions
        self.learners_a_LASSO = [utils.dynamicRieszLASSO.Learner_a_LASSO(lasso_a_settings = lasso_a_settings) for _ in range(self.T)]
        self.learners_a_RF = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings = rf_a_settings) for _ in range(self.T)]
        self.learners_a_Net = [utils.dynamicRieszNet.Learner_a_Net(net_a_settings = net_a_settings) for _ in range(self.T)]

        rf_a_settings1 = rf_a_settings_global.copy()
        rf_a_settings1['poly_degree'] = 1
        self.learners_a_RF1 = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings = rf_a_settings1) for _ in range(self.T)]

    def train(self):
        # Loop backwards over f_t functions
        # for t in range(self.T-1,-1,-1):
        #     if t == self.T-1:
        #         f_hat = self.Y
        #     else:
        #         f_hat = self.learners_f_LASSO[t+1].predict(self.X[:, :(self.X_index[t+1]+1)], self.d[:, :(t+1+1)])
        #     self.learners_f_LASSO[t].fit(f_hat, self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)])
        for t in range(self.T-1,-1,-1):
            if t == self.T-1:
                f_hat = self.Y
            else:
                f_hat = self.learners_f_RF[t+1].predict(self.X[:, :(self.X_index[t+1]+1)], self.d[:, :(t+1+1)])
            self.learners_f_RF[t].fit(f_hat, self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)])
        # for t in range(self.T-1,-1,-1):
        #     if t == self.T-1:
        #         f_hat = self.Y
        #     else:
        #         f_hat = self.learners_f_Net[t+1].predict(self.X[:, :(self.X_index[t+1]+1)], self.d[:, :(t+1+1)])
        #     self.learners_f_Net[t].fit(f_hat, self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)])
        # for t in range(self.T-1,-1,-1):
        #     if t == self.T-1:
        #         f_hat = self.Y
        #     else:
        #         f_hat = self.learners_f_RF0[t+1].predict(self.X[:, :(self.X_index[t+1]+1)], self.d[:, :(t+1+1)])
        #     self.learners_f_RF0[t].fit(f_hat, self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)])
            
        # Loop forwards over a_t functions
        # for t in range(self.T):
        #     if t == 0:
        #         a_prev = torch.ones(self.Y.shape)
        #     else:
        #         a_prev = self.learners_a_LASSO[t-1].predict(self.X[:, :(self.X_index[t-1]+1)], self.D[:, :t])
        #     self.learners_a_LASSO[t].fit(self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)], self.d[:, :(t+1)], a_prev)
        for t in range(self.T):
            if t == 0:
                a_prev = torch.ones(self.Y.shape)
            else:
                a_prev = self.learners_a_RF[t-1].predict(self.X[:, :(self.X_index[t-1]+1)], self.D[:, :t])
            self.learners_a_RF[t].fit(self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)], self.d[:, :(t+1)], a_prev)
        # for t in range(self.T):
        #     if t == 0:
        #         a_prev = torch.ones(self.Y.shape)
        #     else:
        #         a_prev = self.learners_a_Net[t-1].predict(self.X[:, :(self.X_index[t-1]+1)], self.D[:, :t])
        #     self.learners_a_Net[t].fit(self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)], self.d[:, :(t+1)], a_prev)
        # for t in range(self.T):
        #     if t == 0:
        #         a_prev = torch.ones(self.Y.shape)
        #     else:
        #         a_prev = self.learners_a_RF1[t-1].predict(self.X[:, :(self.X_index[t-1]+1)], self.D[:, :t])
        #     self.learners_a_RF1[t].fit(self.X[:, :(self.X_index[t]+1)], self.D[:, :(t+1)], self.d[:, :(t+1)], a_prev)

def estimateDynamicRiesz_all(Y, X, D, d, X_index, folds,

                            lasso_a_settings = lasso_a_settings_global,
                                lasso_f_settings = lasso_f_settings_global,
                                    lasso_cv_settings = lasso_cv_settings_global,
                                        rf_a_settings = rf_a_settings_global,
                                            rf_f_settings = rf_f_settings_global,
                                                net_a_settings = net_a_settings_global,
                                                    net_f_settings = net_f_settings_global):
        
    fold_results = torch.zeros(Y.shape[0],16)
    RR1 = torch.zeros(Y.shape[0],16)
    RR2 = torch.zeros(Y.shape[0],16)
    f1 = torch.zeros(Y.shape[0],16)
    f2 = torch.zeros(Y.shape[0],16)

    # Iterate through folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(Y)):

        # Splitting data
        X_train, X_test = X[train_index], X[test_index]
        D_train, D_test = D[train_index], D[test_index]
        d_train, d_test = d[train_index], d[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Train data for fold
        trainer = Trainer_all(Y_train, X_train, D_train, d_train, X_index)
        trainer.train()

        # Predict theta for every observation in the test data
        # fold_results[test_index,:1], RR1[test_index,:1], RR2[test_index,:1], f1[test_index,:1], f2[test_index,:1] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_LASSO, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,1:2], RR1[test_index,1:2], RR2[test_index,1:2], f1[test_index,1:2], f2[test_index,1:2] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_RF, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,2:3], RR1[test_index,2:3], RR2[test_index,2:3], f1[test_index,2:3], f2[test_index,2:3] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_Net, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,3:4], RR1[test_index,3:4], RR2[test_index,3:4], f1[test_index,3:4], f2[test_index,3:4] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_RF1, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,4:5], RR1[test_index,4:5], RR2[test_index,4:5], f1[test_index,4:5], f2[test_index,4:5] = thetahat(trainer.learners_f_RF, trainer.learners_a_LASSO, Y_test, X_test, D_test, d_test, X_index)   
        fold_results[test_index,5:6], RR1[test_index,5:6], RR2[test_index,5:6], f1[test_index,5:6], f2[test_index,5:6] = thetahat(trainer.learners_f_RF, trainer.learners_a_RF, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,6:7], RR1[test_index,6:7], RR2[test_index,6:7], f1[test_index,6:7], f2[test_index,6:7] = thetahat(trainer.learners_f_RF, trainer.learners_a_Net, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,7:8], RR1[test_index,7:8], RR2[test_index,7:8], f1[test_index,7:8], f2[test_index,7:8] = thetahat(trainer.learners_f_RF, trainer.learners_a_RF1, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,8:9], RR1[test_index,8:9], RR2[test_index,8:9], f1[test_index,8:9], f2[test_index,8:9] = thetahat(trainer.learners_f_Net, trainer.learners_a_LASSO, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,9:10], RR1[test_index,9:10], RR2[test_index,9:10], f1[test_index,9:10], f2[test_index,9:10] = thetahat(trainer.learners_f_Net, trainer.learners_a_RF, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,10:11], RR1[test_index,10:11], RR2[test_index,10:11], f1[test_index,10:11], f2[test_index,10:11] = thetahat(trainer.learners_f_Net, trainer.learners_a_Net, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,11:12], RR1[test_index,11:12], RR2[test_index,11:12], f1[test_index,11:12], f2[test_index,11:12] = thetahat(trainer.learners_f_Net, trainer.learners_a_RF1, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,12:13], RR1[test_index,12:13], RR2[test_index,12:13], f1[test_index,12:13], f2[test_index,12:13] = thetahat(trainer.learners_f_RF0, trainer.learners_a_LASSO, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,13:14], RR1[test_index,13:14], RR2[test_index,13:14], f1[test_index,13:14], f2[test_index,13:14] = thetahat(trainer.learners_f_RF0, trainer.learners_a_RF, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,14:15], RR1[test_index,14:15], RR2[test_index,14:15], f1[test_index,14:15], f2[test_index,14:15] = thetahat(trainer.learners_f_RF0, trainer.learners_a_Net, Y_test, X_test, D_test, d_test, X_index)   
        # fold_results[test_index,15:16], RR1[test_index,15:16], RR2[test_index,15:16], f1[test_index,15:16], f2[test_index,15:16] = thetahat(trainer.learners_f_RF0, trainer.learners_a_RF1, Y_test, X_test, D_test, d_test, X_index)   

    point = torch.mean(fold_results,0)
    sigma2 = torch.mean( (fold_results - point) ** 2 ,0)
    sigma = torch.sqrt(sigma2)
    
    return point, sigma, RR1, RR2, f1, f2
