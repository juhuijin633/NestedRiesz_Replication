import copy

import utils.dynamicRieszLASSO
import utils.dynamicRieszRF
import utils.dynamicRieszNet
from utils.hyperparams import (
    lasso_a_settings,
    lasso_f_settings,
    net_a_settings,
    net_f_settings,
    rf_a_settings,
    rf_f_settings,
)
import torch
import numpy as np
from sklearn.model_selection import KFold
import pdb
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

def thetahat(learners_f, learners_a, Y, G, X, D, S, subsetting = False):
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
        
    predictors_2 = torch.hstack((X, S))
    predictors_1 = X

    rr_variables_1 = torch.hstack((G, D))
    rr_variables_2 = G
    d1_1 = torch.tensor([0, 1]).repeat(rr_variables_1.shape[0], 1)
    d2_1 = torch.ones(rr_variables_2.shape)
    d1_0 = torch.zeros(rr_variables_1.shape)
    d2_0 = torch.ones(rr_variables_2.shape)


    pi = (1 - G) / torch.mean((1 - G.double()))

    # First the estimate of Yi(1)
    if subsetting:
        theta = pi * (learners_f[0].predict(predictors_1, torch.ones(D.shape))) 
    else:
        theta = pi * (learners_f[0].predict(predictors_1, d1_1)) 


    #theta -= learners_a[0].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_1) - learners_f[0].predict(predictors_1, rr_variables_1))
    if subsetting: 
        #theta -= learners_a[0].predict(predictors_1, rr_variables_1) * (torch.tensor(learners_f[1].predict(predictors_2)) - learners_f[0].predict(predictors_1, D))
        theta -= learners_a[0].predict(predictors_1, rr_variables_1) * (torch.tensor(learners_f[1].predict(predictors_2)).view(-1, 1) - learners_f[0].predict(predictors_1, D))
        theta -= learners_a[1].predict(predictors_2, rr_variables_2) * ( (Y) - torch.tensor(learners_f[1].predict(predictors_2)).view(-1, 1))

    else:
        #theta -= learners_a[0].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_1) - learners_f[0].predict(predictors_1, D))
        theta -= learners_a[0].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_1) - learners_f[0].predict(predictors_1, rr_variables_1))
        theta -= learners_a[1].predict(predictors_2, rr_variables_2) * ( (Y) - learners_f[1].predict(predictors_2, rr_variables_2))

    

    # Now add the estimate of Yi(0)
    if subsetting:#
        theta -= pi * (learners_f[0].predict(predictors_1, torch.zeros(D.shape)))      
    else:
        theta -= pi * (learners_f[0].predict(predictors_1, d1_0))      
    #

    if subsetting:
        #theta += learners_a[2].predict(predictors_1, rr_variables_1) * (torch.tensor(learners_f[1].predict(predictors_2))- learners_f[0].predict(predictors_1, D))
        theta += learners_a[2].predict(predictors_1, rr_variables_1) * (torch.tensor(learners_f[1].predict(predictors_2)).view(-1, 1)- learners_f[0].predict(predictors_1, D))

        theta += learners_a[3].predict(predictors_2, rr_variables_2) * ( (Y) - torch.tensor(learners_f[1].predict(predictors_2)).view(-1, 1))
    else:
        #theta += learners_a[2].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_0) - learners_f[0].predict(predictors_1, D))
        theta += learners_a[2].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_0) - learners_f[0].predict(predictors_1, rr_variables_1))

        theta += learners_a[3].predict(predictors_2, rr_variables_2) * ( (Y) - learners_f[1].predict(predictors_2, rr_variables_2))
    

    # pdb.set_trace()

    return theta.float()

def estimateDynamicRiesz(Y, G, X, D, S, folds,
                          method_a = "LASSO", method_f = "LASSO",
                            lasso_a_settings = lasso_a_settings,
                                lasso_f_settings = lasso_f_settings,
                                        rf_a_settings = rf_a_settings,
                                            rf_f_settings = rf_f_settings,
                                                net_a_settings = net_a_settings,
                                                    net_f_settings = net_f_settings, seed = None, subsetting = False):
        
    fold_results = torch.zeros(Y.shape, dtype=torch.float64)

    # Iterate through folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(Y)):

        # Splitting data
        Y_train, Y_test = Y[train_index], Y[test_index]
        G_train, G_test = G[train_index], G[test_index]
        X_train, X_test = X[train_index], X[test_index]
        D_train, D_test = D[train_index], D[test_index]
        S_train, S_test = S[train_index], S[test_index]

        # Train data for fold
        trainer = Trainer(Y_train, G_train, X_train, D_train, S_train,
                          method_a, method_f,
                            lasso_a_settings = lasso_a_settings,
                                lasso_f_settings = lasso_f_settings,
                                        rf_a_settings = rf_a_settings,
                                            rf_f_settings = rf_f_settings,
                                                net_a_settings = net_a_settings,
                                                    net_f_settings = net_f_settings, seed = seed, subsetting = subsetting)
        trainer.train()

        # Predict theta for every observation in the test data
        fold_results[test_index] = thetahat(trainer.learners_f, trainer.learners_a, Y_test, G_test, X_test, D_test, S_test, subsetting= subsetting ).double()   

    point = torch.mean(fold_results)
    sigma2 = torch.mean( (fold_results - point) ** 2 )
    sigma = torch.sqrt(sigma2)
    
    return point, sigma, fold_results

class Trainer: 
    def __init__(self, Y, G, X, D, S,
                  method_a = 'LASSO', method_f = "LASSO",
                    lasso_a_settings = lasso_a_settings,
                      lasso_f_settings = lasso_f_settings,
                            rf_a_settings = rf_a_settings,
                                rf_f_settings = rf_f_settings,
                                    net_a_settings = net_a_settings,
                                        net_f_settings = net_f_settings, 
                                        seed = None, subsetting = False):
                
        """
        Dictionary mapping function 

        Y (nX) : Vector of outcomes
        X (nx(p1...pT)) : Full matrix of covariates
        D (nxT) : Full matrix of treatment assignments 
        d (nxT) : Full matrix of counterfactual assignments 
        
        X_index (T,1) : Shows the index where each new set of covariates starts. E.g. (X,X,X,X2,X2,X3,X3,X3) -> [0,3,5]. 
        """
        self.Y = Y
        self.G = G
        self.X = X
        self.D = D
        self.S = S
        self.T = 2
        self.method_f = method_f
        self.method_a = method_a
        self.subsetting = subsetting

        # initalise trainers for f functions
        if method_f == 'LASSO':
            lasso_f_cfg = copy.deepcopy(lasso_f_settings)
            lasso_f_cfg["seed"] = seed
            if subsetting: # include a different estimator for f2
                self.learners_f = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings=lasso_f_cfg), Lasso(random_state=seed)]
            else:
                self.learners_f = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings=lasso_f_cfg) for _ in range(self.T)]
        if method_f == 'RF':
            rf_f_cfg = copy.deepcopy(rf_f_settings)
            rf_f_cfg["random_state"] = seed
            if subsetting:
                self.learners_f = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings=rf_f_cfg), RandomForestRegressor(random_state=seed)]
            else:
                self.learners_f = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings=rf_f_cfg) for _ in range(self.T)]
        if method_f == 'Net':
            self.learners_f = [utils.dynamicRieszNet.Learner_f_Net(net_f_settings=copy.deepcopy(net_f_settings)) for _ in range(self.T)]
        
        # initalise trainers for a functions
        if method_a == "LASSO":
            lasso_a_cfg = copy.deepcopy(lasso_a_settings)
            lasso_a_cfg["seed"] = seed
            self.learners_a = [utils.dynamicRieszLASSO.Learner_a_LASSO(lasso_a_settings=lasso_a_cfg) for _ in range(self.T * 2)]
        if method_a == "RF":
            rf_a_cfg = copy.deepcopy(rf_a_settings)
            rf_a_cfg["random_state"] = seed
            self.learners_a = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings=rf_a_cfg) for _ in range(self.T * 2)]
        if method_a == "Net":
            self.learners_a = [utils.dynamicRieszNet.Learner_a_Net(net_a_settings=copy.deepcopy(net_a_settings)) for _ in range(self.T * 2)]

    def train(self):
        predictors_2 = torch.hstack((self.X, self.S))
        predictors_1 = self.X

        rr_variables_1 = torch.hstack((self.G, self.D))
        rr_variables_2 = self.G
        d1_1 = torch.tensor([0, 1]).repeat(rr_variables_1.shape[0], 1)
        d2 = torch.ones(rr_variables_2.shape)
        d1_0 = torch.zeros(rr_variables_1.shape)
        # Estimate f2, only on the dataset with  G=1,#
        mask_G0 = (1 - self.G).bool().squeeze()
        mask_G1 =  self.G.bool().squeeze()

        if self.subsetting:
            self.learners_f[1].fit(predictors_2[mask_G1], self.Y[mask_G1].ravel())
        else:
            self.learners_f[1].fit(self.Y, predictors_2, rr_variables_2)


        # Estimate f1
        if self.subsetting:
            f2_hat = torch.tensor(self.learners_f[1].predict(predictors_2))
        else:
            f2_hat = self.learners_f[1].predict(predictors_2, d2)

        if self.subsetting:
            self.learners_f[0].fit(f2_hat[mask_G0], predictors_1[mask_G0], self.D[mask_G0])
        else:
            self.learners_f[0].fit(f2_hat, predictors_1, rr_variables_1)


        # Estimate a1^1
        a_prev_1 = (1 - self.G) / torch.mean(1 - self.G.double())
        if self.subsetting:
             self.learners_a[0].fit(predictors_1[mask_G0], self.D[mask_G0].view(-1,1), torch.ones(self.D[mask_G0].shape), torch.ones(self.D[mask_G0].shape))
        else:
            self.learners_a[0].fit(predictors_1, rr_variables_1, d1_1, a_prev_1)



        # Estimate a2^1
        if self.subsetting:
            a_prev_21 = self.learners_a[0].predict(predictors_1, rr_variables_1) 
        else:
            a_prev_21 = self.learners_a[0].predict(predictors_1, rr_variables_1)
            self.learners_a[1].fit(predictors_2, rr_variables_2, d2, a_prev_21)

        # Estimate a1^0
        a_prev_1 = (1 - self.G) / torch.mean(1 - self.G.double())
        self.learners_a[2].fit(predictors_1, rr_variables_1, d1_0, a_prev_1)


        # Estimate a2^0
        a_prev_20 = self.learners_a[2].predict(predictors_1, rr_variables_1)
        self.learners_a[3].fit(predictors_2, rr_variables_2, d2, a_prev_20)
