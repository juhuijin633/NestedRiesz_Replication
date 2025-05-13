import torch

class DiD_DGP:
    def __init__(self, dim_X=3, dim_Z = 3, beta_1=2.0, beta_2=1.5, c_1=1.0, delta_1 = 1 , delta_2 =1 , delta_3=0.5 , alpha_1 =.5, gamma_1 = None, gamma_2= None, g  = None):
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
        X2 = X1 + torch.ones(n, self.dim_X) *  (torch.mul(D,Y1) *(1 + eta)).unsqueeze(1)  + eps_x
        X21 = X2[:, 0]
        Y2 = Y1 + self.c_1 * D.squeeze() + self.beta_2 * (X21 > 0).float() + self.delta_3* D.squeeze() * X21 + torch.randn(n)

        self.ATT = torch.mean(Y2[D.squeeze() == 1] - Y1[D.squeeze() == 1])
        return {
            "X1": X1,
            "X2": X2,
            "D": D.squeeze(),
            "Z": Z,
            "Y1": Y1,
            "Y2": Y2,
            "ATT": self.ATT,
        }
