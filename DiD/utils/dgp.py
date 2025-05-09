import torch

class DiD_DGP:
    def __init__(self, dim_X=5, dim_Z = 3, beta_1=2.0, beta_2=1.5, c_1=1.0, delta=0.5, gamma1 = None, gamma2= None, g  = None):
        self.dim_X = dim_X
        self.dim_Z = dim_Z
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.c_1 = c_1
        self.delta = delta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        if gamma1 is None:
            self.gamma1 = torch.zeros(dim_Z)
        if gamma2 is None:
            self.gamma2 = torch.zeros(dim_Z)
        self.g = g # this function specifies the propensity model
        self.ATT = None
       

    def generate(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        X1 = torch.randn(n, self.dim_X)
        Z = torch.randn(n, self.dim_Z)
        X11 = X1[:, 0]
        prob_D = self.g(X11 + Z @ self.gamma1) # propensity score
        D = torch.bernoulli(prob_D).unsqueeze(1)
        scalar_noise = torch.randn(n, 1)
        noise = torch.randn(n, self.dim_X)
        X2 = X1 + D * (1 + scalar_noise) * torch.ones_like(X1) + noise
        Y1 = self.beta_1 * (X11 > 0).float() + X11 + Z @ self.gamma2
        X21 = X2[:, 0]
        Y2 = Y1 + self.c_1 * D.squeeze() + self.beta_2 * (X21 > 0).float() + self.delta * D.squeeze() * X21


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
