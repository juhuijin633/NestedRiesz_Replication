import pickle
import numpy as np

results = {
    "TSLS":           (217.94468690566887, 24261836.74004646, np.array([148.21802527, 287.67134854])),
    "TSLS with ridge":(214.34229872091177, 23901023.29690901, np.array([145.13605431, 283.54854313])),
    "RKHS Nystrom":   (265.52932066577876,  6803720.1389495,  np.array([228.60519204, 302.45344929])),
    "rf":             (291.5446832045578,   6243287.899568968, np.array([256.17397824, 326.91538817])),
    "linear_l1":      (291.5446832045578,   6243287.899568968, np.array([256.17397824, 326.91538817])),
    "linear_l2":      (291.5446832045578,   6243287.899568968, np.array([256.17397824, 326.91538817])),
    "net":            (192.7536925036367,  28218296.322498806, np.array([117.55636097, 267.95102404])),
}

with open("results_nnpiv_earnings.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved results_nnpiv_earnings.pkl")
for k, (theta, var, ci) in results.items():
    print(f"  {k}: theta={theta:.4f}, ci=[{ci[0]:.4f}, {ci[1]:.4f}]")
