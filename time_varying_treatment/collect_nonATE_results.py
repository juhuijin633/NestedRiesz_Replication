import torch
import math
import pandas as pd

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
N_values = [500, 1000, 2000]
tmax     = 500

# -----------------------------------------------------------------------
# Non-ATE simulation configurations
# Each entry defines: label, result_dir, func_name, lower/upper bounds.
# result_subdir is derived automatically:
#   - 'logistic' (no bounds in path)
#   - '{func}_{lower}_{upper}' for truncated functions
# -----------------------------------------------------------------------
configs = [
    # --- truncated_logistic ---
    {'label': 'Linear DGP + truncated logistic [0.1, 0.9]',    'result_dir': 'results',           'func': 'truncated_logistic', 'lower': 0.1, 'upper': 0.9},
    {'label': 'Linear DGP + truncated logistic [0.3, 0.7]',    'result_dir': 'results',           'func': 'truncated_logistic', 'lower': 0.3, 'upper': 0.7},
    # --- nonlinear DGP + truncated_adv ---
    {'label': 'Nonlinear DGP + truncated adv [0.1, 0.9]',      'result_dir': 'results_nonlinear', 'func': 'truncated_adv',      'lower': 0.1, 'upper': 0.9},
    {'label': 'Nonlinear DGP + truncated adv [0.3, 0.7]',      'result_dir': 'results_nonlinear', 'func': 'truncated_adv',      'lower': 0.3, 'upper': 0.7},
    # --- linear DGP + truncated_adv ---
    {'label': 'Linear DGP + truncated adv [0.1, 0.9]',         'result_dir': 'results',           'func': 'truncated_adv',      'lower': 0.1, 'upper': 0.9},
    {'label': 'Linear DGP + truncated adv [0.3, 0.7]',         'result_dir': 'results',           'func': 'truncated_adv',      'lower': 0.3, 'upper': 0.7},
    # --- logistic (no bounds) ---
    {'label': 'Linear DGP + logistic',                          'result_dir': 'results',           'func': 'logistic',           'lower': None, 'upper': None},
]

methods = ["Oracle", "Bradic", "LASSO-LASSO", "RF-RF", "Net-Net"]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def result_subdir(func, lower, upper):
    if func == 'logistic' or lower is None:
        return func
    return f'{func}_{lower}_{upper}'

def load_results(result_dir, N, func, lower, upper, tmax):
    subdir = result_subdir(func, lower, upper)
    results, missing = [], []
    for t in range(tmax):
        path = f'{result_dir}/N{N}/{subdir}/result_{t}.pt'
        try:
            results.append(torch.load(path))
        except FileNotFoundError:
            missing.append(t)
    return results, missing

def compute_table(results, N):
    n          = len(results)
    theta_true = results[0]['theta_true']

    pred_theta = torch.zeros(n, 5)
    pred_sig   = torch.zeros(n, 5)

    for i, r in enumerate(results):
        pred_theta[i, 0] = r['oracle_theta']
        pred_sig[i, 0]   = r['oracle_sig']

        bt = r.get('bradic_theta', float('nan'))
        bs = r.get('bradic_sig',   float('nan'))
        pred_theta[i, 1] = bt if not (isinstance(bt, float) and math.isnan(bt)) else float('nan')
        pred_sig[i, 1]   = bs if not (isinstance(bs, float) and math.isnan(bs)) else float('nan')

        pred_theta[i, 2:] = r['pred_theta']   # [LASSO-LASSO, RF-RF, Net-Net]
        pred_sig[i, 2:]   = r['pred_sig']

    bias            = torch.nanmean(pred_theta - theta_true, 0)
    rmse            = torch.sqrt(torch.nanmean((pred_theta - theta_true)**2, 0))
    ub              = pred_theta + 1.96 * pred_sig / (N ** 0.5)
    lb              = pred_theta - 1.96 * pred_sig / (N ** 0.5)
    coverage        = torch.nanmean(((theta_true >= lb) & (theta_true <= ub)).float(), 0)
    interval_length = torch.nanmean(ub - lb, 0)

    return pd.DataFrame({
        "Method"         : methods,
        "Bias"           : bias.tolist(),
        "RMSE"           : rmse.tolist(),
        "Coverage"       : coverage.tolist(),
        "Interval Length": interval_length.tolist(),
    }), theta_true

# -----------------------------------------------------------------------
# Print tables
# -----------------------------------------------------------------------
for cfg in configs:
    for N in N_values:
        results, missing = load_results(
            cfg['result_dir'], N, cfg['func'], cfg['lower'], cfg['upper'], tmax
        )

        if not results:
            print(f"\n[{cfg['label']}, N={N}] — no results found, skipping")
            continue

        if missing:
            print(f"\nWarning [{cfg['label']}, N={N}]: {len(missing)} missing iterations")

        df, theta_true = compute_table(results, N)

        print(f"\n{'='*65}")
        print(f"{cfg['label']}")
        print(f"N = {N}  |  tmax = {tmax}")
        print(f"True theta = {theta_true:.4f}  |  Iterations: {len(results)} / {tmax}")
        print(f"{'='*65}")
        print(df.to_string(index=False, float_format="{:.4f}".format))
