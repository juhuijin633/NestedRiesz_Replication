import torch
import math
import pandas as pd

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
N_values = [500, 1000, 2000]
tmax     = 500

# Simulation configurations
configs = [
    # --- E[Y(1,1)] ---
    {'label': 'Linear DGP + truncated logistic',              'result_dir': 'results',                  'func': 'truncated_logistic', 'theta_key': 'theta_true'},
    {'label': 'Nonlinear DGP + truncated adversarial',        'result_dir': 'results_nonlinear',        'func': 'truncated_adv',      'theta_key': 'theta_true'},
    {'label': 'Linear DGP + truncated adversarial',           'result_dir': 'results',                  'func': 'truncated_adv',      'theta_key': 'theta_true'},
    {'label': 'Linear DGP + logistic',                        'result_dir': 'results',                  'func': 'logistic',           'theta_key': 'theta_true'},
    # --- ATE = E[Y(1,1)] - E[Y(0,0)] ---
    {'label': 'ATE: Linear DGP + truncated logistic',         'result_dir': 'results_ate',              'func': 'truncated_logistic', 'theta_key': 'ate_true'},
    {'label': 'ATE: Nonlinear DGP + truncated adversarial',   'result_dir': 'results_nonlinear_ate',    'func': 'truncated_adv',      'theta_key': 'ate_true'},
    {'label': 'ATE: Linear DGP + truncated adversarial',      'result_dir': 'results_ate',              'func': 'truncated_adv',      'theta_key': 'ate_true'},
    {'label': 'ATE: Linear DGP + logistic',                   'result_dir': 'results_ate',              'func': 'logistic',           'theta_key': 'ate_true'},
]

methods = ["Oracle", "Bradic", "LASSO-LASSO", "RF-RF", "Net-Net"]

# -----------------------------------------------------------------------
# Load results
# -----------------------------------------------------------------------
def load_results(result_dir, N, func_name, tmax):
    results, missing = [], []
    for t in range(tmax):
        path = f'{result_dir}/N{N}/{func_name}/result_{t}.pt'
        try:
            results.append(torch.load(path))
        except FileNotFoundError:
            missing.append(t)
    return results, missing

def compute_table(results, N, theta_key='theta_true'):
    n          = len(results)
    theta_true = results[0][theta_key]

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
        results, missing = load_results(cfg['result_dir'], N, cfg['func'], tmax)

        if not results:
            print(f"\n[{cfg['label']}, N={N}] — no results found, skipping")
            continue

        if missing:
            print(f"\nWarning [{cfg['label']}, N={N}]: {len(missing)} missing iterations")

        df, theta_true = compute_table(results, N, theta_key=cfg['theta_key'])

        print(f"\n{'='*65}")
        print(f"{cfg['label']}")
        print(f"N = {N}  |  tmax = {tmax}")
        print(f"True theta = {theta_true:.4f}  |  Iterations: {len(results)} / {tmax}")
        print(f"{'='*65}")
        print(df.to_string(index=False, float_format="{:.4f}".format))
