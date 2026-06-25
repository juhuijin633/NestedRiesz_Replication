"""Data-generating processes for time-varying treatment simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.distributions import Normal


def _to_float(value) -> float:
    """Scalar float from Python numeric or 0-d torch tensor."""
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)


def get_treatment_func(
    func_name: str,
    lower: float = 0.10,
    upper: float = 0.90,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def logistic(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) / (1 + torch.exp(x))

    def truncated_logistic(x: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * logistic(x)

    def truncated_adv(x: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * (x > 0).float()

    func_map = {
        "logistic": logistic,
        "truncated_logistic": truncated_logistic,
        "truncated_adv": truncated_adv,
    }
    if func_name not in func_map:
        raise ValueError(f"Unknown treatment function: {func_name}")
    return func_map[func_name]


@dataclass
class SimulationData:
    Y: torch.Tensor
    X: torch.Tensor
    D: torch.Tensor
    X_index: torch.Tensor
    pi1: torch.Tensor
    pi2_0: torch.Tensor
    pi2_1: torch.Tensor
    mu1_1: torch.Tensor
    mu2_1: torch.Tensor
    mu1_0: torch.Tensor | None
    mu2_0: torch.Tensor | None
    theta_true: float
    ate_true: float | None


def generate_linear(
    n: int,
    func_name: str,
    lower: float = 0.10,
    upper: float = 0.90,
) -> SimulationData:
    """Linear outcome DGP (time_varying_treatment/run_sim.py)."""
    dim_x1, dim_x2 = 5, 5
    treatment_probability_func = get_treatment_func(func_name, lower, upper)

    beta_pi1_0 = 0
    beta_pi1_s1 = torch.tensor([1, 1, 1] + [0] * (dim_x1 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_pi2_0_0 = 0
    beta_pi2_0_s1 = torch.tensor([0.5, 0, -0.5] + [0] * (dim_x1 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_pi2_0_s2 = torch.tensor([0.5, 0, 0.5] + [0] * (dim_x2 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_pi2_1_0 = 0
    beta_pi2_1_s1 = torch.tensor([1, 1, 0] + [0] * (dim_x1 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_pi2_1_s2 = torch.tensor([1, -1, 0] + [0] * (dim_x2 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_g0_0 = 1
    beta_g0_s1 = torch.tensor([1, 1, -1] + [0] * (dim_x1 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_g0_s2 = torch.tensor([1, 1, 1] + [0] * (dim_x2 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_g1_0 = -1
    beta_g1_s1 = torch.tensor([-1, 1, -1] + [0] * (dim_x1 - 3), dtype=torch.float32).reshape(-1, 1)
    beta_g1_s2 = torch.tensor([-1, -1, 1] + [0] * (dim_x2 - 3), dtype=torch.float32).reshape(-1, 1)

    s1 = torch.randn(n, dim_x1)
    pi1 = treatment_probability_func(beta_pi1_0 + s1 @ beta_pi1_s1).reshape(-1, 1)
    a1 = torch.bernoulli(pi1).int().reshape(-1, 1)

    delta1 = torch.randn(n, 1)
    delta2 = torch.randn(n, dim_x2)

    s2 = s1 + a1 * (1 + delta1) + delta2
    s2_1 = s1 + 1 + delta1 + delta2
    s2_0 = s1 + delta2

    pi2_0 = treatment_probability_func(beta_pi2_0_0 + s1 @ beta_pi2_0_s1 + s2_0 @ beta_pi2_0_s2)
    pi2_1 = treatment_probability_func(beta_pi2_1_0 + s1 @ beta_pi2_1_s1 + s2_1 @ beta_pi2_1_s2)
    pi2 = (1 - a1) * pi2_0 + a1 * pi2_1
    a2 = torch.bernoulli(pi2).int()

    g = (
        (a1 + a2 == 0).float() * (beta_g0_0 + s1 @ beta_g0_s1 + s2 @ beta_g0_s2)
        + (a1 * a2 == 1).float() * (beta_g1_0 + s1 @ beta_g1_s1 + s2 @ beta_g1_s2)
    )
    y = g + torch.randn(n, 1)

    mu2_1 = beta_g1_0 + s1 @ beta_g1_s1 + s2_1 @ beta_g1_s2
    mu1_1 = beta_g1_0 + s1 @ (beta_g1_s1 + beta_g1_s2) + beta_g1_s2.sum()
    theta_true = _to_float(beta_g1_0 + beta_g1_s2.sum())

    mu2_0 = beta_g0_0 + s1 @ beta_g0_s1 + s2_0 @ beta_g0_s2
    mu1_0 = beta_g0_0 + s1 @ (beta_g0_s1 + beta_g0_s2)
    theta0 = _to_float(beta_g0_0)
    ate_true = theta_true - theta0

    x = torch.hstack((s1, s2))
    x_index = torch.tensor([s1.shape[1] - 1, s1.shape[1] + s2.shape[1] - 1])
    d = torch.hstack((a1, a2))

    return SimulationData(
        Y=y,
        X=x,
        D=d,
        X_index=x_index,
        pi1=pi1,
        pi2_0=pi2_0,
        pi2_1=pi2_1,
        mu1_1=mu1_1,
        mu2_1=mu2_1,
        mu1_0=mu1_0,
        mu2_0=mu2_0,
        theta_true=theta_true,
        ate_true=ate_true,
    )


def generate_nonlinear(
    n: int,
    func_name: str,
    lower: float = 0.10,
    upper: float = 0.90,
) -> SimulationData:
    """Nonlinear outcome DGP (time_varying_treatment/run_sim_nonlinear.py)."""
    dim_x1, dim_x2 = 2, 2
    treatment_probability_func = get_treatment_func(func_name, lower, upper)

    beta_pi1_0 = 0
    beta_pi1_x1 = torch.tensor([1.0] + [0.0] * (dim_x1 - 1), dtype=torch.float32).reshape(-1, 1)
    beta_pi2_0_0 = 0
    beta_pi2_0_x2 = torch.tensor([0.5] + [0.0] * (dim_x2 - 1), dtype=torch.float32).reshape(-1, 1)
    beta_pi2_1_0 = 0
    beta_pi2_1_x2 = torch.tensor([1.0] + [0.0] * (dim_x2 - 1), dtype=torch.float32).reshape(-1, 1)

    c1, c2, c12 = 2.2, 1.2, 0.5
    beta1, beta2 = 1.2, 0.8
    delta1, delta2, delta12 = 1.0, 1.0, 0.5

    nd_theta = Normal(torch.tensor(1.0), torch.sqrt(torch.tensor(3.0)))
    cdf_theta = (1.0 - nd_theta.cdf(torch.tensor(0.0))).item()
    theta_true = c1 + c2 + c12 + 0.5 * beta1 + cdf_theta * beta2 + (delta2 + delta12) * 1.0
    theta_true_00 = 0.5 * beta1 + 0.5 * beta2
    ate_true = theta_true - theta_true_00

    s1 = torch.randn(n, dim_x1)
    pi1 = treatment_probability_func(beta_pi1_0 + s1 @ beta_pi1_x1).reshape(-1, 1)
    a1 = torch.bernoulli(pi1).int().reshape(-1, 1)

    delta1_noise = torch.randn(n, 1)
    delta2_noise = torch.randn(n, dim_x2)

    s2 = s1 + a1 * (1 + delta1_noise) + delta2_noise
    s2_1 = s1 + 1 + delta1_noise + delta2_noise
    s2_0 = s1 + delta2_noise

    pi2_0 = treatment_probability_func(beta_pi2_0_0 + s2_0 @ beta_pi2_0_x2)
    pi2_1 = treatment_probability_func(beta_pi2_1_0 + s2_1 @ beta_pi2_1_x2)
    pi2 = (1 - a1) * pi2_0 + a1 * pi2_1
    a2 = torch.bernoulli(pi2).int()

    g = (
        s1[:, :1]
        + c1 * a1
        + c2 * a2
        + c12 * a1 * a2
        + beta1 * (s1[:, :1] > 0).float()
        + beta2 * (s2[:, :1] > 0).float()
        + delta1 * a1 * s1[:, :1]
        + delta2 * a2 * s2[:, :1]
        + delta12 * a1 * a2 * s2[:, :1]
    )
    y = g + torch.randn(n, 1)

    mu2_1 = (
        s1[:, :1]
        + c1
        + c2
        + c12
        + beta1 * (s1[:, :1] > 0).float()
        + beta2 * (s2_1[:, :1] > 0).float()
        + delta1 * s1[:, :1]
        + delta2 * s2_1[:, :1]
        + delta12 * s2_1[:, :1]
    )

    nd_mu1 = Normal(torch.tensor(0.0), torch.sqrt(torch.tensor(2.0)))
    cdf_mu1 = 1.0 - nd_mu1.cdf(-1.0 - s1[:, :1])
    mu1_1 = (
        s1[:, :1]
        + c1
        + c2
        + c12
        + beta1 * (s1[:, :1] > 0).float()
        + beta2 * cdf_mu1
        + delta1 * s1[:, :1]
        + (delta2 + delta12) * (1.0 + s1[:, :1])
    )

    mu2_0 = s1[:, :1] + beta1 * (s1[:, :1] > 0).float() + beta2 * (s2_0[:, :1] > 0).float()
    nd_std = Normal(torch.tensor(0.0), torch.tensor(1.0))
    cdf_mu1_0 = 1.0 - nd_std.cdf(-s1[:, :1])
    mu1_0 = s1[:, :1] + beta1 * (s1[:, :1] > 0).float() + beta2 * cdf_mu1_0

    x = torch.hstack((s1, s2))
    x_index = torch.tensor([s1.shape[1] - 1, s1.shape[1] + s2.shape[1] - 1])
    d = torch.hstack((a1, a2))

    return SimulationData(
        Y=y,
        X=x,
        D=d,
        X_index=x_index,
        pi1=pi1,
        pi2_0=pi2_0,
        pi2_1=pi2_1,
        mu1_1=mu1_1,
        mu2_1=mu2_1,
        mu1_0=mu1_0,
        mu2_0=mu2_0,
        theta_true=theta_true,
        ate_true=ate_true,
    )


def generate(
    dgp: str,
    n: int,
    func_name: str,
    lower: float = 0.10,
    upper: float = 0.90,
) -> SimulationData:
    if dgp == "linear":
        return generate_linear(n, func_name, lower, upper)
    if dgp == "nonlinear":
        return generate_nonlinear(n, func_name, lower, upper)
    raise ValueError(f"Unknown DGP type: {dgp}")
