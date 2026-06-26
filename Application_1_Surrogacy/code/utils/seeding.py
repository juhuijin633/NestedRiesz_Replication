"""Replication reproducibility for Application 1 (Surrogacy).

Upstream NestedRiesz (application_fit_final.py) used torch.manual_seed(0) only
before each auto estimator. This replication adds full RNG seeding, single-thread
BLAS, deterministic cuDNN, and CPU-only auto-NN so repeated runs on the pinned
riesz environment (setup/clean_requirements.txt) agree with each other.

Verify env: conda activate riesz && pip check against setup/clean_requirements.txt
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from utils.hyperparams import AUTO_SEED, MANUAL_SEED

_CONFIGURED = False


def configure_runtime() -> None:
    """Process-wide settings: call once at the start of estimate scripts."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    _CONFIGURED = True


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_auto(seed: int = AUTO_SEED) -> None:
    """Before each auto estimator (Net / Lasso / RF)."""
    configure_runtime()
    _seed_all(seed)


def seed_manual(seed: int = MANUAL_SEED) -> None:
    """Once per outcome before the manual estimator block."""
    configure_runtime()
    _seed_all(seed)
