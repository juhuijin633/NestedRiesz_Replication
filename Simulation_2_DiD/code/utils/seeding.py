"""Replication reproducibility for Application 2 (DiD)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

_CONFIGURED = False


def configure_runtime() -> None:
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


def seed_all(seed: int | None) -> None:
    if seed is None:
        return
    configure_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
