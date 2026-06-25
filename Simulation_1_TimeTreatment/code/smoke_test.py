#!/usr/bin/env python3
"""One-replication pre-flight check (run on cluster before sbatch).

Usage:
    cd Simulation_1_TimeTreatment/code
    module load python/3.10.9-fasrc01 R/4.4.3-fasrc01
    conda activate riesz
    python smoke_test.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

run = importlib.import_module("1_run_simulation")

if __name__ == "__main__":
    config = run.CONFIGS[0]
    run._run_one(config, n=100, t=0, force=True)
    print("Smoke test passed.")
