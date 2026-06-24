#!/usr/bin/env python3
"""Run the time-varying treatment simulation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(name: str, label: str, extra: list[str] | None = None) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(label, flush=True)
    print(f"{'=' * 60}", flush=True)
    cmd = [PYTHON, str(CODE_DIR / name)]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time-varying treatment simulation pipeline.")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", choices=[
        "linear_truncated_logistic",
        "nonlinear_truncated_adv",
        "linear_truncated_adv",
        "linear_logistic",
    ])
    parser.add_argument("--N", type=int, choices=[500, 1000, 2000])
    parser.add_argument("--iteration", type=int, help="Single MC replication (0..499).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configs × N × 500 iterations locally (very slow).",
    )
    args = parser.parse_args()

    sim_extra: list[str] = []
    if args.force:
        sim_extra.append("--force")
    if args.all:
        sim_extra.append("--all")
    if args.config is not None:
        sim_extra.extend(["--config", args.config])
    if args.N is not None:
        sim_extra.extend(["--N", str(args.N)])
    if args.iteration is not None:
        sim_extra.extend(["--iteration", str(args.iteration)])

    collect_extra = ["--force"] if args.force else []

    if not args.skip_simulation:
        on_cluster = "SLURM_ARRAY_TASK_ID" in os.environ
        if not on_cluster and not args.all and (args.config is None or args.N is None):
            print(
                "Local default: specify --config and --N, or --all for full sweep.\n"
                "On cluster, submit run_simulations.sbatch instead."
            )
            if args.config is None or args.N is None:
                raise SystemExit(2)
        run_script("1_run_simulation.py", "1. Run Simulation", sim_extra or None)
    if not args.skip_collect:
        run_script("2_collect_results.py", "2. Collect Results", collect_extra)


if __name__ == "__main__":
    main()
