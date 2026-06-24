#!/usr/bin/env python3
"""Run the DiD simulation pipeline."""

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
    parser = argparse.ArgumentParser(description="Run DiD simulation pipeline.")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--N", type=int, choices=[500, 1000, 2000])
    parser.add_argument(
        "--model",
        choices=["logistic", "truncated_logistic", "truncated_step"],
    )
    parser.add_argument("--all", action="store_true", help="Run all 9 jobs locally (default without SLURM).")
    args = parser.parse_args()

    sim_extra: list[str] = []
    if args.force:
        sim_extra.append("--force")
    if args.all:
        sim_extra.append("--all")
    if args.N is not None:
        sim_extra.extend(["--N", str(args.N)])
    if args.model is not None:
        sim_extra.extend(["--model", args.model])

    collect_extra = ["--force"] if args.force else []

    if not args.skip_simulation:
        on_cluster = "SLURM_ARRAY_TASK_ID" in os.environ
        if not on_cluster and not args.all and (args.N is None or args.model is None):
            sim_extra.append("--all")
        run_script("1_run_simulation.py", "1. Run Simulation", sim_extra or None)
    if not args.skip_collect:
        run_script("2_collect_results.py", "2. Collect Results", collect_extra)


if __name__ == "__main__":
    main()
