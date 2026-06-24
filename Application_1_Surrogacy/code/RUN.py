#!/usr/bin/env python3
"""Run the full surrogate application pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(name: str, label: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(label, flush=True)
    print(f"{'=' * 60}", flush=True)
    subprocess.run([PYTHON, str(CODE_DIR / name)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run surrogate application pipeline.")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-auto", action="store_true")
    parser.add_argument("--skip-manual", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    if not args.skip_clean:
        run_script("1_clean_data.py", "1. Clean Data")
    if not args.skip_auto:
        run_script("2_calc_auto_estimates.py", "2. Calculate Auto-Estimates")
    if not args.skip_manual:
        run_script("3_calc_manual_estimates.py", "3. Calculate Manual-Estimates")
    if not args.skip_figures:
        run_script("4_tables_figs.py", "4. Tables and Figures")

    print("\nPipeline Complete.", flush=True)


if __name__ == "__main__":
    main()
