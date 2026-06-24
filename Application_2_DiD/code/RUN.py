#!/usr/bin/env python3
"""Run the full DiD application pipeline."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Run DiD application pipeline.")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-estimates", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute estimates.")
    args = parser.parse_args()

    extra = ["--force"] if args.force else []

    if not args.skip_fetch:
        run_script("1_fetch_data.py", "1. Fetch Data")
    if not args.skip_estimates:
        run_script("2_calc_estimates.py", "2. Calc Estimates", extra)
    if not args.skip_figures:
        run_script("3_tables_figs.py", "3. Tables & Figures")


if __name__ == "__main__":
    main()
