#!/usr/bin/env python3
"""Download min-wage application data to data/raw/."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "minwage_data.csv"
DATA_URL = "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/minwage_data.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        print("1. Fetch Data — already exists, skipping.")
        return
    print("1. Fetch Data")
    urlretrieve(DATA_URL, RAW_PATH)
    print("   Complete.")


if __name__ == "__main__":
    main()
