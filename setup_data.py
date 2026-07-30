"""
Verify data availability and optionally run the processing pipeline.

Usage:
    python setup_data.py          # check status only
    python setup_data.py --run    # run pipeline when raw data exists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from paths import PATHS, RAW_DATA_PATH, PROCESSED_TRAIN_PATH, VARIABLE_DEFINITIONS_PATH


def check_data() -> int:
    """Print data status and return exit code (0 = ready, 1 = action needed)."""
    PATHS.ensure_data_dirs()

    print("Credit Risk Model - Data Setup")
    print("=" * 40)
    print(f"Raw data:       {RAW_DATA_PATH} {'[OK]' if RAW_DATA_PATH.is_file() else '[MISSING]'}")
    print(
        f"Definitions:    {VARIABLE_DEFINITIONS_PATH} "
        f"{'[OK]' if VARIABLE_DEFINITIONS_PATH.is_file() else '[optional]'}"
    )
    print(
        f"Processed data: {PROCESSED_TRAIN_PATH} "
        f"{'[OK]' if PROCESSED_TRAIN_PATH.is_file() else '[MISSING]'}"
    )
    print()

    if PROCESSED_TRAIN_PATH.is_file():
        print("Ready: processed training data is available.")
        return 0

    if RAW_DATA_PATH.is_file():
        print("Action: run `python setup_data.py --run` or `python run_pipeline.py`")
        return 1

    print(
        "Action: place the Xente transaction CSV at data/raw/data.csv\n"
        "        (dataset: https://www.kaggle.com/datasets/xente-challenge/xente-fraud-detection)\n"
        "        Then run: python setup_data.py --run"
    )
    return 1


def run_pipeline() -> int:
    if not RAW_DATA_PATH.is_file():
        print(f"Cannot run pipeline: raw data not found at {RAW_DATA_PATH}")
        return 1

    print("Running data processing pipeline...")
    result = subprocess.run([sys.executable, "run_pipeline.py"], check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or generate project data")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the processing pipeline when raw data is available",
    )
    args = parser.parse_args()

    if args.run:
        return run_pipeline()
    return check_data()


if __name__ == "__main__":
    raise SystemExit(main())
