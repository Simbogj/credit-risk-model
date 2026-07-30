"""Tests for centralized path configuration and reproducibility."""

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import (
    LEGACY_PROCESSED_PATH,
    PROCESSED_TRAIN_PATH,
    RAW_DATA_PATH,
    resolve_processed_data,
)


def test_standard_processed_path_is_train_data_csv():
    assert PROCESSED_TRAIN_PATH.name == "train_data.csv"
    assert PROCESSED_TRAIN_PATH.parent.name == "processed"


def test_legacy_path_differs_from_standard():
    assert LEGACY_PROCESSED_PATH != PROCESSED_TRAIN_PATH


def test_resolve_processed_data_default():
    assert resolve_processed_data() == PROCESSED_TRAIN_PATH


def test_resolve_processed_data_explicit_override(tmp_path):
    custom = tmp_path / "custom.csv"
    custom.write_text("a\n1")
    assert resolve_processed_data(str(custom)) == custom


def test_raw_data_expected_location():
    assert RAW_DATA_PATH.parent.name == "raw"
    assert RAW_DATA_PATH.name == "data.csv"
