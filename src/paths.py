"""
Centralized path configuration for the credit risk project.

All scripts should import paths from here to keep data locations consistent
across pipeline, training, dashboard, and API components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "data.csv"
VARIABLE_DEFINITIONS_PATH = RAW_DATA_DIR / "Xente_Variable_Definitions.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_data.csv"

# Legacy alias kept for backward compatibility in docs and migration notes
LEGACY_PROCESSED_PATH = PROCESSED_DATA_DIR / "processed_customers.csv"

MODEL_PATH = MODEL_DIR / "DecisionTree_best.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.joblib"
METRICS_PATH = MODEL_DIR / "metrics_summary.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable container for commonly used project paths."""

    project_root: Path = PROJECT_ROOT
    raw_data: Path = RAW_DATA_PATH
    processed_train: Path = PROCESSED_TRAIN_PATH
    model_dir: Path = MODEL_DIR
    model: Path = MODEL_PATH
    features: Path = FEATURES_PATH
    metrics: Path = METRICS_PATH
    comparison: Path = COMPARISON_PATH

    def ensure_data_dirs(self) -> None:
        """Create data directories if they do not exist."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths()


def resolve_processed_data(explicit_path: Optional[str] = None) -> Path:
    """
    Resolve the processed training CSV path.

    Checks explicit path first, then standard location, then legacy filename.
    """
    if explicit_path:
        return Path(explicit_path)

    candidates = (
        PROCESSED_TRAIN_PATH,
        LEGACY_PROCESSED_PATH,
        PROJECT_ROOT / "data" / "processed_customers.csv",
    )
    for path in candidates:
        if path.is_file():
            return path

    return PROCESSED_TRAIN_PATH
