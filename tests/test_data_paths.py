import pathlib
import sys

# Ensure the project root is in sys.path for imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from app import load_data
from paths import PROCESSED_TRAIN_PATH

def test_load_data_returns_expected_columns():
    df = load_data()
    required_cols = {
        "is_high_risk",
        "AvgAmount",
        "TransactionCount",
        "TotalAmount",
        "Recency",
        "Frequency",
        "Monetary",
        "FraudRate",
        "StdAmount",
        "CreditRatio",
        "DebitRatio",
    }
    if PROCESSED_TRAIN_PATH.is_file():
        assert "is_high_risk" in df.columns
        assert len(df) > 2
    else:
        assert set(df.columns) == required_cols
        assert len(df) == 2  # synthetic demo data has two rows

def test_dockerfile_copy_line_commented():
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    content = dockerfile_path.read_text()
    # The line should be commented out
    assert "# COPY data/processed_customers/ ./data/processed_customers/" in content
    # Ensure no uncommented COPY line exists (ignoring leading whitespace)
    lines = content.splitlines()
    assert not any(line.lstrip().startswith("COPY data/processed_customers/ ./data/processed_customers/") for line in lines)

