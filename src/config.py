from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
RESPONSE_COLUMN = "response"
NUMERIC_FEATURES = [
    "text_len",
    "num_turns",
    "risk_keyword_count",
    "question_mark_count",
    "exclamation_count",
]
