"""
Configuration settings for the prodml package.
"""

from pathlib import Path
from typing import List

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH_PKL = MODELS_DIR / "final_random_forest.pkl"
SCALER_PATH_PKL = MODELS_DIR / "scaler.pkl"
MODEL_PATH_ONNX = MODELS_DIR / "final_random_forest.onnx"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Model Parameters
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.49

# Feature Definitions
CATEGORICAL_FEATURES: List[str] = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

NUMERICAL_FEATURES: List[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

BOOLEAN_FEATURES: List[str] = ["Weekend"]
TARGET_COL: str = "Revenue"
