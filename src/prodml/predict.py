"""
Inference prediction engine for prodml.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from prodml.config import (
    DEFAULT_THRESHOLD,
    FEATURE_NAMES_PATH,
    MODEL_PATH_PKL,
    SCALER_PATH_PKL,
)
from prodml.features import FeaturePipeline
from prodml.logging_conf import logger


class OnlineShoppersPredictor:
    """
    Predictor class loading trained model, scaler, and feature metadata to produce predictions.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH_PKL,
        scaler_path: Path = SCALER_PATH_PKL,
        feature_names_path: Path = FEATURE_NAMES_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names_path = feature_names_path
        self.threshold = threshold

        self.model = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self.pipeline = FeaturePipeline()

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Loads exported joblib model, scaler, and feature list.
        """
        logger.info("Loading predictor model artifacts...")
        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path} or {self.scaler_path}. Please train and export model first."
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if self.feature_names_path.exists():
            with open(self.feature_names_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)
        elif hasattr(self.model, "feature_names_in_"):
            self.feature_columns = list(self.model.feature_names_in_)
        else:
            raise ValueError("Feature columns metadata missing.")

        self.pipeline.scaler = self.scaler
        self.pipeline.feature_columns = self.feature_columns
        self.pipeline.is_fitted = True

        logger.info(
            "Predictor model artifacts loaded successfully",
            extra={
                "model_type": type(self.model).__name__,
                "num_features": len(self.feature_columns),
                "threshold": self.threshold,
            },
        )

    def predict_dataframe(self, df: pd.DataFrame, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Produces predictions and probability scores for a pandas DataFrame.
        """
        if threshold is None:
            threshold = self.threshold

        # Preprocess & align features
        X_scaled = self.pipeline.transform(df)

        # Get probabilities for positive class (Revenue = 1 / Purchase)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        results = []
        for prob in probabilities:
            prediction = int(prob >= threshold)
            label = "Purchase" if prediction == 1 else "No Purchase"
            results.append(
                {
                    "prediction": prediction,
                    "label": label,
                    "probability": float(round(prob, 4)),
                    "threshold_used": threshold,
                }
            )

        return results

    def predict_dict(self, record: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
        """
        Produces prediction for a single dictionary record.
        """
        df = pd.DataFrame([record])
        return self.predict_dataframe(df, threshold=threshold)[0]
