"""
Model training and evaluation module for prodml.
"""

from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prodml.config import DEFAULT_THRESHOLD, RANDOM_STATE
from prodml.logging_conf import logger


def train_baseline_model(
    X_train_scaled: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Trains baseline Random Forest Classifier.
    """
    logger.info("Training baseline Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    logger.info("Baseline Random Forest Classifier trained successfully.")
    return model


def train_tuned_model(X_train_scaled: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains tuned Random Forest Classifier using optimal parameters found during EDA.
    """
    logger.info("Training tuned Random Forest Classifier (n_estimators=300)...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    logger.info("Tuned Random Forest Classifier trained successfully.")
    return model


def evaluate_model(
    model: Any,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Evaluates model performance against target metrics using probability thresholding.
    """
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "mae": float(mean_absolute_error(y_test, y_prob)),
        "threshold": threshold,
    }

    logger.info("Model evaluation completed", extra=metrics)
    return metrics
