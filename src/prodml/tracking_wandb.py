"""
Weights & Biases (W&B) experiment tracking module for prodml.
"""

import os
from typing import Any, Dict, Optional, Tuple
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
import wandb

from prodml.config import DEFAULT_THRESHOLD, RANDOM_STATE
from prodml.logging_conf import logger

WANDB_PROJECT_NAME = "online-shoppers-purchasing-intent"


def run_wandb_experiment(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    mode: str = "offline",
) -> Dict[str, float]:
    """
    Runs a single Weights & Biases experiment run logging parameters and metrics.
    Defaults to 'offline' mode for non-interactive automated execution.
    """
    try:
        os.environ["WANDB_MODE"] = mode
        run = wandb.init(
            project=WANDB_PROJECT_NAME,
            name=run_name,
            config=params,
            reinit=True,
            mode=mode,
        )
    except Exception as exc:
        logger.warning(
            f"Failed to initialize W&B in mode '{mode}', falling back to 'disabled' mode",
            extra={"error": str(exc)},
        )
        os.environ["WANDB_MODE"] = "disabled"
        run = wandb.init(
            project=WANDB_PROJECT_NAME,
            name=run_name,
            config=params,
            reinit=True,
            mode="disabled",
        )

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Compute predictions & metrics
    y_prob = model.predict_proba(X_test)[:, 1]
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

    # Log metrics to W&B
    wandb.log(metrics)
    wandb.finish()

    logger.info(f"W&B Run {run_name} completed successfully", extra=metrics)
    return metrics
