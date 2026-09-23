"""
MLflow experiment tracking and model registry module for prodml.
"""

from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prodml.config import BASE_DIR, DEFAULT_THRESHOLD, RANDOM_STATE
from prodml.logging_conf import logger

MLFLOW_EXPERIMENT_NAME = "online-shoppers-purchasing-intent"


def setup_mlflow(tracking_uri: Optional[str] = None) -> None:
    """
    Sets up MLflow tracking URI and experiment.
    """
    if tracking_uri is None:
        db_path = BASE_DIR / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow configured successfully", extra={"tracking_uri": tracking_uri})


def run_mlflow_experiment(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[str, Dict[str, float]]:
    """
    Runs a single MLflow experiment run logging parameters, metrics, and model artifact.
    """
    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Starting MLflow Run: {run.info.run_id}", extra={"run_name": run_name})

        # Train model with parameters
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
        }

        # Log parameters & metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_params(
            {"threshold": threshold, "random_state": params.get("random_state", RANDOM_STATE)}
        )
        mlflow.log_metrics(metrics)

        # Log sklearn model artifact
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

        logger.info(
            f"MLflow Run {run.info.run_id} completed successfully",
            extra=metrics,
        )

        return run.info.run_id, metrics


def register_best_model_to_staging(
    registered_model_name: str = "OnlineShoppersRF",
) -> Tuple[str, str]:
    """
    Finds the best run by f1_score in MLflow, registers it in the Model Registry,
    and transitions its stage to 'Staging'.
    """
    setup_mlflow()
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=10,
    )

    if not runs:
        raise ValueError("No MLflow runs found to register.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("f1_score", 0.0)

    logger.info(
        f"Selected Best Run ID: {best_run_id} with F1-Score: {best_f1:.4f}",
        extra={"run_id": best_run_id, "f1_score": best_f1},
    )

    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, registered_model_name)

    # Transition model to Staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=True,
    )

    logger.info(
        f"Model '{registered_model_name}' version {model_version.version} transitioned to STAGING stage.",
        extra={"registered_model": registered_model_name, "version": model_version.version},
    )

    return best_run_id, model_version.version
