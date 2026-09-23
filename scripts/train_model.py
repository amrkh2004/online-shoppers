"""
DVC Stage 2: Model Training script.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml

from prodml.config import BASE_DIR, TARGET_COL
from prodml.export import export_model_artifacts
from prodml.features import FeaturePipeline
from prodml.logging_conf import logger


def main():
    logger.info("=== DVC Stage 2: Train Model ===")

    params_file = BASE_DIR / "params.yaml"
    with open(params_file, "r") as f:
        params = yaml.safe_load(f).get("train", {})

    processed_dir = BASE_DIR / "data" / "processed"
    train_df = pd.read_csv(processed_dir / "train.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(int)

    pipeline = FeaturePipeline()
    X_train_scaled, feature_names = pipeline.fit_transform(X_train)

    rf_params = {
        "n_estimators": params.get("n_estimators", 300),
        "max_depth": params.get("max_depth", None),
        "min_samples_split": params.get("min_samples_split", 2),
        "min_samples_leaf": params.get("min_samples_leaf", 1),
        "random_state": params.get("random_state", 42),
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_scaled, y_train)

    export_model_artifacts(
        model=model,
        scaler=pipeline.scaler,
        feature_names=feature_names,
    )

    logger.info("Trained and exported model artifacts via DVC train stage.")


if __name__ == "__main__":
    main()
