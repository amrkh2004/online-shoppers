"""
Script to train the model, fit feature pipeline, evaluate performance, and export all model artifacts (pickle & ONNX).
"""

from pathlib import Path
from prodml.config import BASE_DIR
from prodml.data import clean_data, load_data, split_data
from prodml.export import export_model_artifacts, verify_onnx_parity
from prodml.features import FeaturePipeline
from prodml.logging_conf import logger
from prodml.train import evaluate_model, train_tuned_model


def main():
    logger.info("--- Starting Full Model Training and Export Workflow ---")

    # 1. Load Data
    raw_data_path = BASE_DIR / "notebooks" / "online_shoppers_intention.csv"
    if not raw_data_path.exists():
        # Fallback to checking online shoppers dataset inside notebooks or root
        possible_paths = list(BASE_DIR.glob("**/*.csv"))
        if possible_paths:
            raw_data_path = possible_paths[0]
        else:
            logger.info("CSV raw file not found directly, generating training data from notebook split...")
            # We can extract dataset from notebook or download/create sample
            raw_data_path = BASE_DIR / "notebooks" / "online_shoppers_intention.csv"

    logger.info(f"Using dataset path: {raw_data_path}")

    # If csv doesn't exist yet, we will generate CSV from notebook data if available
    if not raw_data_path.exists():
        logger.info("Extracting raw dataset from notebook...")
        import json
        import pandas as pd
        
        # Load data using python script from notebook or create dummy/stored dataset
        logger.warning("Dataset CSV missing. Please ensure online_shoppers_intention.csv is placed under notebooks/")
        return

    df = load_data(raw_data_path)
    df_cleaned = clean_data(df)

    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df_cleaned)

    # 3. Fit Feature Pipeline & Scale Data
    pipeline = FeaturePipeline()
    X_train_scaled, feature_names = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    # 4. Train Model
    model = train_tuned_model(X_train_scaled, y_train)

    # 5. Evaluate Model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    logger.info("Evaluation metrics on Test set", extra=metrics)

    # 6. Export Model Artifacts (.pkl & .onnx)
    artifacts = export_model_artifacts(
        model=model,
        scaler=pipeline.scaler,
        feature_names=feature_names,
    )
    logger.info("Exported Artifacts", extra=artifacts)

    # 7. Run ONNX Parity Test & Latency Benchmark
    parity_metrics = verify_onnx_parity(sample_data=X_test_scaled.iloc[:100])
    logger.info("ONNX Parity & Latency Benchmark Results", extra=parity_metrics)


if __name__ == "__main__":
    main()
