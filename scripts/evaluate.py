"""
DVC Stage 3: Model Evaluation script outputting metrics.json.
"""

import json

import pandas as pd
import yaml

from prodml.config import BASE_DIR, TARGET_COL
from prodml.logging_conf import logger
from prodml.predict import OnlineShoppersPredictor
from prodml.train import evaluate_model


def main():
    logger.info("=== DVC Stage 3: Evaluate Model ===")

    params_file = BASE_DIR / "params.yaml"
    with open(params_file, "r") as f:
        params = yaml.safe_load(f).get("evaluate", {})

    threshold = params.get("threshold", 0.49)

    processed_dir = BASE_DIR / "data" / "processed"
    test_df = pd.read_csv(processed_dir / "test.csv")

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].astype(int)

    predictor = OnlineShoppersPredictor(threshold=threshold)
    X_test_scaled = predictor.pipeline.transform(X_test)

    metrics = evaluate_model(predictor.model, X_test_scaled, y_test, threshold=threshold)

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = reports_dir / "metrics.json"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Evaluation metrics exported", extra={"metrics_file": str(metrics_file), "metrics": metrics}
    )


if __name__ == "__main__":
    main()
