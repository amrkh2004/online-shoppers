"""
Script to execute 3 MLflow runs with varied hyperparameters, log parameters, metrics, MAE, and artifacts,
and transition the best performing model to 'Staging' in the MLflow Model Registry.
"""

from prodml.config import BASE_DIR
from prodml.data import clean_data, load_data, split_data
from prodml.features import FeaturePipeline
from prodml.logging_conf import logger
from prodml.tracking_mlflow import register_best_model_to_staging, run_mlflow_experiment


def main():
    logger.info("=== Running 3 MLflow Experiments ===")

    raw_data_path = BASE_DIR / "notebooks" / "online_shoppers_intention.csv"
    if not raw_data_path.exists():
        possible_paths = list(BASE_DIR.glob("**/*.csv"))
        if possible_paths:
            raw_data_path = possible_paths[0]

    df = load_data(raw_data_path)
    df_cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_cleaned)

    pipeline = FeaturePipeline()
    X_train_scaled, _ = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    # 3 hyperparameter experiment configurations
    experiment_configs = [
        {
            "run_name": "Run_1_Baseline_RF",
            "params": {
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "run_name": "Run_2_Tuned_RF_Large",
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "run_name": "Run_3_Regularized_RF",
            "params": {
                "n_estimators": 150,
                "max_depth": 15,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    ]

    for config in experiment_configs:
        run_id, metrics = run_mlflow_experiment(
            params=config["params"],
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            run_name=config["run_name"],
        )
        logger.info(
            f"Finished {config['run_name']} (Run ID: {run_id}) - F1: {metrics['f1_score']:.4f}, MAE: {metrics['mae']:.4f}"
        )

    # Register best model to Staging
    logger.info("Promoting best model run to Staging stage in MLflow Model Registry...")
    best_run_id, version = register_best_model_to_staging()
    logger.info(
        f"=== Successfully Registered Best Model (Run: {best_run_id}) to Staging (Version {version}) ==="
    )


if __name__ == "__main__":
    main()
