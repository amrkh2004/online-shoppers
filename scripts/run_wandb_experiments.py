"""
Script to execute Weights & Biases (W&B) experiment tracking runs.
"""

from prodml.config import BASE_DIR
from prodml.data import clean_data, load_data, split_data
from prodml.features import FeaturePipeline
from prodml.logging_conf import logger
from prodml.tracking_wandb import run_wandb_experiment


def main():
    logger.info("=== Running Weights & Biases (W&B) Experiments ===")

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

    experiment_configs = [
        {
            "run_name": "WB_Run_1_Baseline",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "run_name": "WB_Run_2_Tuned",
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    ]

    for config in experiment_configs:
        metrics = run_wandb_experiment(
            params=config["params"],
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            run_name=config["run_name"],
            mode="offline",
        )
        logger.info(f"Finished W&B {config['run_name']} - F1: {metrics['f1_score']:.4f}, MAE: {metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
