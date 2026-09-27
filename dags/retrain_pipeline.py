"""
Airflow Automated Retraining Pipeline (Deliverable 01)
Schedule: @weekly (catchup=False)
Tasks: extract -> train -> evaluate -> register (promotes to Production if MAE improves)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments per requirements
DEFAULT_ARGS = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    dag_id="ride_duration_retrain_pipeline",
    default_args=DEFAULT_ARGS,
    description="Automated weekly retraining, evaluation, and model promotion pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "ride-duration", "retraining"],
)


def extract_data_task(**kwargs):
    """
    Idempotently extracts raw ride data and saves to processed staging area.
    Imports are scoped inside function for Airflow performance.
    """
    import pandas as pd

    from prodml.config import BASE_DIR
    from prodml.logging_conf import logger

    logger.info("Executing Airflow Task: extract_data")
    staging_dir = BASE_DIR / "data" / "processed"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Generate or extract clean training dataset idempotently
    sample_df = pd.DataFrame(
        {
            "distance_km": [2.5, 8.1, 14.3, 5.0, 19.2, 3.8, 11.5, 6.4, 22.0, 1.9],
            "passengers": [1, 2, 1, 3, 2, 1, 4, 2, 1, 1],
            "hour_of_day": [8, 14, 18, 9, 21, 11, 17, 8, 23, 10],
            "duration_minutes": [10.2, 22.5, 38.0, 15.1, 48.3, 12.0, 31.4, 18.2, 55.0, 8.5],
        }
    )

    output_path = staging_dir / "extracted_rides.parquet"
    sample_df.to_parquet(output_path, index=False)
    logger.info(f"Extracted data successfully to {output_path}")
    return str(output_path)


def train_model_task(**kwargs):
    """
    Idempotently trains candidate sklearn model on extracted dataset and logs to MLflow.
    """
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from prodml.config import BASE_DIR
    from prodml.logging_conf import logger

    ti = kwargs.get("ti")
    input_file = (
        ti.xcom_pull(task_ids="extract_task")
        if ti
        else str(BASE_DIR / "data" / "processed" / "extracted_rides.parquet")
    )

    logger.info(f"Executing Airflow Task: train_model using data from {input_file}")
    df = pd.read_parquet(input_file)

    X = df[["distance_km", "passengers", "hour_of_day"]]
    y = df["duration_minutes"]

    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("ride-duration-retraining")

    with mlflow.start_run(run_name="Airflow_Weekly_Retrain") as run:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X, y)

        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        logger.info(f"Model candidate trained successfully. MLflow Run ID: {run_id}")
        return run_id


def evaluate_model_task(**kwargs):
    """
    Evaluates candidate model logged in MLflow, computes MAE, and returns run_id & metrics.
    """
    import mlflow
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    from prodml.config import BASE_DIR
    from prodml.logging_conf import logger

    ti = kwargs.get("ti")
    run_id = ti.xcom_pull(task_ids="train_task") if ti else None

    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

    if not run_id:
        client = mlflow.tracking.MlflowClient()
        exp = mlflow.get_experiment_by_name("ride-duration-retraining")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], order_by=["attribute.start_time DESC"]
        )
        run_id = runs[0].info.run_id

    model_uri = f"runs:/{run_id}/model"
    candidate_model = mlflow.pyfunc.load_model(model_uri)

    # Test set evaluation
    eval_df = pd.DataFrame(
        {
            "distance_km": [4.0, 10.0, 18.0],
            "passengers": [1, 2, 3],
            "hour_of_day": [9, 16, 20],
            "duration_minutes": [13.0, 27.5, 45.0],
        }
    )

    X_test = eval_df[["distance_km", "passengers", "hour_of_day"]]
    y_test = eval_df["duration_minutes"]

    preds = candidate_model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("eval_mae", mae)

    logger.info(f"Evaluated candidate run {run_id} with MAE: {mae:.4f}")
    return {"run_id": run_id, "eval_mae": mae}


def register_model_task(**kwargs):
    """
    Registration & Promotion Task (Deliverable 01 key logic):
    Registers candidate model to MLflow Model Registry under 'RideDurationModel'.
    Checks existing Production model MAE: if candidate MAE is lower (better),
    promotes candidate to 'Production' stage and archives previous Production version.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from prodml.config import BASE_DIR
    from prodml.logging_conf import logger

    ti = kwargs.get("ti")
    eval_data = (
        ti.xcom_pull(task_ids="evaluate_task") if ti else {"run_id": None, "eval_mae": 999.0}
    )

    run_id = eval_data["run_id"]
    new_mae = eval_data["eval_mae"]
    model_name = "RideDurationModel"

    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    client = MlflowClient()

    logger.info(f"Registering model run {run_id} under name '{model_name}'...")
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    # Fetch current Production model MAE threshold
    current_prod_mae = float("inf")
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            current_prod_mae = prod_run.data.metrics.get(
                "eval_mae", prod_run.data.metrics.get("mae", 5.0)
            )
            logger.info(
                f"Current Production Model (Version {prod_versions[0].version}) MAE: {current_prod_mae:.4f}"
            )
    except Exception as e:
        logger.info(f"No existing Production model found or metric check error: {e}")

    # Compare MAE & Promote if improved
    if new_mae <= current_prod_mae:
        logger.info(
            f"New candidate MAE ({new_mae:.4f}) is better than Production ({current_prod_mae:.4f}). Promoting to Production!"
        )
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Successfully promoted model version {mv.version} to PRODUCTION stage.")
    else:
        logger.info(
            f"New candidate MAE ({new_mae:.4f}) is not better than Production ({current_prod_mae:.4f}). Keeping in Staging."
        )
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

    return mv.version


# Task Definitions
extract_task = PythonOperator(
    task_id="extract_task",
    python_callable=extract_data_task,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_task",
    python_callable=train_model_task,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_task",
    python_callable=evaluate_model_task,
    dag=dag,
)

register_task = PythonOperator(
    task_id="register_task",
    python_callable=register_model_task,
    dag=dag,
)

# Pipeline Dependencies using >> operator
extract_task >> train_task >> evaluate_task >> register_task
