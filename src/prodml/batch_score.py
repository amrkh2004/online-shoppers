"""
Batch Scorer module (Deliverable 03)
Reads Parquet input data, loads the Production model from MLflow Registry,
performs batch predictions, adds run_date, and writes output Parquet.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

from prodml.config import BASE_DIR
from prodml.logging_conf import logger


def load_model_from_registry(
    model_name: str = "RideDurationModel",
    stage: str = "Production",
    tracking_uri: Optional[str] = None,
):
    """
    Loads model from MLflow Registry given model name and stage.
    """
    if tracking_uri is None:
        db_path = BASE_DIR / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"

    try:
        logger.info(f"Loading MLflow model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.warning(f"Could not load model from {model_uri}: {e}. Trying fallback models...")
        for fallback_name in ["OnlineShoppersRF", "RideDurationModel"]:
            for fallback_stage in [stage, "Staging", "latest"]:
                try:
                    fallback_uri = f"models:/{fallback_name}/{fallback_stage}"
                    logger.info(f"Attempting fallback load: {fallback_uri}")
                    return mlflow.pyfunc.load_model(fallback_uri)
                except Exception:
                    continue
        raise RuntimeError(f"Unable to load any model from MLflow registry: {e}")


def batch_score(
    input_path: Path,
    output_path: Path,
    model_name: str = "RideDurationModel",
    stage: str = "Production",
    model_obj=None,
) -> pd.DataFrame:
    """
    Reads input Parquet file, applies predictions, appends `run_date`, and saves to output Parquet.
    """
    logger.info(f"Starting batch scoring for input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # Read input data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Load model if not directly provided
    if model_obj is None:
        model_obj = load_model_from_registry(model_name=model_name, stage=stage)

    predictions = model_obj.predict(df)

    # Store results
    df_out = df.copy()
    df_out["prediction"] = predictions
    df_out["run_date"] = datetime.now(timezone.utc).isoformat()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    logger.info(f"Batch scoring completed. Saved {len(df_out)} predictions to {output_path}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Batch Scorer CLI for MLflow Production Models")
    parser.add_argument(
        "--input",
        type=str,
        default=str(BASE_DIR / "data" / "scoring" / "input" / "scoring_input.parquet"),
        help="Path to input Parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "data" / "scoring" / "output" / "scoring_output.parquet"),
        help="Path to output Parquet file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="RideDurationModel",
        help="Registered MLflow model name",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        help="MLflow model stage (e.g. Production, Staging)",
    )

    args = parser.parse_args()
    batch_score(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model_name,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
