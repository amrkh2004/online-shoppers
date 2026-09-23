"""
FastAPI Service Application for Online Shoppers Purchasing Intention Prediction.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status

from prodml import __version__
from prodml.api.schemas import (
    BatchPredictionResponseSchema,
    BatchShopperInputSchema,
    HealthResponseSchema,
    MetadataResponseSchema,
    PredictionResponseSchema,
    ShopperInputSchema,
)
from prodml.config import DEFAULT_THRESHOLD
from prodml.logging_conf import logger
from prodml.predict import OnlineShoppersPredictor

# Shared global predictor instance loaded once at startup
predictor: Optional[OnlineShoppersPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler ensuring model artifacts are loaded once at startup.
    """
    global predictor
    logger.info("Initializing FastAPI Service Startup...")
    try:
        predictor = OnlineShoppersPredictor()
        logger.info("Model loaded successfully at startup.")
    except Exception as exc:
        logger.warning(
            "Failed to load model artifacts on startup. API running in degraded state until model trained.",
            extra={"error": str(exc)},
        )
        predictor = None
    yield
    logger.info("Shutting down FastAPI Service.")


app = FastAPI(
    title="Online Shoppers Purchasing Intention ML API",
    description="Production-grade FastAPI service for predicting online shopper purchasing intent.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponseSchema, tags=["Health"])
def health_check():
    """
    Health check endpoint returning service status.
    """
    return {"status": "ok", "version": __version__}


@app.get("/metadata", response_model=MetadataResponseSchema, tags=["Metadata"])
def get_metadata():
    """
    Returns model metadata, baseline metrics, and loaded status.
    """
    num_features = len(predictor.feature_columns) if predictor else 0
    return {
        "model_name": "Tuned Random Forest Classifier",
        "version": __version__,
        "num_features": num_features,
        "classification_threshold": DEFAULT_THRESHOLD,
        "baseline_metrics": {
            "Baseline_Random_Forest_F1": 0.6749,
            "Baseline_XGBoost_F1": 0.6597,
            "Tuned_Random_Forest_F1": 0.6815,
            "Tuned_Random_Forest_ROC_AUC": 0.9217,
        },
    }


@app.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_single(input_data: ShopperInputSchema):
    """
    Predict purchasing intention for a single shopper visit.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure model artifacts are trained and exported.",
        )

    try:
        record = input_data.model_dump()
        result = predictor.predict_dict(record)
        return result
    except Exception as exc:
        logger.error("Error processing prediction request", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_batch(batch_data: BatchShopperInputSchema):
    """
    Predict purchasing intention for a batch of shopper visits.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure model artifacts are trained and exported.",
        )

    try:
        records = [item.model_dump() for item in batch_data.inputs]
        import pandas as pd

        df = pd.DataFrame(records)
        results = predictor.predict_dataframe(df)
        return {"predictions": results, "total_count": len(results)}
    except Exception as exc:
        logger.error("Error processing batch prediction request", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(exc)}",
        )
