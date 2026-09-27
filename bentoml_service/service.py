"""
BentoML Web Service implementation (Deliverable 02)
Exposes micro-batched /predict and /healthz endpoints using MLflow Production model.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import bentoml

# Pydantic Schemas for API contracts
class PredictRequest(BaseModel):
    distance_km: float = Field(..., description="Trip distance in kilometers", example=7.5)
    passengers: int = Field(1, description="Number of passengers", example=2)
    hour_of_day: int = Field(12, description="Hour of day (0-23)", example=14)

class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Predicted trip duration in minutes")
    status: str = Field("success", description="Status of prediction request")

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    service: str = "bentoml-ride-duration"


# Initialize BentoML model runner for micro-batching
try:
    model_runner = bentoml.sklearn.get("ride_duration_model:latest").to_runner()
except Exception:
    model_runner = None


@bentoml.service(
    name="ride_duration_service",
    resources={"cpu": "1000m"},
)
class RideDurationService:
    runner = model_runner

    def __init__(self):
        if self.runner is not None:
            pass

    @bentoml.api(route="/predict")
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Async prediction endpoint supporting BentoML micro-batching.
        """
        features = [[request.distance_km, request.passengers, request.hour_of_day]]
        
        if self.runner is not None:
            # BentoML async runner execution
            result = await self.runner.predict.async_run(features)
        else:
            # Fallback direct sklearn evaluation
            model_ref = bentoml.sklearn.get("ride_duration_model:latest").load_model()
            result = model_ref.predict(features)
        
        prediction_val = float(np.ravel(result)[0])
        return PredictResponse(prediction=prediction_val, status="success")

    @bentoml.api(route="/healthz")
    def healthz(self) -> HealthCheckResponse:
        """
        Health check endpoint for liveness and readiness probes.
        """
        return HealthCheckResponse(status="healthy", service="bentoml-ride-duration")
