import asyncio
import pytest
from bentoml_service.service import PredictRequest, PredictResponse, HealthCheckResponse, RideDurationService
from unittest.mock import MagicMock, patch

def test_pydantic_schemas():
    req = PredictRequest(distance_km=10.5, passengers=2, hour_of_day=15)
    assert req.distance_km == 10.5
    assert req.passengers == 2
    assert req.hour_of_day == 15

    res = PredictResponse(prediction=25.4, status="success")
    assert res.prediction == 25.4
    assert res.status == "success"

def test_healthz_endpoint():
    service = RideDurationService()
    health = service.healthz()
    assert health.status == "healthy"
    assert health.service == "bentoml-ride-duration"

def test_predict_endpoint():
    service = RideDurationService()
    with patch("bentoml.sklearn.get") as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = [18.2]
        mock_get.return_value.load_model.return_value = mock_model
        
        req = PredictRequest(distance_km=5.0, passengers=1, hour_of_day=10)
        res = asyncio.run(service.predict(req))
        assert res.status == "success"
        assert isinstance(res.prediction, float)
