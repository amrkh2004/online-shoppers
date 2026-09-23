import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Tuned Random Forest Classifier"
    assert "baseline_metrics" in data


def test_predict_endpoint(client):
    payload = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 12,
        "ProductRelated_Duration": 310.0,
        "BounceRates": 0.0,
        "ExitRates": 0.01,
        "PageValues": 38.5,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data


def test_predict_invalid_input(client):
    payload = {
        "Administrative": -5,  # Invalid negative count
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable entity validation error


def test_predict_batch_endpoint(client):
    payload = {
        "inputs": [
            {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 12,
                "ProductRelated_Duration": 310.0,
                "BounceRates": 0.0,
                "ExitRates": 0.01,
                "PageValues": 38.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert len(data["predictions"]) == 1


def test_predict_when_predictor_none(client, monkeypatch):
    import prodml.api.main as main_mod

    monkeypatch.setattr(main_mod, "predictor", None)
    res_single = client.post("/predict", json={})
    assert res_single.status_code == 503

    res_batch = client.post("/predict/batch", json={"inputs": []})
    assert res_batch.status_code == 503


def test_predict_exception_handling(client, monkeypatch):
    import prodml.api.main as main_mod

    def mock_predict_dict(record):
        raise RuntimeError("Model prediction failed unexpectedly")

    monkeypatch.setattr(main_mod.predictor, "predict_dict", mock_predict_dict)

    payload = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 12,
        "ProductRelated_Duration": 310.0,
        "BounceRates": 0.0,
        "ExitRates": 0.01,
        "PageValues": 38.5,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "Prediction error" in response.json()["detail"]


def test_predict_batch_exception_handling(client, monkeypatch):
    import prodml.api.main as main_mod

    def mock_predict_dataframe(df):
        raise RuntimeError("Batch prediction failed unexpectedly")

    monkeypatch.setattr(main_mod.predictor, "predict_dataframe", mock_predict_dataframe)

    payload = {
        "inputs": [
            {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 12,
                "ProductRelated_Duration": 310.0,
                "BounceRates": 0.0,
                "ExitRates": 0.01,
                "PageValues": 38.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 500
    assert "Batch prediction error" in response.json()["detail"]


def test_lifespan_degraded_startup(monkeypatch):
    import asyncio

    import prodml.api.main as main_mod

    def mock_init_fail(*args, **kwargs):
        raise FileNotFoundError("Missing model files")

    monkeypatch.setattr(main_mod, "OnlineShoppersPredictor", mock_init_fail)

    async def run_lifespan():
        async with main_mod.lifespan(main_mod.app):
            assert main_mod.predictor is None

    asyncio.run(run_lifespan())
