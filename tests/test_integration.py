"""
Full Integration Test suite for FastAPI REST endpoints, Pydantic 422 validations, and Response Schema Contract.
"""

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_contract(client):
    """Verifies /health response schema and status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert data["status"] == "ok"
    assert isinstance(data["version"], str)


def test_metadata_endpoint_contract(client):
    """Verifies /metadata response schema contract."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "num_features" in data
    assert "classification_threshold" in data
    assert "baseline_metrics" in data
    assert isinstance(data["num_features"], int)
    assert isinstance(data["classification_threshold"], float)


def test_predict_success_and_schema_contract(client):
    """Verifies /predict success case and exact schema contract types."""
    valid_payload = {
        "Administrative": 2,
        "Administrative_Duration": 50.0,
        "Informational": 1,
        "Informational_Duration": 10.0,
        "ProductRelated": 15,
        "ProductRelated_Duration": 350.0,
        "BounceRates": 0.01,
        "ExitRates": 0.02,
        "PageValues": 25.4,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    # Schema Contract verification
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert "threshold_used" in data

    assert isinstance(data["prediction"], int)
    assert data["prediction"] in (0, 1)
    assert isinstance(data["label"], str)
    assert data["label"] in ("Purchase", "No Purchase")
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["threshold_used"], float)


@pytest.mark.parametrize(
    "invalid_payload,expected_loc_field",
    [
        ({"Administrative": -5}, "Administrative"),  # Negative integer
        ({"Informational": -1}, "Informational"),  # Negative integer
        ({"ProductRelated": -10}, "ProductRelated"),  # Negative integer
        ({"BounceRates": 1.5}, "BounceRates"),  # Out of range > 1.0
        ({"ExitRates": -0.05}, "ExitRates"),  # Out of range < 0.0
        ({"SpecialDay": 1.2}, "SpecialDay"),  # Out of range > 1.0
        ({"Month": "InvalidMonth"}, "Month"),  # Invalid enum value
        ({"VisitorType": "Alien_Visitor"}, "VisitorType"),  # Invalid enum value
    ],
)
def test_predict_422_validation_errors(client, invalid_payload, expected_loc_field):
    """Verifies 422 Unprocessable Entity for invalid input payloads."""
    base_valid = {
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
    payload = {**base_valid, **invalid_payload}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(expected_loc_field in err["loc"] for err in errors)


def test_predict_invalid_data_types_422(client):
    """Verifies 422 when invalid field data types are provided."""
    invalid_type_payload = {"Administrative": "invalid_non_int_string"}
    response = client.post("/predict", json=invalid_type_payload)
    assert response.status_code == 422


def test_predict_batch_contract(client):
    """Verifies /predict/batch endpoint contract and response schema."""
    batch_payload = {
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
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total_count" in data
    assert data["total_count"] == 1
    assert len(data["predictions"]) == 1
