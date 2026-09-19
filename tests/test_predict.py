import pytest
from prodml.predict import OnlineShoppersPredictor


@pytest.fixture
def sample_input():
    return {
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


def test_predictor_single_prediction(sample_input):
    predictor = OnlineShoppersPredictor()
    result = predictor.predict_dict(sample_input)

    assert "prediction" in result
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Purchase", "No Purchase")
    assert 0.0 <= result["probability"] <= 1.0
