import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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


def test_predictor_missing_artifacts(tmp_path):
    missing_model = tmp_path / "model.pkl"
    missing_scaler = tmp_path / "scaler.pkl"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        OnlineShoppersPredictor(model_path=missing_model, scaler_path=missing_scaler)


def test_predictor_feature_names_fallback(tmp_path):
    # Train dummy model with feature_names_in_
    X = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [3.0, 4.0]})
    y = pd.Series([0, 1])
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(X, y)

    model_file = tmp_path / "model.pkl"
    scaler_file = tmp_path / "scaler.pkl"
    missing_names = tmp_path / "nonexistent.json"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    predictor = OnlineShoppersPredictor(
        model_path=model_file, scaler_path=scaler_file, feature_names_path=missing_names
    )
    assert predictor.feature_columns == ["feat_1", "feat_2"]


def test_predictor_missing_features_error(tmp_path):
    # Train model without feature_names_in_ (using numpy array)
    X = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [3.0, 4.0]})
    y = pd.Series([0, 1])
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(X.values, y)

    model_file = tmp_path / "model.pkl"
    scaler_file = tmp_path / "scaler.pkl"
    missing_names = tmp_path / "nonexistent.json"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    with pytest.raises(ValueError, match="Feature columns metadata missing"):
        OnlineShoppersPredictor(
            model_path=model_file, scaler_path=scaler_file, feature_names_path=missing_names
        )
