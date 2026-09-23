from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from prodml.config import MODEL_PATH_PKL
from prodml.export import export_model_artifacts, verify_onnx_parity


def test_onnx_parity():
    joblib_model = joblib.load(MODEL_PATH_PKL)
    feature_names = list(joblib_model.feature_names_in_)
    sample_df = pd.DataFrame(
        np.zeros((10, len(feature_names)), dtype=np.float32), columns=feature_names
    )
    parity_results = verify_onnx_parity(sample_data=sample_df, num_samples=10)
    assert "max_prob_difference" in parity_results
    assert parity_results["max_prob_difference"] < 0.05
    assert parity_results["mismatch_count"] == 0


def test_verify_onnx_parity_missing_files(tmp_path):
    missing_pkl = tmp_path / "nonexistent.pkl"
    missing_onnx = tmp_path / "nonexistent.onnx"
    with pytest.raises(FileNotFoundError, match="Model artifacts must be exported"):
        verify_onnx_parity(model_pkl_path=missing_pkl, model_onnx_path=missing_onnx)


def test_export_model_artifacts(tmp_path):
    X = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0], "feat_2": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_scaled, y)

    pkl_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    onnx_path = tmp_path / "model.onnx"
    names_path = tmp_path / "features.json"

    paths = export_model_artifacts(
        model=model,
        scaler=scaler,
        feature_names=["feat_1", "feat_2"],
        pkl_model_path=pkl_path,
        scaler_path=scaler_path,
        onnx_model_path=onnx_path,
        feature_names_path=names_path,
    )

    assert Path(paths["joblib_model"]).exists()
    assert Path(paths["joblib_scaler"]).exists()
    assert Path(paths["onnx_model"]).exists()
    assert Path(paths["feature_names"]).exists()
