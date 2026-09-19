"""
Model serialization (pickle/joblib and ONNX) and parity testing for prodml.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union
import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import (
    DEFAULT_THRESHOLD,
    FEATURE_NAMES_PATH,
    MODEL_PATH_ONNX,
    MODEL_PATH_PKL,
    SCALER_PATH_PKL,
)
from prodml.logging_conf import logger


def export_model_artifacts(
    model: any,
    scaler: any,
    feature_names: List[str],
    pkl_model_path: Path = MODEL_PATH_PKL,
    scaler_path: Path = SCALER_PATH_PKL,
    onnx_model_path: Path = MODEL_PATH_ONNX,
    feature_names_path: Path = FEATURE_NAMES_PATH,
) -> Dict[str, str]:
    """
    Exports trained model and scaler to joblib/pickle format and ONNX format.
    """
    logger.info("Exporting model artifacts...")

    # Ensure output directories exist
    pkl_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save Joblib artifacts
    joblib.dump(model, pkl_model_path)
    joblib.dump(scaler, scaler_path)

    # Save feature names JSON
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Convert to ONNX format
    initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info("Successfully exported joblib and ONNX model artifacts.")

    return {
        "joblib_model": str(pkl_model_path),
        "joblib_scaler": str(scaler_path),
        "onnx_model": str(onnx_model_path),
        "feature_names": str(feature_names_path),
    }


def verify_onnx_parity(
    model_pkl_path: Path = MODEL_PATH_PKL,
    model_onnx_path: Path = MODEL_PATH_ONNX,
    sample_data: pd.DataFrame = None,
    num_samples: int = 100,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Verifies prediction parity between Joblib/Pickle model and ONNX model runtime.
    Also measures prediction latency for both formats.
    """
    logger.info("Starting ONNX vs Joblib Parity Test and Latency Benchmark...")

    if not model_pkl_path.exists() or not model_onnx_path.exists():
        raise FileNotFoundError("Model artifacts must be exported before running parity test.")

    # Load models
    joblib_model = joblib.load(model_pkl_path)
    sess = rt.InferenceSession(str(model_onnx_path))

    # Generate synthetic or use passed sample data
    num_features = len(joblib_model.feature_names_in_) if hasattr(joblib_model, "feature_names_in_") else 70
    if sample_data is None:
        data_arr = np.random.randn(num_samples, num_features).astype(np.float32)
    else:
        data_arr = sample_data.values.astype(np.float32)

    # Benchmark Joblib prediction
    t0 = time.perf_counter()
    joblib_probs = joblib_model.predict_proba(data_arr)[:, 1]
    joblib_latency_ms = (time.perf_counter() - t0) * 1000 / num_samples

    # Benchmark ONNX prediction
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    onnx_res = sess.run(None, {input_name: data_arr})
    # ONNX predict_proba output is list of dicts or array of probabilities
    if len(onnx_res) > 1 and isinstance(onnx_res[1], list):
        onnx_probs = np.array([prob_dict[1] for prob_dict in onnx_res[1]], dtype=np.float32)
    else:
        onnx_probs = onnx_res[1][:, 1] if len(onnx_res) > 1 else onnx_res[0][:, 1]
    onnx_latency_ms = (time.perf_counter() - t0) * 1000 / num_samples

    # Calculate parity delta
    max_prob_diff = float(np.max(np.abs(joblib_probs - onnx_probs)))

    joblib_preds = (joblib_probs >= threshold).astype(int)
    onnx_preds = (onnx_probs >= threshold).astype(int)
    prediction_mismatches = int(np.sum(joblib_preds != onnx_preds))

    metrics = {
        "max_prob_difference": max_prob_diff,
        "mismatch_count": prediction_mismatches,
        "joblib_latency_ms_per_sample": round(joblib_latency_ms, 4),
        "onnx_latency_ms_per_sample": round(onnx_latency_ms, 4),
        "speedup_factor": round(joblib_latency_ms / max(onnx_latency_ms, 1e-6), 2),
    }

    logger.info("ONNX Parity Test Completed", extra=metrics)
    return metrics
