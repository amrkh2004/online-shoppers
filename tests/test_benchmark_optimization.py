import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from prodml.benchmark import ModelEvaluator, run_full_benchmark_journey
from prodml.optimization import (
    quantize_onnx_model,
)


def test_quantize_onnx_model_exists():
    from prodml.config import MODEL_PATH_ONNX

    if MODEL_PATH_ONNX.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "quantized.onnx"
            res = quantize_onnx_model(
                input_onnx_path=MODEL_PATH_ONNX,
                output_quantized_path=out_path,
            )
            assert res.exists()
            assert res.stat().st_size > 0


def test_model_evaluator_sklearn():
    X = np.random.randn(20, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    import joblib

    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = Path(tmp_dir) / "model.pkl"
        joblib.dump(model, pkl_path)

        res = ModelEvaluator.evaluate_sklearn(pkl_path, X, y, iterations=10)
        assert "p50_latency_ms" in res
        assert "size_mb" in res
        assert "peak_ram_mb" in res
        assert res["accuracy"] >= 0.0


def test_benchmark_journey_generation():
    df = run_full_benchmark_journey(iterations=20)
    assert not df.empty
    assert "size_mb" in df.columns
    assert "p50_latency_ms" in df.columns
    assert "speedup_factor" in df.columns
