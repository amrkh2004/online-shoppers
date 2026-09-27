from consumer import get_model, predict_on_event, store_result
from unittest.mock import MagicMock
import tempfile
from pathlib import Path
import json

def test_get_model_caching():
    m1 = get_model()
    m2 = get_model()
    assert m1 is m2

def test_predict_on_event():
    evt = {"event_id": "test_1", "distance_km": 8.0, "passengers": 2, "hour_of_day": 14}
    res = predict_on_event(evt)
    assert res["event_id"] == "test_1"
    assert "prediction" in res
    assert "latency_ms" in res
    assert isinstance(res["prediction"], float)

def test_store_result_file(tmp_path):
    res_data = {"event_id": "test_2", "prediction": 22.5, "latency_ms": 1.2}
    target_file = tmp_path / "consumer_predictions.log"
    store_result(res_data, storage_type="file", output_path=target_file)
    assert target_file.exists()
    line = json.loads(target_file.read_text())
    assert line["event_id"] == "test_2"
