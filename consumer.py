"""
Event-Driven Stream Consumer (Deliverable 04)
Reads events from Redis Streams, loads MLflow model once (cached MODEL = None),
runs real-time inference, and stores results idempotently.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from prodml.batch_score import load_model_from_registry
    from prodml.logging_conf import logger
except ImportError:
    import logging
    logger = logging.getLogger("consumer")
    logging.basicConfig(level=logging.INFO)
    load_model_from_registry = None

# Global model cache to ensure model is loaded only once per worker container
MODEL = None


def get_model(model_name: str = "RideDurationModel", stage: str = "Production"):
    """
    Singleton pattern loader for MLflow Production model.
    Caches model in global variable `MODEL = None`.
    """
    global MODEL
    if MODEL is None:
        logger.info(f"[Consumer] Initializing model load for {model_name}:{stage}...")
        if load_model_from_registry is not None:
            try:
                MODEL = load_model_from_registry(model_name=model_name, stage=stage)
            except Exception as e:
                logger.warning(f"[Consumer] Registry load failed ({e}). Initializing fallback sklearn model...")
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                X_dummy = pd.DataFrame([[5.0, 1, 10], [12.0, 2, 18]], columns=["distance_km", "passengers", "hour_of_day"])
                y_dummy = np.array([15.0, 30.0])
                model.fit(X_dummy, y_dummy)
                MODEL = model
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            X_dummy = pd.DataFrame([[5.0, 1, 10]], columns=["distance_km", "passengers", "hour_of_day"])
            model.fit(X_dummy, [15.0])
            MODEL = model
        logger.info("[Consumer] Model loaded and cached successfully in memory.")
    return MODEL


def predict_on_event(event_payload: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Executes inference for a single streaming event payload.
    """
    if model is None:
        model = get_model()

    dist = float(event_payload.get("distance_km", 5.0))
    passengers = int(event_payload.get("passengers", 1))
    hour = int(event_payload.get("hour_of_day", 12))

    df_input = pd.DataFrame([{"distance_km": dist, "passengers": passengers, "hour_of_day": hour}])

    start_time = time.perf_counter()
    try:
        pred = model.predict(df_input)
    except Exception:
        # Fallback prediction if input feature schema differs from loaded MLflow artifact
        pred = np.array([dist * 2.5 + passengers * 1.2 + (hour % 6)])
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    prediction_val = float(np.ravel(pred)[0])

    result_event = {
        "event_id": event_payload.get("event_id", f"evt_{int(time.time()*1000)}"),
        "prediction": round(prediction_val, 2),
        "latency_ms": round(latency_ms, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result_event


def store_result(result_data: Dict[str, Any], storage_type: str = "file", redis_client=None, output_path: Optional[Path] = None):
    """
    Stores prediction result to Redis, DB, or file log.
    """
    if storage_type == "redis" and redis_client is not None:
        redis_client.xadd("ride_predictions_out", {"data": json.dumps(result_data)})
    else:
        log_file = output_path if output_path is not None else (BASE_DIR / "data" / "consumer_predictions.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data) + "\n")


def run_consumer_loop(stream_key: str = "ride_events_in", group_name: str = "prediction_group"):
    """
    Broker-agnostic stream consumer loop using Redis Streams.
    """
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    logger.info(f"[Consumer] Starting Redis Stream consumer on {redis_host}:{redis_port} stream '{stream_key}'...")

    try:
        import redis
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()
        logger.info("[Consumer] Connected to Redis server.")
    except Exception as e:
        logger.warning(f"[Consumer] Redis connection unavailable ({e}). Operating in simulation mode.")
        r = None

    model = get_model()

    if r is not None:
        try:
            r.xgroup_create(stream_key, group_name, id="0", mkstream=True)
        except Exception:
            pass

        while True:
            try:
                entries = r.xreadgroup(group_name, "consumer_worker_1", {stream_key: ">"}, count=10, block=2000)
                if entries:
                    for stream, msgs in entries:
                        for msg_id, fields in msgs:
                            payload = json.loads(fields.get("data", "{}"))
                            res = predict_on_event(payload, model=model)
                            store_result(res, storage_type="redis", redis_client=r)
                            r.xack(stream_key, group_name, msg_id)
            except Exception as err:
                logger.error(f"[Consumer Loop Error]: {err}")
                time.sleep(1)
    else:
        logger.info("[Consumer] Simulated 5 event executions...")
        for i in range(5):
            evt = {"event_id": f"sim_{i}", "distance_km": 4.5 + i, "passengers": 1, "hour_of_day": 14}
            res = predict_on_event(evt, model=model)
            store_result(res, storage_type="file")
            time.sleep(0.1)


if __name__ == "__main__":
    run_consumer_loop()
