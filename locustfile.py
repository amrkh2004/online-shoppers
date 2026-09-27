"""
Locust load testing script (Deliverable 06)
Simulates concurrent traffic to BentoML & FastAPI /predict and /healthz endpoints.
"""

import random
from locust import HttpUser, task, between, events


class RideDurationUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def predict_endpoint(self):
        payload = {
            "distance_km": round(random.uniform(1.0, 30.0), 2),
            "passengers": random.randint(1, 4),
            "hour_of_day": random.randint(0, 23),
        }
        headers = {"Content-Type": "application/json"}

        with self.client.post("/predict", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    res_json = response.json()
                    if "prediction" in res_json or "prediction_duration" in res_json:
                        response.success()
                    else:
                        response.failure(f"Missing prediction in response: {response.text}")
                except Exception as e:
                    response.failure(f"JSON decode error: {e}")
            else:
                response.failure(f"HTTP Status {response.status_code}: {response.text}")

    @task(1)
    def health_endpoint(self):
        with self.client.get("/healthz", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                # Try fallback health route
                with self.client.get("/health", catch_response=True) as fb_res:
                    if fb_res.status_code == 200:
                        fb_res.success()
                    else:
                        response.failure(f"Health check failed with code {response.status_code}")
