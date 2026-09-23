# 🛒 Online Shoppers Purchasing Intention — Production ML Service

[![CI/CD Pipeline](https://github.com/amrkh2004/online-shoppers/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/amrkh2004/online-shoppers/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)
[![Test Coverage](https://img.shields.io/badge/Coverage-99%25-brightgreen.svg)](tests/)

Production Machine Learning package and RESTful API for predicting online shoppers purchasing intention based on website session behavior.

---

## ⚡ Quickstart — 3 Commands to Prediction

Get from zero to running predictions in **3 simple commands**:

```bash
# 1. Clone the repository
git clone https://github.com/amrkh2004/online-shoppers.git && cd online-shoppers

# 2. Build & run the API service with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# 3. Make a real prediction request
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
           "Weekend": false
         }'
```

---

## 📊 Example API Response

```json
{
  "prediction": 1,
  "label": "Purchase",
  "probability": 0.8415,
  "threshold_used": 0.49
}
```

---

## 🌐 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck returning service status & version |
| `GET` | `/metadata` | Model metadata, feature list, and baseline performance |
| `POST` | `/predict` | Predict purchasing intention for a single shopper session |
| `POST` | `/predict/batch` | Predict purchasing intention for batch shopper sessions |

Interactive API Documentation (Swagger UI) is available at `http://localhost:8000/docs`.

---

## ⚙️ Local Development & Testing

```bash
# Install package in editable mode with dev dependencies
pip install -e .[dev]

# Train model and export Joblib & ONNX artifacts
python scripts/train_and_export.py

# Run test suite with coverage report
pytest
```

---

## 📈 Experiment Tracking: MLflow vs Weights & Biases (W&B)

Both **MLflow** and **Weights & Biases (W&B)** are integrated to track hyperparameters, model metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, MAE), and artifacts across experiments.

```bash
# Run MLflow 3-run experiment suite & transition best model to Staging
python scripts/run_mlflow_experiments.py

# Run Weights & Biases experiment suite
python scripts/run_wandb_experiments.py
```

### 🔬 Tool Comparison Summary

| Feature / Criteria | 🧪 MLflow | ⚡ Weights & Biases (W&B) |
| :--- | :--- | :--- |
| **Primary Focus** | Open-source end-to-end MLOps lifecycle & Model Registry | SaaS/Cloud real-time experiment tracking & collaboration |
| **Hosting Model** | Self-hosted (Local SQLite/S3 or Server) | Cloud-hosted dashboard with local offline fallback |
| **Model Registry & Staging** | ✅ Built-in Model Registry & Stage transitions (`Staging`, `Production`) | Requires W&B Artifacts / Model Registry |
| **Metrics Tracked** | Accuracy, Precision, Recall, F1, ROC-AUC, MAE | Accuracy, Precision, Recall, F1, ROC-AUC, MAE |
| **Offline Support** | ✅ Fully native local file/SQLite store | ✅ Offline mode supported via `WANDB_MODE=offline` |
| **Best Used For** | Standardized model governance, artifact registry, and local pipelines | Interactive dashboards, team collaboration, and hyperparameter tuning |

---

## 📁 Project Tree

```text
online-shoppers/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── feature_names.json
│   ├── final_random_forest.onnx
│   ├── final_random_forest.pkl
│   └── scaler.pkl
├── notebooks/
│   └── Online_Shoppers_Purchasing (3).ipynb
├── reports/
│   └── module-1.md
├── scripts/
│   └── train_and_export.py
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       ├── train.py
│       └── api/
│           ├── __init__.py
│           ├── main.py
│           └── schemas.py
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_export.py
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_train.py
├── .dockerignore
├── .gitignore
├── pyproject.toml
└── README.md
```
