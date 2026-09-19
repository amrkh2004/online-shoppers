# Module 1 Report — Production Machine Learning Packaging & Service

## 1. Baseline Model Metrics & Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 0.8517 | 0.5203 | 0.6702 | 0.5858 | 0.7778 | Baseline |
| **Logistic Regression (Baseline)** | 0.8595 | 0.5354 | 0.7723 | 0.6324 | 0.9081 | Baseline |
| **XGBoost (Baseline)** | 0.8935 | 0.6597 | 0.6597 | 0.6597 | 0.9263 | Baseline |
| **Random Forest (Baseline)** | 0.8923 | 0.6393 | 0.7147 | 0.6749 | 0.9193 | Baseline |
| **Tuned Random Forest (Final)** | **0.8943** | **0.6449** | **0.7225** | **0.6815** | **0.9217** | **Production** |

*Note: The final production model utilizes a custom probability threshold of `0.49` optimized for F1-Score.*

---

## 2. Model Serialization & ONNX Parity Benchmark

| Format | File Size | Latency per Sample (ms) | Max Probability Delta | Prediction Parity |
| :--- | :---: | :---: | :---: | :---: |
| **Joblib / Pickle (`.pkl`)** | 53.8 MB | ~0.62 ms | 0.0000 | 100% Match |
| **ONNX (`.onnx`)** | 26.7 MB | ~0.14 ms | < 0.0001 | 100% Match |

### Key Takeaways:
- **ONNX Size Reduction**: ONNX format achieves a **50.3% reduction** in model artifact file size compared to Joblib pickle.
- **Latency Speedup**: ONNX runtime provides a **~4.4x speedup** in prediction latency per sample.
- **Parity Test**: Prediction output matches with zero classification mismatches and probability differences < `1e-4`.

---

## 3. Docker Container Image Comparison

| Container Build Strategy | Image Size | Non-Root User | Healthcheck |
| :--- | :---: | :---: | :---: |
| **Single-Stage Build (`python:3.11`)** | ~1.25 GB | ❌ No | ❌ No |
| **Multi-Stage Build (`python:3.11-slim`)** | ~340 MB | ✅ Yes (`appuser`) | ✅ `/health` |

---

## 4. MLOps Maturity Self-Assessment

### Current Level: **Level 1 — Automated ML Service Packaging & Testing**

**Self-Assessment Summary**:
The project achieves Level 1 maturity by standardizing the codebase into a Python package (`prodml`), maintaining 89% automated test coverage, eliminating unformatted prints via structured JSON logs, providing FastAPI microservice endpoints, and serving containerized builds via Multi-stage Docker. 

**What is missing to reach Level 2 (CI/CD Automated Pipelines & Continuous Training)**:
To transition to Level 2 maturity, the repository requires automated GitHub Actions CI/CD workflows for pre-commit linting, unit testing, and automated Docker image deployment upon Pull Request merge. Additionally, an automated continuous training (CT) pipeline triggered by new data or performance drift is needed to update model weights without manual intervention.
