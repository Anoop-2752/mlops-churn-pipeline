# Churn MLOps Pipeline

An end-to-end MLOps pipeline for predicting customer churn — covering data validation, experiment tracking, model registry, containerized serving, CI/CD, drift monitoring, and orchestrated retraining.

This project was built as a hands-on reference implementation of the full ML lifecycle, using the IBM Telco Customer Churn dataset as the use case. The focus is on the **infrastructure and operational practices** around the model, not just the model itself.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Raw CSV data │ --> │ Data          │ --> │ Preprocessing    │ --> │ Training +        │
│ (Kaggle)     │     │ Validation    │     │ (ColumnTransform │     │ Experiment        │
│              │     │ (Pandera)     │     │  + SMOTE)        │     │ Tracking (MLflow) │
└─────────────┘     └──────────────┘     └─────────────────┘     └─────────┬─────────┘
                                                                              │
                                                                              v
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Drift         │ <-- │ Orchestration │     │ Containerized    │ <-- │ Model Registry     │
│ Detection     │     │ (Prefect)     │     │ Serving          │     │ (MLflow, staged)   │
│ (Evidently)   │     │               │     │ (FastAPI+Docker) │     │                    │
└──────┬───────┘     └──────┬────────┘     └─────────────────┘     └──────────────────┘
       │                    │
       └────── triggers retraining if drift detected ──────┘

All stages are tested (pytest) and run automatically on push via GitHub Actions CI.
```

**Runtime stack (docker-compose):**
```
┌─────────────────────────┐       ┌─────────────────────────┐
│   mlflow (container)     │ <---- │   api (container)        │
│   - Tracking server       │       │   - FastAPI               │
│   - Model registry        │       │   - Loads model from      │
│   - SQLite backend         │       │     mlflow:5000           │
│   - Artifact proxy store   │       │   - Serves /predict        │
│   Port: 5000               │       │   Port: 8002 -> 8000        │
└─────────────────────────┘       └─────────────────────────┘
```

---

## Tech Stack

| Stage | Tools |
|---|---|
| Data validation | Pandera |
| Preprocessing | scikit-learn (ColumnTransformer, OneHotEncoder, StandardScaler) |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Experiment tracking & registry | MLflow 3.13.0 |
| Models | Logistic Regression, Random Forest, XGBoost |
| Serving | FastAPI + Pydantic |
| Containerization | Docker, docker-compose |
| Testing | pytest, FastAPI TestClient |
| CI/CD | GitHub Actions |
| Drift monitoring | Evidently AI |
| Orchestration | Prefect |

---

## Project Structure

```
churn-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions: tests + Docker builds
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis (exploration only)
├── reports/
│   ├── drift_report.html        # Evidently report - simulated drifted data
│   └── no_drift_report.html     # Evidently report - sanity check (no drift)
├── src/
│   ├── data_validation.py       # Stage 1: Pandera schema + cleaning
│   ├── preprocessing.py         # Stage 2: ColumnTransformer + feature prep
│   ├── train.py                 # Stage 3: Train models + MLflow logging
│   ├── register_model.py        # Stage 3b: Register best run in MLflow registry
│   ├── show_metrics.py          # Utility: print all run metrics
│   ├── check_registry.py        # Utility: check registered model stage
│   ├── serve.py                 # Stage 4: FastAPI serving
│   ├── monitoring.py            # Stage 8: Evidently drift reports
│   └── pipeline.py              # Stage 9: Prefect orchestration flow
├── tests/
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   └── test_serve.py
├── conftest.py                  # Ensures src/ is importable in tests
├── Dockerfile                   # FastAPI service image
├── Dockerfile.mlflow            # MLflow tracking server image
├── docker-compose.yml           # Orchestrates mlflow + api containers
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/Anoop-2752/churn-mlops-pipeline.git
cd churn-mlops-pipeline
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Run the pipeline locally (without Docker)

```bash
# Stage 1-2: validate data and inspect the preprocessing output
python src/data_validation.py
python src/preprocessing.py

# Stage 3: train models and log to local MLflow (./mlruns)
python src/train.py

# View experiments
mlflow ui   # open http://localhost:5000

# Stage 3b: register the best run
python src/register_model.py
python src/check_registry.py

# Stage 4: serve the model
uvicorn src.serve:app --reload   # open http://127.0.0.1:8000/docs
```

### 3. Run the full stack with Docker

```bash
docker-compose up -d --build
```

- MLflow UI: `http://localhost:5000`
- API docs: `http://localhost:8002/docs`

If running training/registration against the Dockerized MLflow server (rather than local file storage), point the client at it first:

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000   # Windows
python src/train.py
python src/register_model.py
```

### 4. Run tests

```bash
pytest -v
```

### 5. Drift monitoring

```bash
python src/monitoring.py
# open reports/drift_report.html and reports/no_drift_report.html
```

### 6. Orchestrated pipeline (Prefect)

```bash
python src/pipeline.py              # simulates drift -> triggers retraining
python src/pipeline.py --no-drift   # no drift -> skips retraining

# Optional: view runs in Prefect UI
prefect server start                # open http://127.0.0.1:4200
```

---

## Model Selection

Three models were trained with SMOTE applied to the training fold only (to avoid data leakage into the test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.737 | 0.503 | **0.797** | **0.616** | **0.839** |
| XGBoost | 0.776 | 0.579 | 0.570 | 0.574 | 0.816 |
| Random Forest | 0.769 | 0.565 | 0.559 | 0.562 | 0.820 |

**Logistic Regression was selected**, despite having the lowest raw accuracy. For churn prediction, **recall** is the more business-relevant metric — missing an actual churner (false negative) typically costs more than an unnecessary retention offer (false positive). Logistic Regression catches ~80% of churners vs. ~57% for the tree-based models, and also has the best ROC-AUC and F1. This is a deliberate illustration of the "accuracy paradox" on imbalanced datasets: the model with the *highest accuracy* was not the *best model* for the business objective.

The full pipeline (preprocessing → SMOTE → classifier) is registered as a single MLflow artifact, so the exact same object handles raw-feature input at both training and inference time, with no train/serve skew.

---

## API Reference

### `GET /health`
Returns service and model-load status.
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model-info`
Returns the currently loaded model's registry metadata.
```json
{ "model_name": "churn-classifier", "version": "1", "stage": "Staging" }
```

### `POST /predict`
Accepts raw customer attributes (see `CustomerData` schema in `src/serve.py` for full field list and allowed values) and returns a churn prediction.

**Example request:**
```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}
```

**Example response:**
```json
{ "churn_prediction": "Yes", "churn_probability": 0.8126, "model_version": "1" }
```

**Sanity check:** the same request with `Contract: "Two year"` and `tenure: 60` returns `churn_probability: 0.0448` — confirming the model has learned that long-tenure, long-contract customers are far less likely to churn, consistent with the EDA findings.

Invalid inputs (e.g., an unrecognized `Contract` value, negative `tenure`) return `422 Unprocessable Entity` via Pydantic validation, before ever reaching the model.

---

## Testing Strategy

14 tests across three layers:

- **`test_data_validation.py`** — confirms the Pandera schema accepts clean data and rejects invalid categories/out-of-range values.
- **`test_preprocessing.py`** — confirms feature/target separation, the expected 40-column output shape, and that unseen categories at inference time don't crash the pipeline (`handle_unknown="ignore"`).
- **`test_serve.py`** — tests the FastAPI endpoints using a mocked model (injected directly into `model_store`, bypassing the MLflow-dependent `lifespan` startup), so tests run fast and don't require a live MLflow server. Covers valid predictions, missing fields, invalid categories, and out-of-range values.

CI runs the full suite on every push/PR via GitHub Actions, then builds both Docker images if tests pass.

---

## Monitoring & Drift Detection

`src/monitoring.py` generates two Evidently AI reports comparing "current" data against the training (reference) distribution:

- **`no_drift_report.html`** — a random sample from the same distribution. Used as a sanity check; should show no significant drift.
- **`drift_report.html`** — a sample with deliberately shifted `MonthlyCharges` (+30%, simulating a price increase), `tenure` (shifted toward 0, simulating a new-customer acquisition wave), and `InternetService` (forced to "Fiber optic", simulating a sales push). Correctly flags all three as drifted while leaving untouched columns unflagged.

---

## Orchestration

`src/pipeline.py` defines a Prefect flow that ties the system together:

1. Load reference feature distribution (training data)
2. Simulate incoming production data (optionally with drift)
3. Run a Kolmogorov-Smirnov test on key numeric features against the reference
4. **If drift is detected** → automatically retrain all three models and re-register the best one
5. **If no drift** → skip retraining

This models the realistic decision point in production ML systems: drift detection isn't useful on its own — it needs to gate an action (retrain, alert, or no-op).

---

## Challenges & Lessons Learned

This section documents real issues hit during development — debugging infrastructure is most of what "MLOps" actually involves in practice.

1. **MLflow client/server version mismatch.** The local MLflow client was v3.13.0, but the containerized MLflow server was initially pinned to v2.14.1. Logging a model via `mlflow.sklearn.log_model()` failed with a `404` on `/api/2.0/mlflow/logged-models` — an endpoint that didn't exist in the older server version. **Fix:** pin the server image to the exact same MLflow version as the local client.

2. **Artifact storage path mismatch in Docker.** The MLflow server was initially configured with `--default-artifact-root /mlflow/artifacts`, a path inside the *server's* container filesystem. When the client (running on the host) tried to log artifacts, it attempted to write directly to that non-existent local path and the run failed with status `FAILED`. **Fix:** use `--artifacts-destination` (without `--default-artifact-root`) to enable MLflow's artifact proxy — the client sends artifacts over HTTP to the server, which stores them internally.

3. **DNS rebinding protection blocking inter-container requests.** Once the MLflow server was reachable, the FastAPI container's requests to `http://mlflow:5000` were rejected with `403 Invalid Host header - possible DNS rebinding attack detected`. MLflow's server validates the `Host` header against an allowlist (default: `localhost`/`127.0.0.1` only). **Fix:** add `--allowed-hosts "mlflow:*,localhost:*,127.0.0.1:*"` to the server startup command.

4. **Accuracy vs. recall trade-off.** XGBoost had the highest raw accuracy, but Logistic Regression had substantially higher recall and ROC-AUC. For churn prediction, recall is the more defensible metric — selecting "the highest accuracy model" without considering the business cost of false negatives would have been the wrong call.

5. **Pandera import deprecation.** `import pandera as pa` triggers a `FutureWarning` in recent Pandera versions in favor of `import pandera.pandas as pa` — fixed proactively to avoid a future breaking change.

6. **Test isolation from external dependencies.** The FastAPI app's `lifespan` loads the model from the MLflow registry on startup — but tests shouldn't depend on a live MLflow server. **Fix:** `TestClient(app)` without a `with` block skips `lifespan` entirely, allowing a mocked model to be injected directly into `model_store` for fast, isolated API tests.

---

## Possible Future Improvements

- Replace MLflow's deprecated stage-based registry (`Staging`/`Production`) with the newer alias-based system (`@champion`/`@challenger`)
- Run Prefect flows on a schedule (deployment) rather than manually
- Use a managed/cloud-hosted MLflow tracking server and artifact store (e.g., S3-backed) instead of local Docker volumes
- Log real `/predict` request payloads to build a genuine "current data" dataset for drift monitoring, instead of simulated samples
- Add a Postgres backend for MLflow instead of SQLite, for concurrent-write support in a multi-user setting