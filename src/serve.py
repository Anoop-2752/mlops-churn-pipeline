"""
Stage 4: FastAPI Serving
---------------------------
Loads the registered "churn-classifier" model (Staging stage) from the
MLflow Model Registry at startup and exposes it via a REST API.

The loaded object is the FULL pipeline (preprocessing -> SMOTE -> classifier)
registered in train.py. SMOTE is a no-op at inference time (imblearn only
resamples during .fit()), so .predict()/.predict_proba() work directly on
raw feature input after the ColumnTransformer step.

Run:
    uvicorn src.serve:app --reload
Then visit:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""

from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = "churn-classifier"
MODEL_STAGE = "Staging"

# Global holder for the loaded model + metadata (populated at startup)
model_store = {}


# ---------------------------------------------------------------------------
# Request schema - mirrors the raw input columns (minus customerID, Churn)
# ---------------------------------------------------------------------------

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    model_version: str


# ---------------------------------------------------------------------------
# App setup with startup/shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load model once ---
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])

    if not versions:
        raise RuntimeError(
            f"No model found for '{MODEL_NAME}' in stage '{MODEL_STAGE}'. "
            f"Run register_model.py first."
        )

    version_info = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    model_store["pipeline"] = mlflow.sklearn.load_model(model_uri)
    model_store["version"] = version_info.version
    model_store["stage"] = MODEL_STAGE

    print(f"[startup] Loaded '{MODEL_NAME}' version {version_info.version} "
          f"(stage={MODEL_STAGE})")

    yield

    # --- Shutdown: nothing to clean up ---
    model_store.clear()


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in model_store}


@app.get("/model-info")
def model_info():
    if "pipeline" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": MODEL_NAME,
        "version": model_store["version"],
        "stage": model_store["stage"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if "pipeline" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pipeline = model_store["pipeline"]

    # Convert the single request into a one-row DataFrame matching the
    # column names expected by the ColumnTransformer.
    input_df = pd.DataFrame([customer.model_dump()])

    proba = pipeline.predict_proba(input_df)[0, 1]
    prediction = "Yes" if proba >= 0.5 else "No"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(float(proba), 4),
        model_version=str(model_store["version"]),
    )