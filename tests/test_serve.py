"""
Tests for Stage 4: FastAPI Serving

Design note: the app's `lifespan` function loads the model from the MLflow
registry on startup. We don't want tests to depend on a live MLflow server
(slow, flaky, requires Docker/network in CI). Instead, we directly inject a
fake pipeline into `model_store` and use TestClient WITHOUT the `with`
context manager - this skips lifespan entirely, so /predict and /health
work against our fake model without ever touching MLflow.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serve import app, model_store

client = TestClient(app)  # no `with` -> lifespan does NOT run


VALID_PAYLOAD = {
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


@pytest.fixture(autouse=True)
def fake_model():
    """Inject a fake pipeline into model_store before each test, clean up after."""
    fake_pipeline = MagicMock()
    # predict_proba returns [[P(no_churn), P(churn)]] for one row
    fake_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])

    model_store["pipeline"] = fake_pipeline
    model_store["version"] = "1"
    model_store["stage"] = "Staging"

    yield

    model_store.clear()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_when_model_not_loaded():
    model_store.clear()  # simulate model not loaded
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "churn-classifier"
    assert data["version"] == "1"
    assert data["stage"] == "Staging"


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert data["churn_prediction"] == "Yes"   # 0.8 >= 0.5 threshold
    assert data["churn_probability"] == 0.8
    assert data["model_version"] == "1"


def test_predict_rejects_invalid_category():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["Contract"] = "Annually"  # not a valid Literal value

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_predict_rejects_negative_tenure():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["tenure"] = -5  # violates Field(ge=0, le=72)

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["MonthlyCharges"]

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422