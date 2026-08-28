"""Tests for the FastAPI feature-generation service."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_features_valid_personnel():
    response = client.post("/features/generate/3")
    assert response.status_code == 200
    body = response.json()
    assert body["personnel_id"] == 3
    assert "features" in body
    assert "workload_score" in body["features"]


def test_generate_features_response_structure():
    response = client.post("/features/generate/5")
    body = response.json()
    assert set(body.keys()) == {"personnel_id", "biometric_available", "features"}
    assert isinstance(body["features"], dict)
    assert isinstance(body["biometric_available"], bool)


def test_generate_features_invalid_personnel_id_returns_404():
    response = client.post("/features/generate/999999")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_generate_features_missing_biometric_consent_returns_null_biometric_features():
    # personnel_id=1 has wellness consent only (no biometric) per the dataset.
    response = client.post("/features/generate/1")
    body = response.json()
    assert body["biometric_available"] is False
    assert body["features"]["sleep_deficit"] is None


def test_generate_features_non_integer_personnel_id_returns_422():
    response = client.post("/features/generate/not-a-number")
    assert response.status_code == 422
