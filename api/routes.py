"""
FastAPI routes for the Feature Engineering microservice.

    POST /features/generate/{personnel_id}   -> FeatureVectorResponse
    GET  /health                             -> HealthResponse

Run with:  uvicorn api.routes:app --reload --port 8001
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from feature_engineering.pipeline import PersonnelNotFoundError

from .schemas import ErrorResponse, FeatureVectorResponse, HealthResponse
from .service import generate_features_for_personnel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SIH26186 Feature Engineering Service",
    description="Consent-aware feature generation for the welfare-risk model.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/features/generate/{personnel_id}",
    response_model=FeatureVectorResponse,
    responses={404: {"model": ErrorResponse}},
)
def generate_features(personnel_id: int) -> FeatureVectorResponse:
    """
    Generates the current, consent-aware engineered feature vector for one
    personnel record. Intended to be called by the Risk Engine service
    just before running inference.
    """
    try:
        payload = generate_features_for_personnel(personnel_id)
    except PersonnelNotFoundError as exc:
        logger.info("Feature generation requested for unknown personnel_id=%s", personnel_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error generating features for personnel_id=%s", personnel_id)
        raise HTTPException(status_code=500, detail="Internal error generating features")

    return FeatureVectorResponse(**payload)
