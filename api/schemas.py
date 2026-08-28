"""Pydantic request/response schemas for the Feature Engineering API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class FeatureVectorResponse(BaseModel):
    personnel_id: int = Field(..., description="Personnel identifier the features belong to")
    biometric_available: bool = Field(
        ..., description="Whether biometric-derived features are usable for this personnel "
                          "(requires both consent and a valid biometric record)."
    )
    features: dict[str, Optional[float]] = Field(
        ..., description="Stable, ordered engineered feature vector. A value of null means "
                          "that feature could not be computed (e.g. no biometric consent)."
    )


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    detail: Optional[str] = None
