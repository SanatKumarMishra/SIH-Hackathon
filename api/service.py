"""
Service layer: thin bridge between the FastAPI routes and the
feature-engineering pipeline. Kept separate from `routes.py` so the
core logic is unit-testable without spinning up FastAPI/TestClient.
"""
from __future__ import annotations

import logging
from typing import Optional

from feature_engineering.load_data import BaseDataSource, ExcelDataSource
from feature_engineering.pipeline import PersonnelNotFoundError, get_feature_vector

logger = logging.getLogger(__name__)

# A single, reused data source instance (Excel today; swap for
# PostgresDataSource() once the live DB is available — no other code
# needs to change).
_DEFAULT_SOURCE: Optional[BaseDataSource] = None


def _get_source() -> BaseDataSource:
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE is None:
        _DEFAULT_SOURCE = ExcelDataSource()
    return _DEFAULT_SOURCE


def generate_features_for_personnel(personnel_id: int) -> dict:
    """
    Returns the feature-vector payload for one personnel_id, or raises
    `PersonnelNotFoundError` (mapped to HTTP 404 in routes.py).
    """
    return get_feature_vector(personnel_id, source=_get_source())


def reset_source_cache() -> None:
    """Used by tests to force a fresh data source (e.g. after swapping fixtures)."""
    global _DEFAULT_SOURCE
    _DEFAULT_SOURCE = None
