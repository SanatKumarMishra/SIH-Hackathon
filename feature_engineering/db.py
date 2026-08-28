"""
PostgreSQL connection layer.

Credentials are NEVER hard-coded. They are read from environment
variables (optionally loaded from a local `.env` file via
`python-dotenv`, if installed and present):

    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

This module intentionally does not redefine the existing application
schema — it only provides a thin, reusable SQLAlchemy engine/session
so `load_data.PostgresDataSource` can query the tables that already
exist (personnel, hr_records, wellness_assessments, biometric_data,
consent_records).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import PostgresSettings

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()  # no-op if there is no .env file
except ImportError:  # pragma: no cover - optional dependency
    pass


@lru_cache(maxsize=1)
def get_engine():
    """
    Returns a cached SQLAlchemy engine built from environment variables.

    Import of sqlalchemy/psycopg is deferred to this function so the
    rest of the package (and the Excel-based demo path) works even in
    environments where a PostgreSQL driver is not installed.
    """
    from sqlalchemy import create_engine

    settings = PostgresSettings()
    logger.info(
        "Connecting to PostgreSQL at %s:%s/%s as %s",
        settings.host, settings.port, settings.database, settings.user,
    )
    return create_engine(settings.sqlalchemy_url, pool_pre_ping=True)


def healthcheck() -> bool:
    """Simple connectivity check used by the API's /health endpoint."""
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("PostgreSQL healthcheck failed: %s", exc)
        return False
