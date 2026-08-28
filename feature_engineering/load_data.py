"""
Data loading layer.

Two backends are supported behind the same interface:

* `ExcelDataSource`  — reads the SIH26186 10-table workbook (current
  prototype source of truth).
* `PostgresDataSource` — reads the same logical tables from the real
  operational PostgreSQL database (see `db.py`). This lets the
  feature-engineering pipeline swap backends without any change to
  `preprocessing.py` / `features.py` / `pipeline.py`.

Both backends expose `.load_table(name)` and `.load_all()`, and both
are wrapped by `forbid_downstream_tables`, a hard guard that raises if
anyone tries to load `risk_predictions`, `risk_factors`, or
`recommendations` for feature-building purposes.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DataLeakageError(RuntimeError):
    """Raised when code attempts to read a downstream table as a feature input."""


def forbid_downstream_tables(table_name: str) -> None:
    """Hard guard against accidentally using model-output tables as features."""
    if table_name in config.DOWNSTREAM_TABLES:
        raise DataLeakageError(
            f"Refusing to load '{table_name}' as a feature input — it is a "
            f"downstream model OUTPUT table (risk_predictions / risk_factors / "
            f"recommendations are never allowed as predictive features)."
        )


class BaseDataSource(ABC):
    """Common interface for any data backend the feature pipeline can use."""

    @abstractmethod
    def load_table(self, table_name: str) -> pd.DataFrame:
        ...

    def load_all(self) -> Dict[str, pd.DataFrame]:
        return {name: self.load_table(name) for name in config.UPSTREAM_TABLES}


class ExcelDataSource(BaseDataSource):
    """Loads tables from the SIH26186 Excel workbook (one sheet per table)."""

    def __init__(self, xlsx_path: Path | str = config.DEFAULT_XLSX_PATH):
        self.xlsx_path = Path(xlsx_path)
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"Excel workbook not found at {self.xlsx_path}")
        self._xl = pd.ExcelFile(self.xlsx_path)

    def load_table(self, table_name: str) -> pd.DataFrame:
        forbid_downstream_tables(table_name)
        if table_name not in self._xl.sheet_names:
            raise KeyError(
                f"Sheet '{table_name}' not found in workbook. "
                f"Available sheets: {self._xl.sheet_names}"
            )
        df = self._xl.parse(table_name)
        logger.info("Loaded sheet '%s' with shape %s", table_name, df.shape)
        return _parse_known_datetime_columns(table_name, df)


class PostgresDataSource(BaseDataSource):
    """
    Loads tables directly from the operational PostgreSQL database.

    This mirrors ExcelDataSource's interface exactly, so switching the
    application from the Excel prototype to the live DB is a one-line
    change in `pipeline.py` / `api/service.py` (swap the data source
    class), nothing else needs to change.
    """

    def __init__(self, engine=None):
        # Imported lazily so that `psycopg`/`sqlalchemy` are only required
        # when someone actually uses the Postgres backend.
        from .db import get_engine

        self.engine = engine or get_engine()

    def load_table(self, table_name: str) -> pd.DataFrame:
        forbid_downstream_tables(table_name)
        query = f"SELECT * FROM {table_name}"  # table_name is never user input
        df = pd.read_sql(query, self.engine)
        logger.info("Loaded table '%s' from PostgreSQL with shape %s", table_name, df.shape)
        return _parse_known_datetime_columns(table_name, df)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_DATETIME_COLUMNS = {
    "hr_records": ["record_date"],
    "wellness_assessments": ["timestamp"],
    "biometric_data": ["timestamp"],
    "consent_records": ["timestamp"],
}


def _parse_known_datetime_columns(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    for col in _DATETIME_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_upstream_tables(source: BaseDataSource | None = None) -> Dict[str, pd.DataFrame]:
    """Convenience entry point: load every allowed upstream table at once."""
    source = source or ExcelDataSource()
    return source.load_all()
