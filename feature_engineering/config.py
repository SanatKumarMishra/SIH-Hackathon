"""
Central configuration for the feature engineering package.

Everything that is environment-specific (file paths, DB credentials,
thresholds) lives here so the rest of the code stays declarative and
easy for a teammate to tune without touching pipeline logic.

Nothing in this file should ever contain a real secret — DB
credentials are read from environment variables / a .env file (see
`db.py`), never hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

DEFAULT_XLSX_PATH = DATA_RAW_DIR / "SIH26186_Complete_Dataset_10Tables.xlsx"
DEFAULT_OUTPUT_CSV = DATA_PROCESSED_DIR / "ml_features.csv"
FEATURE_DICTIONARY_PATH = Path(__file__).resolve().parent / "feature_dictionary.md"

# --------------------------------------------------------------------------
# Upstream / downstream table classification (data-leakage guard)
# --------------------------------------------------------------------------
# These tables are the ONLY tables the feature pipeline is allowed to read.
UPSTREAM_TABLES = {
    "personnel",
    "hr_records",
    "wellness_assessments",
    "biometric_data",
    "consent_records",
}

# These tables are downstream MODEL OUTPUTS. They must never be used to
# build predictive features. `pipeline.py` actively refuses to load them
# for feature purposes (see `load_data.forbid_downstream_tables`).
DOWNSTREAM_TABLES = {
    "risk_predictions",
    "risk_factors",
    "recommendations",
}

# Tables that exist in the schema but are not relevant to feature
# engineering at all (auth/audit concerns).
IGNORED_TABLES = {"users", "audit_logs"}

# --------------------------------------------------------------------------
# Consent
# --------------------------------------------------------------------------
CONSENT_BIOMETRIC_LABEL = "Biometric Data"
CONSENT_WELLNESS_LABEL = "Wellness Data"

# --------------------------------------------------------------------------
# Risk-oriented thresholds used by simple, interpretable derived features.
# These are intentionally conservative, documented defaults — they encode
# no clinical claim, only descriptive thresholds for early-warning signals.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    duty_hours_high: float = 12.0          # hours/day considered heavy duty
    deployment_days_high: float = 15.0     # consecutive-style deployment load
    recommended_sleep_hours: float = 7.0   # baseline for sleep_deficit
    low_activity_steps: float = 4000.0     # steps/day considered low activity
    low_hrv: float = 50.0                  # ms, lower HRV ~ more strain (heuristic)


THRESHOLDS = Thresholds()

# --------------------------------------------------------------------------
# Temporal feature settings
# --------------------------------------------------------------------------
MIN_OBSERVATIONS_FOR_ROLLING = 3  # per-personnel records needed before a
                                  # rolling window feature is considered
                                  # meaningful rather than noise.
ROLLING_WINDOWS_DAYS = (7, 30, 90)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# PostgreSQL (used only by feature_engineering/db.py; values come from env)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PostgresSettings:
    host: str = field(default_factory=lambda: os.getenv("PGHOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("PGPORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("PGDATABASE", "sih26186"))
    user: str = field(default_factory=lambda: os.getenv("PGUSER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("PGPASSWORD", ""))

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
