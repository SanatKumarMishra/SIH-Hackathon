"""
End-to-end, reproducible feature-engineering pipeline.

Public entry points:

- `run_pipeline()` — builds the full engineered dataset from raw tables
  and writes `data/processed/ml_features.csv`. Used for offline/batch
  dataset generation (e.g. for a teammate's model-training notebook).

- `get_feature_vector(personnel_id, source=None)` — returns a single,
  stably-ordered feature vector (dict) for one personnel, consent-aware.
  This is what the FastAPI service and the ML teammate's inference code
  both call.

- `build_preprocessing_pipeline()` — returns a scikit-learn
  ColumnTransformer/Pipeline for numeric scaling + categorical encoding,
  to be `.fit()` on the TRAIN split only (never on the full dataset) to
  avoid train/test leakage.

- `group_aware_train_test_split()` — splits by personnel_id so the same
  person never appears in both train and test.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, features
from .load_data import BaseDataSource, ExcelDataSource, load_upstream_tables
from .preprocessing import merge_upstream_tables

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Full dataset generation
# --------------------------------------------------------------------------
def run_pipeline(
    source: Optional[BaseDataSource] = None,
    output_path: Path | str = config.DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """Loads raw tables, cleans, merges, engineers features, and writes the CSV."""
    tables = load_upstream_tables(source)
    merged = merge_upstream_tables(tables)
    engineered = features.build_features(merged)

    final_columns = (
        features.IDENTIFIER_COLUMNS
        + features.METADATA_COLUMNS
        + features.RAW_NUMERIC_COLUMNS
        + features.ENGINEERED_FEATURE_COLUMNS
    )
    final_columns = [c for c in final_columns if c in engineered.columns]
    final_df = engineered[final_columns].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info("Wrote ML-ready feature dataset to %s (shape=%s)", output_path, final_df.shape)
    return final_df


# --------------------------------------------------------------------------
# Preprocessing pipeline (sklearn) — fit on TRAIN split only
# --------------------------------------------------------------------------
def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Returns an unfit ColumnTransformer:
      - numeric: median-impute -> standard-scale
      - categorical (if any): most-frequent-impute -> one-hot encode

    Caller is responsible for calling `.fit()` on X_train only, then
    `.transform()` on X_train/X_val/X_test, to avoid train/test leakage.
    """
    categorical_features = categorical_features or []

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = [("numeric", numeric_pipeline, numeric_features)]

    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def group_aware_train_test_split(
    df: pd.DataFrame,
    group_col: str = "personnel_id",
    test_size: float = 0.2,
    random_state: int = config.RANDOM_STATE,
):
    """
    Splits `df` into train/test such that no personnel_id appears in both
    splits (prevents leakage from the same person's records showing up on
    both sides). Falls back gracefully with a warning on tiny datasets
    where a group split would leave one side empty.
    """
    if df[group_col].nunique() < 2:
        logger.warning(
            "Only %d unique group(s) in '%s' — cannot perform a meaningful "
            "group-aware split; returning the full frame as both train and test.",
            df[group_col].nunique(), group_col,
        )
        return df.copy(), df.copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# --------------------------------------------------------------------------
# Single-personnel feature vector (used by the API / ML inference)
# --------------------------------------------------------------------------
class PersonnelNotFoundError(KeyError):
    pass


def get_feature_vector(personnel_id: int, source: Optional[BaseDataSource] = None) -> dict:
    """
    Returns a stable, ordered dict of engineered features for one
    personnel_id, consent-aware. Biometric-derived features are `None`
    (not zero) when biometric consent is absent, so the downstream model
    / caller can distinguish "no signal" from "signal is zero".

    Raises `PersonnelNotFoundError` if the personnel_id does not exist.
    """
    source = source or ExcelDataSource()
    tables = load_upstream_tables(source)

    if personnel_id not in set(tables["personnel"]["id"]):
        raise PersonnelNotFoundError(f"personnel_id={personnel_id} not found")

    merged = merge_upstream_tables(tables)
    engineered = features.build_features(merged)

    row = engineered.loc[engineered["personnel_id"] == personnel_id]
    if row.empty:
        raise PersonnelNotFoundError(f"personnel_id={personnel_id} has no derivable feature row")
    row = row.iloc[0]

    biometric_available = bool(row.get("biometric_available", False))

    feature_vector = {}
    for col in features.ENGINEERED_FEATURE_COLUMNS:
        value = row.get(col, np.nan)
        if pd.isna(value):
            feature_vector[col] = None
        else:
            feature_vector[col] = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value

    return {
        "personnel_id": int(personnel_id),
        "biometric_available": biometric_available,
        "features": feature_vector,
    }
