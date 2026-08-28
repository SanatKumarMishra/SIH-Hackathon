"""
Preprocessing: cleaning, merging, and consent-gating of the raw upstream
tables into a single per-personnel "wide" frame that `features.py` can
build engineered features on top of.

This module does NOT fit any scaler/encoder — that happens later in
`pipeline.py`'s scikit-learn ColumnTransformer, and only on the
training split, to avoid train/test leakage.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Consent
# --------------------------------------------------------------------------
def build_consent_map(consent_records: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses the long-format `consent_records` table into one row per
    personnel with two boolean flags: `wellness_consent`, `biometric_consent`.

    In the observed dataset each personnel has exactly one consent record
    covering a single data_type. Absence of a record for a given data_type
    is treated as "no consent" (fail-closed), which is the safe default
    for an optional, consent-gated data source.
    """
    if consent_records.empty:
        return pd.DataFrame(columns=["personnel_id", "wellness_consent", "biometric_consent"])

    pivot = (
        consent_records
        .assign(consent=consent_records["consent"].astype(bool))
        .pivot_table(
            index="personnel_id",
            columns="data_type",
            values="consent",
            aggfunc="max",  # if duplicated, any True wins
        )
        .reindex(columns=[config.CONSENT_WELLNESS_LABEL, config.CONSENT_BIOMETRIC_LABEL])
        .fillna(False)
        .reset_index()
    )
    pivot = pivot.rename(
        columns={
            config.CONSENT_WELLNESS_LABEL: "wellness_consent",
            config.CONSENT_BIOMETRIC_LABEL: "biometric_consent",
        }
    )
    pivot["wellness_consent"] = pivot["wellness_consent"].astype(bool)
    pivot["biometric_consent"] = pivot["biometric_consent"].astype(bool)
    return pivot


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------
def _drop_duplicate_records(df: pd.DataFrame, subset_id: str, time_col: str | None) -> pd.DataFrame:
    """Keeps the most recent record per personnel if duplicates exist."""
    before = len(df)
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).drop_duplicates(subset=[subset_id], keep="last")
    else:
        df = df.drop_duplicates(subset=[subset_id], keep="last")
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate record(s) on %s", dropped, subset_id)
    return df


def _clip_invalid_scores(df: pd.DataFrame, columns: list[str], lo: float, hi: float) -> pd.DataFrame:
    """Clips self-report scores to their valid scale and flags out-of-range input."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        invalid_mask = (df[col] < lo) | (df[col] > hi)
        n_invalid = int(invalid_mask.sum())
        if n_invalid:
            logger.warning("Clipping %d out-of-range value(s) in '%s' to [%s, %s]", n_invalid, col, lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def _clip_non_negative(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        n_invalid = int((df[col] < 0).sum())
        if n_invalid:
            logger.warning("Clipping %d negative value(s) in '%s' to 0", n_invalid, col)
        df[col] = df[col].clip(lower=0)
    return df


# --------------------------------------------------------------------------
# Per-table cleaning
# --------------------------------------------------------------------------
def clean_hr_records(hr: pd.DataFrame) -> pd.DataFrame:
    hr = _drop_duplicate_records(hr, "personnel_id", "record_date")
    hr = _clip_non_negative(hr, ["duty_hours", "deployment_days", "leave_days", "training_load"])
    # duty_hours cannot exceed 24/day; deployment/training loads are counts, no fixed ceiling in schema
    if "duty_hours" in hr.columns:
        n_invalid = int((hr["duty_hours"] > 24).sum())
        if n_invalid:
            logger.warning("Clipping %d duty_hours value(s) above 24", n_invalid)
        hr["duty_hours"] = hr["duty_hours"].clip(upper=24)
    for col in ["duty_hours", "deployment_days", "leave_days", "training_load"]:
        if col in hr.columns and hr[col].isna().any():
            median = hr[col].median()
            hr[col] = hr[col].fillna(median)
    return hr


def clean_wellness_assessments(we: pd.DataFrame) -> pd.DataFrame:
    we = _drop_duplicate_records(we, "personnel_id", "timestamp")
    score_cols = ["mood_score", "fatigue_score", "sleep_score", "stress_score"]
    we = _clip_invalid_scores(we, score_cols, lo=0, hi=10)
    for col in score_cols:
        if col in we.columns and we[col].isna().any():
            we[col] = we[col].fillna(we[col].median())
    return we


def clean_biometric_data(bio: pd.DataFrame) -> pd.DataFrame:
    bio = _drop_duplicate_records(bio, "personnel_id", "timestamp")
    bio = _clip_non_negative(bio, ["sleep_duration", "activity", "hrv"])
    if "sleep_duration" in bio.columns:
        n_invalid = int((bio["sleep_duration"] > 24).sum())
        if n_invalid:
            logger.warning("Clipping %d sleep_duration value(s) above 24h", n_invalid)
        bio["sleep_duration"] = bio["sleep_duration"].clip(upper=24)
    for col in ["sleep_duration", "activity", "hrv"]:
        if col in bio.columns and bio[col].isna().any():
            bio[col] = bio[col].fillna(bio[col].median())
    return bio


def clean_personnel(personnel: pd.DataFrame) -> pd.DataFrame:
    return personnel.drop_duplicates(subset=["id"], keep="last")


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------
def merge_upstream_tables(
    tables: Dict[str, pd.DataFrame],
    *,
    apply_consent_gate: bool = True,
) -> pd.DataFrame:
    """
    Cleans and merges personnel + hr_records + wellness_assessments +
    biometric_data + consent_records into a single wide, per-personnel frame.

    When `apply_consent_gate` is True (default), biometric columns are
    set to NaN for any personnel who has not explicitly consented to
    "Biometric Data". A `biometric_available` flag column records whether
    biometric features are usable for that row, so downstream code never
    has to guess.
    """
    personnel = clean_personnel(tables["personnel"])
    hr = clean_hr_records(tables["hr_records"])
    wellness = clean_wellness_assessments(tables["wellness_assessments"])
    biometric = clean_biometric_data(tables["biometric_data"])
    consent_map = build_consent_map(tables["consent_records"])

    # personnel_id in the other tables refers to personnel.id
    merged = personnel.rename(columns={"id": "personnel_id"}).merge(
        hr.drop(columns=["id"], errors="ignore").rename(
            columns={c: f"hr_{c}" for c in hr.columns if c not in ("id", "personnel_id")}
        ),
        on="personnel_id",
        how="left",
    )
    merged = merged.merge(
        wellness.drop(columns=["id"], errors="ignore").rename(
            columns={c: f"wellness_{c}" for c in wellness.columns if c not in ("id", "personnel_id")}
        ),
        on="personnel_id",
        how="left",
    )
    merged = merged.merge(
        biometric.drop(columns=["id"], errors="ignore").rename(
            columns={c: f"bio_{c}" for c in biometric.columns if c not in ("id", "personnel_id")}
        ),
        on="personnel_id",
        how="left",
    )
    merged = merged.merge(consent_map, on="personnel_id", how="left")
    merged["wellness_consent"] = merged["wellness_consent"].fillna(False).astype(bool)
    merged["biometric_consent"] = merged["biometric_consent"].fillna(False).astype(bool)

    bio_cols = [c for c in merged.columns if c.startswith("bio_")]

    if apply_consent_gate:
        no_consent_mask = ~merged["biometric_consent"]
        merged.loc[no_consent_mask, bio_cols] = np.nan
        logger.info(
            "Consent gate applied: biometric data withheld for %d/%d personnel",
            int(no_consent_mask.sum()), len(merged),
        )

    merged["biometric_available"] = merged[bio_cols].notna().all(axis=1) if bio_cols else False

    return merged
