"""
Feature engineering: turns the merged, consent-gated wide frame from
`preprocessing.merge_upstream_tables` into the final set of engineered,
interpretable features.

Every function here documents, in its docstring, exactly which columns
it reads and what it produces — this backs `feature_dictionary.md`.

IMPORTANT — data actually available:
The current dataset is a SINGLE-DAY SNAPSHOT: hr_records, wellness_assessments
and biometric_data each contain exactly one row per personnel, all dated
2026-08-01. There is therefore no genuine day-over-day history yet, so
true rolling/trend temporal features (7d/30d/90d, deterioration slope)
CANNOT be computed from real data today. See `temporal.py` for how those
features are structured and how they activate automatically once
multi-day history exists, and `synthetic/generate_demo_history.py` for an
explicitly-labelled synthetic generator usable for an SIH demo.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

THRESH = config.THRESHOLDS


def add_hr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Workload / HR-derived features.

    Source columns: hr_duty_hours, hr_deployment_days, hr_leave_days, hr_training_load

    - workload_score: normalized composite of duty hours + deployment load,
      scaled 0-1 against dataset-observed max, i.e. relative workload intensity.
    - high_duty_flag: 1 if duty_hours exceeds the configured heavy-duty threshold.
    - extended_deployment_flag: 1 if deployment_days exceeds the configured threshold.
    - leave_utilization_gap: training_load minus leave_days — a simple proxy for
      "load taken on without a matching amount of rest/leave".
    """
    df = df.copy()
    duty = df["hr_duty_hours"]
    deployment = df["hr_deployment_days"]
    leave = df["hr_leave_days"]
    training = df["hr_training_load"]

    duty_norm = _safe_normalize(duty)
    deployment_norm = _safe_normalize(deployment)
    df["workload_score"] = (0.5 * duty_norm + 0.5 * deployment_norm).round(4)

    df["high_duty_flag"] = (duty > THRESH.duty_hours_high).astype(int)
    df["extended_deployment_flag"] = (deployment > THRESH.deployment_days_high).astype(int)
    df["leave_utilization_gap"] = (training - leave).astype(float)
    return df


def add_wellness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Self-reported wellness-derived features.

    Source columns: wellness_mood_score, wellness_fatigue_score,
    wellness_sleep_score, wellness_stress_score (all 0-10 self-report scales).

    - wellness_score: composite well-being score (higher = better), built from
      mood + sleep, minus fatigue + stress, rescaled to 0-1.
    - fatigue_index: normalized fatigue_score (0-1), higher = more fatigued.
    - stress_burden: normalized stress_score (0-1), higher = more stressed.
    - low_mood_flag: 1 if mood_score is in the bottom self-report range (<=3).
    """
    df = df.copy()
    mood = df["wellness_mood_score"]
    fatigue = df["wellness_fatigue_score"]
    sleep_q = df["wellness_sleep_score"]
    stress = df["wellness_stress_score"]

    composite = (mood + sleep_q) - (fatigue + stress)  # range roughly [-20, 20]
    df["wellness_score"] = ((composite + 20) / 40).round(4)
    df["fatigue_index"] = (fatigue / 10).round(4)
    df["stress_burden"] = (stress / 10).round(4)
    df["low_mood_flag"] = (mood <= 3).astype(int)
    return df


def add_biometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Biometric-derived features (consent-gated upstream in preprocessing).

    Source columns: bio_sleep_duration (hours), bio_activity (steps), bio_hrv (ms).
    Rows without biometric consent already have these columns as NaN; the
    resulting engineered features are NaN for those rows too, and
    `biometric_available` (from preprocessing) tells the model / API to
    treat them as missing rather than as a genuine zero-risk signal.

    - sleep_deficit: recommended_sleep_hours - actual sleep_duration (can be negative).
    - low_activity_flag: 1 if activity (steps) below the low-activity threshold.
    - low_hrv_flag: 1 if hrv below the low-HRV heuristic threshold (higher
      physiological strain heuristic — NOT a clinical measurement).
    """
    df = df.copy()
    df["sleep_deficit"] = (THRESH.recommended_sleep_hours - df["bio_sleep_duration"]).round(4)
    df["low_activity_flag"] = (df["bio_activity"] < THRESH.low_activity_steps).astype("Int64")
    df["low_hrv_flag"] = (df["bio_hrv"] < THRESH.low_hrv).astype("Int64")
    # Keep flags as nullable Int64 so NaN (no-consent) rows stay NaN, not 0.
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features combining workload and wellness/biometric signals.
    Only created from columns that exist after the prior feature steps.

    - workload_fatigue_interaction: workload_score * fatigue_index — captures
      compounding risk when high workload coincides with high fatigue.
    - stress_sleep_interaction: stress_burden * (sleep_deficit clipped to >=0,
      then min-max scaled) — captures compounding risk of stress plus
      insufficient sleep. NaN when biometric consent is absent.
    """
    df = df.copy()
    df["workload_fatigue_interaction"] = (df["workload_score"] * df["fatigue_index"]).round(4)

    if "sleep_deficit" in df.columns:
        deficit_pos = df["sleep_deficit"].clip(lower=0)
        deficit_norm = _safe_normalize(deficit_pos)
        df["stress_sleep_interaction"] = (df["stress_burden"] * deficit_norm).round(4)
    return df


def add_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    pressure_index: single composite "overall pressure" signal combining
    workload, stress, and fatigue in equal weight (0-1 scale). Documented
    as an interpretable summary feature, not a replacement for the model.

    Source columns: workload_score, stress_burden, fatigue_index.
    """
    df = df.copy()
    df["pressure_index"] = (
        (df["workload_score"] + df["stress_burden"] + df["fatigue_index"]) / 3
    ).round(4)
    return df


def _safe_normalize(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]; returns 0.5 for every value if the series is constant."""
    s_min, s_max = series.min(), series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Runs the full, ordered feature-engineering sequence on the merged frame."""
    df = merged.copy()
    df = add_hr_features(df)
    df = add_wellness_features(df)
    df = add_biometric_features(df)
    df = add_interaction_features(df)
    df = add_pressure_index(df)
    return df


# Ordered, stable list of engineered feature column names the model consumes.
# `pipeline.get_feature_vector` uses this list to guarantee column order,
# which matters for SHAP explainability downstream.
ENGINEERED_FEATURE_COLUMNS: list[str] = [
    "workload_score",
    "high_duty_flag",
    "extended_deployment_flag",
    "leave_utilization_gap",
    "wellness_score",
    "fatigue_index",
    "stress_burden",
    "low_mood_flag",
    "sleep_deficit",
    "low_activity_flag",
    "low_hrv_flag",
    "workload_fatigue_interaction",
    "stress_sleep_interaction",
    "pressure_index",
]

# Raw numeric inputs the model may also want, prior to any scaling.
RAW_NUMERIC_COLUMNS: list[str] = [
    "hr_duty_hours",
    "hr_deployment_days",
    "hr_leave_days",
    "hr_training_load",
    "wellness_mood_score",
    "wellness_fatigue_score",
    "wellness_sleep_score",
    "wellness_stress_score",
    "bio_sleep_duration",
    "bio_activity",
    "bio_hrv",
]

IDENTIFIER_COLUMNS: list[str] = ["personnel_id", "anonymized_identifier", "unit_id"]
METADATA_COLUMNS: list[str] = ["biometric_consent", "wellness_consent", "biometric_available"]
