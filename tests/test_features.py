"""Tests covering data loading, merging, feature calculation, and leakage guards."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engineering import config, features
from feature_engineering.load_data import DataLeakageError, ExcelDataSource, load_upstream_tables
from feature_engineering.preprocessing import build_consent_map, merge_upstream_tables


@pytest.fixture(scope="module")
def source() -> ExcelDataSource:
    return ExcelDataSource(config.DEFAULT_XLSX_PATH)


@pytest.fixture(scope="module")
def tables(source):
    return load_upstream_tables(source)


@pytest.fixture(scope="module")
def merged(tables):
    return merge_upstream_tables(tables)


@pytest.fixture(scope="module")
def engineered(merged):
    return features.build_features(merged)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def test_load_all_upstream_tables_present(tables):
    for name in config.UPSTREAM_TABLES:
        assert name in tables
        assert isinstance(tables[name], pd.DataFrame)
        assert not tables[name].empty


def test_downstream_tables_cannot_be_loaded(source):
    for downstream in config.DOWNSTREAM_TABLES:
        with pytest.raises(DataLeakageError):
            source.load_table(downstream)


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------
def test_merge_produces_one_row_per_personnel(tables, merged):
    assert len(merged) == len(tables["personnel"])
    assert merged["personnel_id"].is_unique


def test_merge_includes_expected_prefixed_columns(merged):
    for col in ["hr_duty_hours", "wellness_stress_score", "bio_sleep_duration"]:
        assert col in merged.columns


def test_consent_map_shape_and_types(tables):
    consent_map = build_consent_map(tables["consent_records"])
    assert set(consent_map.columns) == {"personnel_id", "wellness_consent", "biometric_consent"}
    assert consent_map["wellness_consent"].dtype == bool
    assert consent_map["biometric_consent"].dtype == bool


def test_biometric_consent_gate_blanks_non_consenting_rows(tables, merged):
    consent_map = build_consent_map(tables["consent_records"])
    no_consent_ids = consent_map.loc[~consent_map["biometric_consent"], "personnel_id"]
    gated_rows = merged[merged["personnel_id"].isin(no_consent_ids)]
    assert gated_rows["bio_sleep_duration"].isna().all()
    assert (gated_rows["biometric_available"] == False).all()  # noqa: E712


def test_biometric_consent_gate_can_be_disabled(tables):
    ungated = merge_upstream_tables(tables, apply_consent_gate=False)
    # With gating off, personnel who have a biometric record should keep it.
    assert ungated["bio_sleep_duration"].notna().sum() > 0


# --------------------------------------------------------------------------
# Feature calculations
# --------------------------------------------------------------------------
def test_engineered_columns_present(engineered):
    for col in features.ENGINEERED_FEATURE_COLUMNS:
        assert col in engineered.columns


def test_workload_score_bounded_0_1(engineered):
    scores = engineered["workload_score"].dropna()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_wellness_score_bounded_0_1(engineered):
    scores = engineered["wellness_score"].dropna()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_sleep_deficit_only_present_when_biometric_available(engineered):
    no_bio = engineered[~engineered["biometric_available"]]
    assert no_bio["sleep_deficit"].isna().all()
    has_bio = engineered[engineered["biometric_available"]]
    assert has_bio["sleep_deficit"].notna().all()


def test_high_duty_flag_matches_threshold(engineered):
    threshold = config.THRESHOLDS.duty_hours_high
    expected = (engineered["hr_duty_hours"] > threshold).astype(int)
    assert (engineered["high_duty_flag"] == expected).all()


def test_pressure_index_bounded_0_1(engineered):
    values = engineered["pressure_index"].dropna()
    assert (values >= 0).all() and (values <= 1).all()


# --------------------------------------------------------------------------
# Leakage: final feature output must never contain downstream columns
# --------------------------------------------------------------------------
def test_no_downstream_columns_in_engineered_output(engineered):
    forbidden_substrings = ["risk_score", "risk_level", "risk_factor", "recommendation"]
    lowered_cols = [c.lower() for c in engineered.columns]
    for forbidden in forbidden_substrings:
        assert not any(forbidden in c for c in lowered_cols), f"Leakage: '{forbidden}' found in feature columns"
