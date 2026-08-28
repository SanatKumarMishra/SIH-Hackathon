"""Tests for cleaning utilities, the sklearn preprocessing pipeline, and splitting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engineering.pipeline import (
    build_preprocessing_pipeline,
    group_aware_train_test_split,
    run_pipeline,
)
from feature_engineering.preprocessing import (
    clean_biometric_data,
    clean_hr_records,
    clean_wellness_assessments,
)


def test_clean_hr_records_clips_out_of_range_duty_hours():
    df = pd.DataFrame({
        "id": [1, 2],
        "personnel_id": [1, 2],
        "record_date": pd.to_datetime(["2026-08-01", "2026-08-01"]),
        "duty_hours": [30, -5],
        "deployment_days": [1, 1],
        "leave_days": [0, 0],
        "training_load": [1, 1],
    })
    cleaned = clean_hr_records(df)
    assert cleaned["duty_hours"].max() <= 24
    assert cleaned["duty_hours"].min() >= 0


def test_clean_wellness_clips_scores_to_0_10():
    df = pd.DataFrame({
        "id": [1, 2],
        "personnel_id": [1, 2],
        "timestamp": pd.to_datetime(["2026-08-01", "2026-08-01"]),
        "mood_score": [15, -3],
        "fatigue_score": [5, 5],
        "sleep_score": [5, 5],
        "stress_score": [5, 5],
    })
    cleaned = clean_wellness_assessments(df)
    assert cleaned["mood_score"].between(0, 10).all()


def test_clean_biometric_fills_missing_with_median():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "personnel_id": [1, 2, 3],
        "timestamp": pd.to_datetime(["2026-08-01"] * 3),
        "sleep_duration": [6.0, np.nan, 8.0],
        "activity": [5000, 6000, 7000],
        "hrv": [50, 60, 70],
    })
    cleaned = clean_biometric_data(df)
    assert cleaned["sleep_duration"].isna().sum() == 0


def test_group_aware_split_keeps_personnel_disjoint():
    df = pd.DataFrame({
        "personnel_id": list(range(1, 21)),
        "x": np.random.rand(20),
    })
    train, test = group_aware_train_test_split(df, test_size=0.3, random_state=1)
    assert set(train["personnel_id"]).isdisjoint(set(test["personnel_id"]))
    assert len(train) + len(test) == len(df)


def test_preprocessing_pipeline_fits_and_transforms_train_only():
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0],
        "num_b": [10.0, 20.0, 30.0, 40.0],
    })
    ct = build_preprocessing_pipeline(numeric_features=["num_a", "num_b"])
    train, test = df.iloc[:2], df.iloc[2:]
    ct.fit(train)  # fit on train only
    transformed_train = ct.transform(train)
    transformed_test = ct.transform(test)
    assert transformed_train.shape[0] == 2
    assert transformed_test.shape[0] == 2


def test_run_pipeline_writes_csv(tmp_path):
    output_path = tmp_path / "ml_features.csv"
    df = run_pipeline(output_path=output_path)
    assert output_path.exists()
    assert len(df) == len(pd.read_csv(output_path))
