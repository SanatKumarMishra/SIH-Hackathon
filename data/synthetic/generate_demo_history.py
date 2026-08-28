"""
*** SYNTHETIC DATA — FOR SIH DEMO PURPOSES ONLY ***

The real dataset (SIH26186_Complete_Dataset_10Tables.xlsx) contains only
ONE observation per personnel (2026-08-01), so genuine rolling/trend
temporal features cannot be computed. This script fabricates a
30-day history per personnel by randomly perturbing the single real
observation day over day, purely so the demo UI can showcase the
7/30-day trend charts described in the architecture doc.

This output is written ONLY to data/synthetic/ and is NEVER read by
the production feature pipeline (feature_engineering/pipeline.py only
reads data/raw/ or PostgreSQL). Do not use this file's output to train
or evaluate a real model — it contains no real signal, only noise
around a single real day.

Usage:
    python data/synthetic/generate_demo_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.config import DEFAULT_XLSX_PATH, DATA_SYNTHETIC_DIR  # noqa: E402

RNG = np.random.default_rng(seed=42)
N_DAYS = 30


def _perturb(base: pd.Series, day_offset: int, scale: float, integer: bool = False) -> pd.Series:
    noise = RNG.normal(loc=0, scale=scale, size=len(base))
    drift = day_offset * RNG.normal(loc=0, scale=scale * 0.05, size=len(base))  # gentle random walk
    values = base.to_numpy(dtype=float) + noise + drift
    if integer:
        values = np.round(values)
    return pd.Series(values, index=base.index)


def generate_synthetic_hr_history(hr: pd.DataFrame) -> pd.DataFrame:
    frames = []
    base_date = pd.to_datetime(hr["record_date"]).min()
    for day in range(N_DAYS):
        day_df = hr.copy()
        day_df["record_date"] = base_date + pd.Timedelta(days=day)
        day_df["duty_hours"] = _perturb(hr["duty_hours"], day, scale=1.5, integer=True).clip(0, 24)
        day_df["deployment_days"] = _perturb(hr["deployment_days"], day, scale=1.0, integer=True).clip(lower=0)
        day_df["leave_days"] = hr["leave_days"]  # leave changes rarely; keep as-is per demo day
        day_df["training_load"] = _perturb(hr["training_load"], day, scale=1.0, integer=True).clip(lower=0)
        day_df["id"] = range(len(frames) * len(hr) + 1, len(frames) * len(hr) + 1 + len(hr))
        frames.append(day_df)
    return pd.concat(frames, ignore_index=True)


def generate_synthetic_wellness_history(we: pd.DataFrame) -> pd.DataFrame:
    frames = []
    base_ts = pd.to_datetime(we["timestamp"]).min()
    for day in range(N_DAYS):
        day_df = we.copy()
        day_df["timestamp"] = base_ts + pd.Timedelta(days=day)
        for col in ["mood_score", "fatigue_score", "sleep_score", "stress_score"]:
            day_df[col] = _perturb(we[col], day, scale=1.2, integer=True).clip(0, 10)
        day_df["id"] = range(len(frames) * len(we) + 1, len(frames) * len(we) + 1 + len(we))
        frames.append(day_df)
    return pd.concat(frames, ignore_index=True)


def generate_synthetic_biometric_history(bio: pd.DataFrame) -> pd.DataFrame:
    frames = []
    base_ts = pd.to_datetime(bio["timestamp"]).min()
    for day in range(N_DAYS):
        day_df = bio.copy()
        day_df["timestamp"] = base_ts + pd.Timedelta(days=day)
        day_df["sleep_duration"] = _perturb(bio["sleep_duration"], day, scale=0.6).clip(0, 24)
        day_df["activity"] = _perturb(bio["activity"], day, scale=800, integer=True).clip(lower=0)
        day_df["hrv"] = _perturb(bio["hrv"], day, scale=5, integer=True).clip(lower=0)
        day_df["id"] = range(len(frames) * len(bio) + 1, len(frames) * len(bio) + 1 + len(bio))
        frames.append(day_df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    xl = pd.ExcelFile(DEFAULT_XLSX_PATH)
    hr = xl.parse("hr_records")
    we = xl.parse("wellness_assessments")
    bio = xl.parse("biometric_data")

    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    generate_synthetic_hr_history(hr).to_csv(DATA_SYNTHETIC_DIR / "hr_records_30day.csv", index=False)
    generate_synthetic_wellness_history(we).to_csv(DATA_SYNTHETIC_DIR / "wellness_assessments_30day.csv", index=False)
    generate_synthetic_biometric_history(bio).to_csv(DATA_SYNTHETIC_DIR / "biometric_data_30day.csv", index=False)

    print(f"Wrote {N_DAYS}-day SYNTHETIC demo history to {DATA_SYNTHETIC_DIR}")
    print("Reminder: this data is for demo visuals only — never use it for real model training/eval.")


if __name__ == "__main__":
    main()
