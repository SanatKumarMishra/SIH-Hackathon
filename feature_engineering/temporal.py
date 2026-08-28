"""
Temporal (multi-day) feature scaffolding.

HONEST LIMITATION (read this first):
The current SIH26186_Complete_Dataset_10Tables.xlsx contains exactly ONE
observation per personnel for hr_records, wellness_assessments and
biometric_data (all dated 2026-08-01). There is no real day-over-day
history yet, so rolling averages, trend/slope, and deterioration
indicators CANNOT be computed from genuine data today.

This module is NOT faking those features. `has_sufficient_history()`
checks the real per-personnel observation count against
`config.MIN_OBSERVATIONS_FOR_ROLLING`, and `compute_rolling_features()`
only ever runs on real historical rows. When it is called on the
current single-snapshot dataset, it returns the frame unchanged (no
rolling columns added) and logs why.

Once the application has accumulated multiple days of hr_records /
wellness_assessments / biometric_data per personnel (i.e. once this
runs against the live PostgreSQL database in production), the exact
same function starts producing real 7/30/90-day rolling averages,
period-over-period change, and simple trend slope — no code change
required, only more real rows.

For an SIH demo that wants to *show* the trend UI working, use the
clearly-labelled synthetic generator in
`data/synthetic/generate_demo_history.py`, which writes to
`data/synthetic/` (never mixed into `data/processed/ml_features.csv`).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def has_sufficient_history(long_df: pd.DataFrame, id_col: str = "personnel_id") -> bool:
    """True if enough personnel have >= MIN_OBSERVATIONS_FOR_ROLLING records."""
    if long_df.empty or id_col not in long_df.columns:
        return False
    counts = long_df.groupby(id_col).size()
    sufficient = (counts >= config.MIN_OBSERVATIONS_FOR_ROLLING).sum()
    return sufficient > 0


def compute_rolling_features(
    long_df: pd.DataFrame,
    value_cols: list[str],
    date_col: str,
    id_col: str = "personnel_id",
) -> pd.DataFrame:
    """
    Computes rolling-window means, period-over-period change, percentage
    change, and a simple linear trend slope per personnel, for each column
    in `value_cols`, over the windows in `config.ROLLING_WINDOWS_DAYS`.

    Only runs if `has_sufficient_history(long_df)` is True; otherwise
    returns `long_df` unchanged and logs the limitation, so no caller
    ever silently receives fabricated temporal columns.

    Expects `long_df` in long format: one row per (personnel_id, date, ...).
    """
    if not has_sufficient_history(long_df, id_col):
        logger.info(
            "Insufficient per-personnel history (< %d observations) — "
            "skipping rolling temporal features. This is expected on the "
            "current single-snapshot dataset.",
            config.MIN_OBSERVATIONS_FOR_ROLLING,
        )
        return long_df

    df = long_df.sort_values([id_col, date_col]).copy()
    df = df.set_index(date_col)

    out_frames = []
    for pid, group in df.groupby(id_col):
        group = group.copy()
        for col in value_cols:
            if col not in group.columns:
                continue
            for window in config.ROLLING_WINDOWS_DAYS:
                group[f"{col}_roll{window}d_mean"] = (
                    group[col].rolling(f"{window}D", min_periods=1).mean()
                )
            group[f"{col}_change"] = group[col].diff()
            group[f"{col}_pct_change"] = group[col].pct_change()
            group[f"{col}_trend_slope"] = _rolling_slope(group[col])
        out_frames.append(group)

    result = pd.concat(out_frames).reset_index()
    return result


def _rolling_slope(series: pd.Series, window: int = 7) -> pd.Series:
    """Simple linear-regression slope of the last `window` points, per row."""

    def slope_of(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return float(np.polyfit(x, values, 1)[0])

    return series.rolling(window, min_periods=2).apply(slope_of, raw=True)


def deterioration_flag(slope: pd.Series, direction: str = "increasing_is_worse") -> pd.Series:
    """
    Flags sustained deterioration from a trend slope series.
    direction="increasing_is_worse" for e.g. stress/fatigue trending up.
    direction="decreasing_is_worse" for e.g. wellness/sleep trending down.
    """
    if direction == "increasing_is_worse":
        return (slope > 0).astype(int)
    return (slope < 0).astype(int)
