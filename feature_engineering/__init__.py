"""
Feature Engineering package for SIH26186 — AI-Powered Welfare & Workload
Analysis Platform.

This package converts raw upstream data (personnel, hr_records,
wellness_assessments, biometric_data, consent_records) into a stable,
leakage-free, ML-ready feature representation that a downstream
risk model (XGBoost / LightGBM) can consume.

Downstream/output tables (risk_predictions, risk_factors,
recommendations) are intentionally NEVER read by this package.
"""

__version__ = "0.1.0"
