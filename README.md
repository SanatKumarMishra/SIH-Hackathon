# SIH26186 — Feature Engineering Component

Consent-aware, leakage-free feature engineering pipeline for the AI-Powered
Welfare & Workload Analysis Platform. Converts HR + wellness + (optional)
biometric data into a stable, ML-ready feature vector for the team's
XGBoost/LightGBM risk model — this is decision-support tooling for an
early-warning welfare signal, **not** a clinical diagnosis system.

## 1. What this does

- Loads the four upstream tables (`personnel`, `hr_records`,
  `wellness_assessments`, `biometric_data`) plus `consent_records`.
- Cleans, merges, and consent-gates them into one wide per-personnel frame.
- Computes 14 engineered features (workload, wellness, biometric, and
  interaction signals) — see `feature_engineering/feature_dictionary.md`.
- Never reads `risk_predictions`, `risk_factors`, or `recommendations` —
  these downstream outputs are actively blocked at the data-loading layer.
- Exposes the pipeline both as a batch CSV generator and as a FastAPI
  microservice (`POST /features/generate/{personnel_id}`).

## 2. Data flow

```
personnel + hr_records + wellness_assessments + biometric_data + consent_records
    ↓ (feature_engineering/load_data.py — leakage guard blocks downstream tables)
preprocessing.py  — clean, merge, consent-gate
    ↓
features.py       — engineered features (workload, wellness, biometric, interactions)
    ↓
pipeline.py       — final ML-ready dataset / single-personnel feature vector
    ↓
data/processed/ml_features.csv   OR   api/routes.py (FastAPI) → Risk Engine / ML model
```

## 3. Source tables (from the provided workbook / PostgreSQL schema)

`personnel`, `hr_records`, `wellness_assessments`, `biometric_data`,
`consent_records` (upstream — used). `risk_predictions`, `risk_factors`,
`recommendations` (downstream outputs — never used as inputs).
`users`, `audit_logs` exist in the schema but are unrelated to feature
engineering.

The provided dataset is a **single-day snapshot** (2026-08-01, 20 personnel,
one record per table per person) — there is no real day-over-day history yet.

## 4. Engineered features

See `feature_engineering/feature_dictionary.md` for the full table of every
feature with its source columns, formula, purpose, and risk direction.
Summary: `workload_score`, `high_duty_flag`, `extended_deployment_flag`,
`leave_utilization_gap`, `wellness_score`, `fatigue_index`, `stress_burden`,
`low_mood_flag`, `sleep_deficit`, `low_activity_flag`, `low_hrv_flag`,
`workload_fatigue_interaction`, `stress_sleep_interaction`, `pressure_index`.

## 5. Preprocessing

`preprocessing.py` cleans each table (clips out-of-range values, fills
missing values with the column median, drops duplicate personnel records)
and merges them into one wide frame, with `personnel_id` as the join key.
Biometric columns are set to `NaN` (never imputed) when biometric consent is
absent — see `biometric_available`.

The sklearn `ColumnTransformer` in `pipeline.build_preprocessing_pipeline()`
(median-impute → standard-scale for numeric; most-frequent-impute → one-hot
for categorical) is only ever `.fit()` on the **training split**, then used
to `.transform()` train/val/test — this avoids train/test leakage.
`pipeline.group_aware_train_test_split()` splits by `personnel_id` so the
same person's records never appear in both splits.

## 6. Temporal feature handling

The dataset does not yet contain multi-day history, so real rolling/trend
features cannot be computed. `feature_engineering/temporal.py` contains the
real rolling-window logic (7/30/90-day means, change, percent change, trend
slope) — it checks `has_sufficient_history()` and only activates once real
multi-day records exist (e.g. once served from the live PostgreSQL DB in
production). `data/synthetic/generate_demo_history.py` is a clearly-labelled
synthetic generator that fabricates 30 days of demo history **for showcasing
the trend UI only** — its output is never read by the production pipeline or
used to train/evaluate the model.

## 7. PostgreSQL integration

`feature_engineering/db.py` builds a SQLAlchemy engine from environment
variables (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` — see
`.env.example`; never hard-coded). `feature_engineering/load_data.py` defines
`PostgresDataSource`, which implements the exact same `BaseDataSource`
interface as `ExcelDataSource`. To move from the Excel prototype to the live
DB, swap the data source instance passed into `run_pipeline()` /
`get_feature_vector()` / `api/service.py` — no other code changes, and the
existing PostgreSQL schema is not modified.

## 8. FastAPI integration

`api/routes.py` exposes:
- `POST /features/generate/{personnel_id}` → `{personnel_id, biometric_available, features: {...}}`
- `GET /health`

`api/service.py` is the thin bridge to `feature_engineering.pipeline.get_feature_vector`,
kept separate so it's testable without an HTTP server.

## 9. Install

```bash
cd project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # only needed for the PostgreSQL backend
```

## 10. Run the pipeline (batch CSV)

```bash
python -c "from feature_engineering.pipeline import run_pipeline; run_pipeline()"
```
Writes `data/processed/ml_features.csv`.

## 11. Run the API

```bash
uvicorn api.routes:app --reload --port 8001
```
Then:
```bash
curl -X POST http://localhost:8001/features/generate/3
```

## 11b. Run as a standalone microservice (Docker)

This component is packaged as its own containerized microservice, matching
the architecture doc's microservices layout (HR Data Service, Wellness
Service, Biometric Service, Risk Engine, etc. each as separate services
behind the API Gateway).

```bash
docker compose up --build
```

This starts:
- `feature-engineering-service` — this component, on `http://localhost:8001`
- `postgres` — a local Postgres instance (dev-only credentials in
  `docker-compose.yml`) pre-wired via env vars, so you can test the
  `PostgresDataSource` backend end-to-end

Then:
```bash
curl -X POST http://localhost:8001/features/generate/3
curl http://localhost:8001/health
```

To build/run just this container without compose:
```bash
docker build -t sih26186-feature-service .
docker run -p 8001:8001 sih26186-feature-service
```
(Without a Postgres container attached, the service falls back to the
bundled Excel prototype data baked into the image at `data/raw/`.)

To join your teammates' larger docker-compose network (API Gateway, HR
Data Service, Wellness Service, Risk Engine, etc.), drop the
`feature-engineering-service` block from `docker-compose.yml` into the
team's top-level compose file and put it on the shared network — no code
changes needed.

## 12. Run tests

```bash
pytest tests/ -v
```

## 13. How the ML teammate consumes the feature vector

Batch training:
```python
import pandas as pd
df = pd.read_csv("data/processed/ml_features.csv")
X = df[FEATURE_COLUMNS]   # see feature_engineering.features.ENGINEERED_FEATURE_COLUMNS
# y = <target column, kept separate, joined in only for training>
```

Live inference (called by the Risk Engine service):
```python
from feature_engineering.pipeline import get_feature_vector
payload = get_feature_vector(personnel_id=42)
X = [payload["features"][c] for c in ENGINEERED_FEATURE_COLUMNS]  # stable order for SHAP mapping
```
`ENGINEERED_FEATURE_COLUMNS` in `feature_engineering/features.py` is the
single source of truth for column order, so SHAP explanations can always be
mapped back to human-readable feature names.

## 14. Limitations

- **No real temporal history** in the current dataset — see section 6.
- Normalization (`workload_score`, etc.) is min-max scaled against the
  *current batch*; as more real data accrues this should be refit or
  replaced with fixed operational bounds.
- Thresholds (`duty_hours_high`, `low_activity_steps`, etc., in
  `feature_engineering/config.py`) are configurable heuristics, not
  clinically validated cutoffs.
- `consent_records` in the sample data has exactly one row per personnel
  covering a single data_type; the consent map treats a missing record for
  a data_type as "no consent" (fail-closed), which is the safe default but
  should be revisited once the real consent UX/schema is finalized.
- No target/label column was present in the upstream tables provided, so
  no supervised train/test evaluation was run — `group_aware_train_test_split`
  and `build_preprocessing_pipeline` are ready to use once a target is
  defined by the team.
