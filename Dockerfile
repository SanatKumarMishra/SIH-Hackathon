# SIH26186 Feature Engineering Microservice
FROM python:3.11-slim

WORKDIR /app

# System deps needed for psycopg (PostgreSQL driver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_engineering/ ./feature_engineering/
COPY api/ ./api/
COPY data/raw/ ./data/raw/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8001"]
