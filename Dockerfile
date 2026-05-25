# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install dependencies in an isolated layer so the final image stays lean.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application files
COPY main.py               ./main.py
COPY save_model_artifacts.py ./save_model_artifacts.py
COPY lstm_autoencoder.pt   ./lstm_autoencoder.pt
COPY scaler_params.json    ./scaler_params.json
COPY threshold.json        ./threshold.json

# Create a non-root user — never run production containers as root
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# SQLite drift log will be written here at runtime
VOLUME ["/app/data"]

EXPOSE 8000

# Uvicorn with 2 workers; increase for higher-traffic deployments
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]