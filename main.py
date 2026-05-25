"""
Robot Sensor Anomaly Detector — Phase 3
FastAPI backend with /predict, /health, and /history endpoints.

Endpoints:
    POST /predict   → score a window of sensor readings
    GET  /health    → model drift stats + latency + anomaly rate
    GET  /history   → last N logged predictions
    GET  /docs      → auto-generated Swagger UI (free from FastAPI)

Usage:
    uvicorn main:app --reload --port 8000
"""

import time
import json
import sqlite3
import statistics
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "lstm_autoencoder.pt"
DB_PATH     = BASE_DIR / "drift_log.db"
SCALER_PATH = BASE_DIR / "scaler_params.json"

# ── Model constants (must match training) ────────────────────────────────────

SEQ_LEN    = 30
INPUT_DIM  = 3       # temperature, vibration, voltage
HIDDEN_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anomaly threshold — p95 from training (saved alongside the model)
# Falls back to a sensible default if not stored
DEFAULT_THRESHOLD = 0.000713


# ── LSTM Autoencoder (identical architecture to training) ─────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS,
                            batch_first=True, dropout=0.2)
        self.out  = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z):
        h0 = self.fc(z).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        out, _ = self.lstm(h0)
        return self.out(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Scaler (MinMax — loaded from JSON saved at training time) ─────────────────

class SensorScaler:
    """Lightweight MinMaxScaler that works from saved min/max arrays."""
    def __init__(self, data_min: list, data_max: list):
        self.min = np.array(data_min, dtype=np.float32)
        self.max = np.array(data_max, dtype=np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min) / (self.max - self.min + 1e-8)

    @classmethod
    def from_json(cls, path: Path):
        with open(path) as f:
            p = json.load(f)
        return cls(p["data_min"], p["data_max"])

    @classmethod
    def default(cls):
        """Fallback scaler using ranges from Phase 1 simulation."""
        return cls(
            data_min=[65.0, 0.70, 23.5],
            data_max=[85.0, 2.80, 27.5],
        )


# ── SQLite drift log ──────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,
            recon_error   REAL    NOT NULL,
            is_anomaly    INTEGER NOT NULL,
            latency_ms    REAL    NOT NULL,
            temp_mean     REAL,
            vib_mean      REAL,
            volt_mean     REAL
        )
    """)
    conn.commit()
    return conn


def log_prediction(conn, recon_error, is_anomaly, latency_ms,
                   temp_mean, vib_mean, volt_mean):
    conn.execute(
        """INSERT INTO predictions
           (ts, recon_error, is_anomaly, latency_ms, temp_mean, vib_mean, volt_mean)
           VALUES (?,?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(),
         float(recon_error), int(is_anomaly), float(latency_ms),
         float(temp_mean), float(vib_mean), float(volt_mean))
    )
    conn.commit()


# ── App state (loaded once at startup) ───────────────────────────────────────

class AppState:
    model:     LSTMAutoencoder
    scaler:    SensorScaler
    threshold: float
    db:        sqlite3.Connection
    boot_time: str

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + scaler + DB on startup; close DB on shutdown."""
    # Model
    state.model = LSTMAutoencoder().to(DEVICE)
    if MODEL_PATH.exists():
        state.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        print(f"[✓] Loaded model from {MODEL_PATH}")
    else:
        print(f"[!] {MODEL_PATH} not found — using random weights (demo mode)")
    state.model.eval()

    # Scaler
    state.scaler = (SensorScaler.from_json(SCALER_PATH)
                    if SCALER_PATH.exists()
                    else SensorScaler.default())

    # Threshold — load from JSON if saved, else use training default
    threshold_path = BASE_DIR / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            state.threshold = json.load(f)["threshold"]
    else:
        state.threshold = DEFAULT_THRESHOLD

    # DB
    state.db = init_db(DB_PATH)
    state.boot_time = datetime.now(timezone.utc).isoformat()
    print(f"[✓] API ready on {DEVICE} | threshold={state.threshold:.6f}")
    yield

    state.db.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Robot Sensor Anomaly Detector",
    description="LSTM Autoencoder API — detects faults in robot sensor streams.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SensorWindow(BaseModel):
    """
    A sliding window of sensor readings.
    Each inner list is one timestep: [temperature, vibration, voltage].
    Must contain exactly SEQ_LEN (30) timesteps.
    """
    readings: list[list[float]] = Field(
        ...,
        description=f"List of {SEQ_LEN} timesteps, each [temp_°C, vib_g, volt_V]",
        examples=[[[70.1, 1.02, 24.0]] * 30],
    )

    @field_validator("readings")
    @classmethod
    def check_shape(cls, v):
        if len(v) != SEQ_LEN:
            raise ValueError(f"Expected {SEQ_LEN} timesteps, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != INPUT_DIM:
                raise ValueError(
                    f"Timestep {i}: expected {INPUT_DIM} sensors, got {len(row)}"
                )
        return v


class PredictResponse(BaseModel):
    is_anomaly:    bool
    recon_error:   float = Field(description="Mean squared reconstruction error")
    threshold:     float
    severity:      str   = Field(description="normal | low | medium | high | critical")
    latency_ms:    float
    timestamp:     str


class HealthResponse(BaseModel):
    status:           str
    uptime_since:     str
    total_predictions: int
    anomaly_rate_pct: float
    avg_latency_ms:   float
    p95_latency_ms:   float
    avg_recon_error:  float
    drift_score:      float = Field(
        description="Ratio of recent vs baseline error — >1.5 signals drift"
    )
    threshold:        float
    device:           str


class PredictionRecord(BaseModel):
    id:          int
    ts:          str
    recon_error: float
    is_anomaly:  bool
    latency_ms:  float
    temp_mean:   float | None
    vib_mean:    float | None
    volt_mean:   float | None


# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_label(error: float, threshold: float) -> str:
    ratio = error / threshold if threshold > 0 else 0
    if ratio < 1.0:   return "normal"
    if ratio < 1.5:   return "low"
    if ratio < 2.5:   return "medium"
    if ratio < 4.0:   return "high"
    return "critical"


def compute_drift(conn: sqlite3.Connection, window: int = 50) -> float:
    """
    Drift score = mean recon_error of last `window` predictions
                  divided by mean recon_error of the first `window` predictions.
    Returns 1.0 (no drift) if not enough data yet.
    """
    rows = conn.execute(
        "SELECT recon_error FROM predictions ORDER BY id"
    ).fetchall()
    if len(rows) < window * 2:
        return 1.0
    baseline = statistics.mean(r[0] for r in rows[:window])
    recent   = statistics.mean(r[0] for r in rows[-window:])
    return round(recent / baseline, 4) if baseline > 0 else 1.0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(body: SensorWindow):
    """
    Score a 30-timestep sensor window.

    Send an array of 30 readings — each reading is
    `[temperature_°C, vibration_g, voltage_V]` — and receive back
    a reconstruction error, anomaly flag, and severity label.
    """
    t0 = time.perf_counter()

    # Scale
    raw = np.array(body.readings, dtype=np.float32)   # (30, 3)
    scaled = state.scaler.transform(raw)               # (30, 3)

    # Infer
    tensor = torch.tensor(scaled).unsqueeze(0).to(DEVICE)  # (1, 30, 3)
    with torch.no_grad():
        recon = state.model(tensor)
    error = float(((tensor - recon) ** 2).mean().cpu())

    latency_ms = (time.perf_counter() - t0) * 1000
    is_anomaly = error > state.threshold
    severity   = severity_label(error, state.threshold)
    ts         = datetime.now(timezone.utc).isoformat()

    # Log to DB
    log_prediction(
        state.db, error, is_anomaly, latency_ms,
        float(raw[:, 0].mean()),
        float(raw[:, 1].mean()),
        float(raw[:, 2].mean()),
    )

    return PredictResponse(
        is_anomaly=is_anomaly,
        recon_error=round(error, 8),
        threshold=round(state.threshold, 8),
        severity=severity,
        latency_ms=round(latency_ms, 3),
        timestamp=ts,
    )


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health():
    """
    Return model health stats: anomaly rate, latency percentiles,
    and a drift score comparing recent vs baseline reconstruction error.

    A drift_score > 1.5 indicates the model may need retraining.
    """
    rows = state.db.execute(
        "SELECT recon_error, is_anomaly, latency_ms FROM predictions"
    ).fetchall()

    total = len(rows)
    if total == 0:
        return HealthResponse(
            status="healthy", uptime_since=state.boot_time,
            total_predictions=0, anomaly_rate_pct=0.0,
            avg_latency_ms=0.0, p95_latency_ms=0.0,
            avg_recon_error=0.0, drift_score=1.0,
            threshold=state.threshold, device=str(DEVICE),
        )

    errors   = [r[0] for r in rows]
    anomalies = [r[1] for r in rows]
    latencies = [r[2] for r in rows]

    latencies_sorted = sorted(latencies)
    p95_idx  = int(0.95 * len(latencies_sorted))

    drift = compute_drift(state.db)
    status = "degraded" if drift > 1.5 else "healthy"

    return HealthResponse(
        status=status,
        uptime_since=state.boot_time,
        total_predictions=total,
        anomaly_rate_pct=round(sum(anomalies) / total * 100, 2),
        avg_latency_ms=round(statistics.mean(latencies), 3),
        p95_latency_ms=round(latencies_sorted[p95_idx], 3),
        avg_recon_error=round(statistics.mean(errors), 8),
        drift_score=drift,
        threshold=round(state.threshold, 8),
        device=str(DEVICE),
    )


@app.get("/history", response_model=list[PredictionRecord], tags=["monitoring"])
async def history(limit: int = 50):
    """Return the last `limit` predictions (default 50, max 500)."""
    if limit > 500:
        raise HTTPException(400, "limit must be ≤ 500")
    rows = state.db.execute(
        """SELECT id, ts, recon_error, is_anomaly, latency_ms,
                  temp_mean, vib_mean, volt_mean
           FROM predictions ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    return [
        PredictionRecord(
            id=r[0], ts=r[1], recon_error=r[2], is_anomaly=bool(r[3]),
            latency_ms=r[4], temp_mean=r[5], vib_mean=r[6], volt_mean=r[7],
        )
        for r in rows
    ]


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "Robot Sensor Anomaly Detector",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }