# Robot Sensor Anomaly Detector

Detects faults in robot sensor streams (temperature, vibration, voltage) using an LSTM Autoencoder trained on normal operating data. Anomalies are flagged when reconstruction error exceeds a learned threshold — no labeled fault data required during training.

Outperforms Isolation Forest baseline by 10 F1 points.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **LSTM Autoencoder** | 0.860 | 0.780 | **0.818** | 0.887 |
| Isolation Forest | 0.734 | 0.703 | 0.718 | 0.813 |

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│   Data Layer        │     │   ML Layer               │     │   Serving Layer     │
│                     │     │                          │     │                     │
│  sensor_simulator   │────▶│  LSTM Autoencoder        │────▶│  FastAPI /predict   │
│  (3 sensor streams) │     │  (seq2seq, latent dim 16)│     │  /health  /history  │
│                     │     │                          │     │                     │
│  Fault types:       │     │  Isolation Forest        │     │  SQLite drift log   │
│  · spike            │     │  (baseline comparison)   │     │  Latency tracking   │
│  · flatline         │     │                          │     │  Drift detection    │
│  · drift            │     │  Threshold: p95 of       │     │                     │
│  · noise burst      │     │  normal recon error      │     └──────────┬──────────┘
└─────────────────────┘     └──────────────────────────┘                │
                                                                         ▼
                                                             ┌─────────────────────┐
                                                             │   React Dashboard   │
                                                             │                     │
                                                             │  Live sensor charts │
                                                             │  Anomaly alert feed │
                                                             │  Model health panel │
                                                             └─────────────────────┘
                                                                         │
                                                                         ▼
                                                             ┌─────────────────────┐
                                                             │   Infrastructure    │
                                                             │                     │
                                                             │  Docker (2-stage)   │
                                                             │  AWS EC2 t2.micro   │
                                                             │  Nginx + HTTPS      │
                                                             └─────────────────────┘
```

---

## Stack

Python · PyTorch · scikit-learn · FastAPI · React · Recharts · Docker · AWS EC2

---

## Quick start

```bash
git clone https://github.com/takshp2024-sys/robot-sensor-anomaly-detector
cd robot-sensor-anomaly-detector

pip install -r requirements.txt

# Phase 1 — generate synthetic sensor data with injected faults
python sensor_simulator.py
# outputs: sensor_data.csv, anomaly_labels.csv, sensor_plot.png

# Phase 2 — train LSTM Autoencoder + Isolation Forest baseline
python anomaly_model.py
# outputs: lstm_autoencoder.pt, model_results.png, model_comparison.csv

# Phase 3 — export scaler + threshold, start API
python save_model_artifacts.py
uvicorn main:app --reload --port 8000
# API live at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

---

## API usage

```bash
# Score a 30-timestep sensor window
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"readings": [[70.5, 1.01, 24.0], ...]}'   # 30 x [temp, vibration, voltage]

# Model health + drift score
curl http://localhost:8000/health

# Last 50 predictions
curl http://localhost:8000/history
```

Example `/predict` response:
```json
{
  "is_anomaly": false,
  "recon_error": 0.00000949,
  "threshold": 0.000713,
  "severity": "normal",
  "latency_ms": 3.98,
  "timestamp": "2026-03-15T14:22:01Z"
}
```

---

## Project structure

```
anomaly-detector/
│
├── sensor_simulator.py      # Generates synthetic sensor time-series with 4 fault types
│                            # (spike, flatline, drift, noise burst) injected at known windows
│
├── anomaly_model.py         # Trains LSTM Autoencoder on normal windows only;
│                            # evaluates against Isolation Forest baseline; saves weights
│
├── main.py                  # FastAPI backend — /predict, /health, /history endpoints;
│                            # logs every prediction to SQLite with latency + drift tracking
│
├── save_model_artifacts.py  # Exports scaler params + anomaly threshold to JSON
│                            # after training — run once before starting the API
│
├── SensorDashboard.jsx      # React dashboard — live Recharts sensor streams,
│                            # anomaly alert feed, model health stats panel
│
├── Dockerfile               # 2-stage build: builder installs deps, runtime copies app only
├── docker-compose.yml       # Container config with healthcheck + SQLite volume mount
├── requirements.txt         # CPU-only PyTorch build (~800 MB image vs 2.5 GB CUDA)
├── deploy.sh                # One-command deploy script for EC2
├── ec2_bootstrap.sh         # EC2 User Data script — installs Docker on first boot
├── scaler_params.json       # Sensor min/max values for MinMax normalization
└── threshold.json           # p95 reconstruction error cutoff from training
```

---

## How the model works

The LSTM Autoencoder is trained exclusively on **normal** sensor windows (30 timesteps × 3 sensors). The encoder compresses each window into a 16-dimensional latent vector; the decoder reconstructs the original sequence. After training, normal patterns reconstruct with low error. Anomalous patterns — spikes, flatlines, drifts — produce high reconstruction error because the model has never learned them.

The anomaly threshold is set at the 95th percentile of reconstruction errors on held-out normal windows, making it adaptive to the actual signal magnitude rather than a fixed value.

---

## Drift monitoring

The `/health` endpoint tracks a **drift score** — the ratio of mean reconstruction error over the last 50 predictions vs the first 50 baseline predictions. A score above 1.5 flips the API status to `degraded`, signaling the model may need retraining on updated sensor distributions.
