"""
save_model_artifacts.py
Run this ONCE after Phase 2 training to export the scaler params
and threshold into JSON files the API can load at startup.

Usage (from your project root):
    python save_model_artifacts.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

SENSORS    = ["temperature", "vibration", "voltage"]
DATA_PATH  = Path("sensor_data.csv")
OUT_DIR    = Path(".")          # same folder as main.py

# ── Rebuild scaler from training data ────────────────────────────────────────

df     = pd.read_csv(DATA_PATH)
scaler = MinMaxScaler()
scaler.fit(df[SENSORS].values)

scaler_params = {
    "data_min": scaler.data_min_.tolist(),
    "data_max": scaler.data_max_.tolist(),
    "feature_names": SENSORS,
}
with open(OUT_DIR / "scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print("[✓] scaler_params.json saved")

# ── Save threshold (p95 reconstruction error on normal windows) ───────────────
# This matches the value printed during Phase 2 training.
# If you want to recompute it, load the model and run reconstruction_errors()
# on normal windows, then take np.percentile(normal_errors, 95).

THRESHOLD = 0.000713   # ← paste the value printed by anomaly_model.py here

with open(OUT_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": THRESHOLD, "percentile": 95}, f, indent=2)
print(f"[✓] threshold.json saved  (threshold={THRESHOLD})")

print("\nAll artifacts exported. You can now start the API:")
print("  uvicorn main:app --reload --port 8000")
