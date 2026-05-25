"""
Robot Sensor Anomaly Detector — Phase 1
Simulates time-series sensor data (temperature, vibration, voltage)
and injects labeled anomaly windows for model training & evaluation.

Usage:
    python sensor_simulator.py
    -> outputs: sensor_data.csv, anomaly_labels.csv, sensor_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

SEED        = 42
N_SAMPLES   = 2000          # total timesteps
SAMPLE_RATE = 1.0           # seconds between readings
OUTPUT_DIR  = Path(".")


# ── Normal signal generators ──────────────────────────────────────────────────

def normal_temperature(n: int, rng: np.random.Generator) -> np.ndarray:
    """Slowly drifting temp with small Gaussian noise (~65–75 °C)."""
    base  = 70.0
    trend = np.linspace(0, 2, n)                          # gradual warmup
    noise = rng.normal(0, 0.4, n)
    wave  = 1.5 * np.sin(np.linspace(0, 4 * np.pi, n))   # thermal cycle
    return base + trend + wave + noise


def normal_vibration(n: int, rng: np.random.Generator) -> np.ndarray:
    """High-freq vibration with low-amplitude noise (~0.8–1.2 g)."""
    base  = 1.0
    noise = rng.normal(0, 0.05, n)
    wave  = 0.1 * np.sin(np.linspace(0, 20 * np.pi, n))
    return base + wave + noise


def normal_voltage(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stable voltage with tiny fluctuations (~23.8–24.2 V)."""
    base  = 24.0
    noise = rng.normal(0, 0.08, n)
    return base + noise


# ── Anomaly injectors ─────────────────────────────────────────────────────────

def inject_spike(signal: np.ndarray, start: int, length: int,
                 magnitude: float) -> np.ndarray:
    """Sudden sharp spike — e.g. electrical surge or mechanical impact."""
    s = signal.copy()
    ramp  = np.linspace(0, magnitude, length // 2)
    decay = np.linspace(magnitude, 0, length - length // 2)
    s[start : start + length] += np.concatenate([ramp, decay])
    return s


def inject_flatline(signal: np.ndarray, start: int, length: int) -> np.ndarray:
    """Sensor flatlines at its last value — sensor failure / disconnect."""
    s = signal.copy()
    s[start : start + length] = s[start]                  # freeze last reading
    return s


def inject_drift(signal: np.ndarray, start: int, length: int,
                 drift_rate: float) -> np.ndarray:
    """Slow increasing drift — wear, thermal runaway, gradual fault."""
    s = signal.copy()
    drift = np.linspace(0, drift_rate * length, length)
    s[start : start + length] += drift
    return s


def inject_noise_burst(signal: np.ndarray, start: int, length: int,
                       scale: float, rng: np.random.Generator) -> np.ndarray:
    """High-frequency noise burst — loose connection or EMI interference."""
    s = signal.copy()
    s[start : start + length] += rng.normal(0, scale, length)
    return s


# ── Main simulation ───────────────────────────────────────────────────────────

def simulate(n: int = N_SAMPLES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    t   = np.arange(n) * SAMPLE_RATE    # time axis in seconds

    # 1. Generate clean baseline signals
    temp  = normal_temperature(n, rng)
    vib   = normal_vibration(n, rng)
    volt  = normal_voltage(n, rng)

    # 2. Define anomaly windows  (start_idx, length, type, sensor, params)
    anomaly_defs = [
        # Temperature spike — overheating event
        dict(start=200,  length=30,  kind="spike",       sensor="temperature",
             params=dict(magnitude=12.0)),
        # Vibration flatline — sensor dropout
        dict(start=450,  length=50,  kind="flatline",    sensor="vibration",
             params={}),
        # Voltage drift — power supply degradation
        dict(start=700,  length=80,  kind="drift",       sensor="voltage",
             params=dict(drift_rate=0.05)),
        # Vibration noise burst — mechanical looseness
        dict(start=950,  length=40,  kind="noise_burst", sensor="vibration",
             params=dict(scale=0.6)),
        # Temperature drift — cooling system failure
        dict(start=1200, length=100, kind="drift",       sensor="temperature",
             params=dict(drift_rate=0.08)),
        # Voltage spike — transient surge
        dict(start=1500, length=20,  kind="spike",       sensor="voltage",
             params=dict(magnitude=3.5)),
        # Simultaneous vibration + temperature spike — severe fault
        dict(start=1750, length=35,  kind="spike",       sensor="vibration",
             params=dict(magnitude=1.8)),
        dict(start=1750, length=35,  kind="spike",       sensor="temperature",
             params=dict(magnitude=8.0)),
    ]

    # 3. Inject anomalies and build label array
    labels = np.zeros(n, dtype=int)   # 0 = normal, 1 = anomaly

    for anom in anomaly_defs:
        s, l, kind = anom["start"], anom["length"], anom["kind"]
        p = anom["params"]

        if anom["sensor"] == "temperature":
            if kind == "spike":       temp = inject_spike(temp, s, l, **p)
            elif kind == "drift":     temp = inject_drift(temp, s, l, **p)
            elif kind == "flatline":  temp = inject_flatline(temp, s, l)
            elif kind == "noise_burst": temp = inject_noise_burst(temp, s, l, rng=rng, **p)

        elif anom["sensor"] == "vibration":
            if kind == "spike":       vib = inject_spike(vib, s, l, **p)
            elif kind == "drift":     vib = inject_drift(vib, s, l, **p)
            elif kind == "flatline":  vib = inject_flatline(vib, s, l)
            elif kind == "noise_burst": vib = inject_noise_burst(vib, s, l, rng=rng, **p)

        elif anom["sensor"] == "voltage":
            if kind == "spike":       volt = inject_spike(volt, s, l, **p)
            elif kind == "drift":     volt = inject_drift(volt, s, l, **p)
            elif kind == "flatline":  volt = inject_flatline(volt, s, l)
            elif kind == "noise_burst": volt = inject_noise_burst(volt, s, l, rng=rng, **p)

        labels[s : s + l] = 1   # mark window as anomalous

    return t, temp, vib, volt, labels, anomaly_defs


# ── Save to CSV ───────────────────────────────────────────────────────────────

def save_csvs(t, temp, vib, volt, labels):
    df = pd.DataFrame({
        "timestamp":   t,
        "temperature": temp.round(4),
        "vibration":   vib.round(4),
        "voltage":     volt.round(4),
        "is_anomaly":  labels,
    })
    df.to_csv(OUTPUT_DIR / "sensor_data.csv", index=False)

    # Separate label file — useful for evaluation later
    label_df = df[["timestamp", "is_anomaly"]]
    label_df.to_csv(OUTPUT_DIR / "anomaly_labels.csv", index=False)

    print(f"[✓] sensor_data.csv    — {len(df):,} rows")
    print(f"[✓] anomaly_labels.csv — {labels.sum():,} anomaly points "
          f"({labels.mean()*100:.1f}%)")
    return df


# ── Visualise ─────────────────────────────────────────────────────────────────

def plot_signals(df: pd.DataFrame, anomaly_defs: list):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Robot Sensor Simulation — Injected Anomalies",
                 fontsize=13, fontweight="bold", y=0.98)

    sensors = [
        ("temperature", "Temperature (°C)",  "#E85D24"),
        ("vibration",   "Vibration (g)",     "#378ADD"),
        ("voltage",     "Voltage (V)",       "#1D9E75"),
    ]

    for ax, (col, ylabel, color) in zip(axes, sensors):
        ax.plot(df["timestamp"], df[col], color=color,
                linewidth=0.8, alpha=0.9, label=col)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # Shade anomaly windows for this sensor
        for anom in anomaly_defs:
            if anom["sensor"] == col:
                s, l = anom["start"], anom["length"]
                ax.axvspan(s, s + l, color="#E24B4A", alpha=0.18, linewidth=0)
                ax.axvline(s, color="#E24B4A", linewidth=0.7,
                           linestyle="--", alpha=0.5)
                mid = s + l / 2
                y_top = ax.get_ylim()[1] if ax.get_ylim()[1] else 1
                label_y = ax.get_ylim()[0] + (ax.get_ylim()[1] -
                          ax.get_ylim()[0]) * 0.88
                ax.text(mid, label_y, anom["kind"],
                        ha="center", fontsize=7.5, color="#A32D2D",
                        fontweight="bold")

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)

    normal_patch = mpatches.Patch(color="#888780", alpha=0.6, label="Normal")
    anom_patch   = mpatches.Patch(color="#E24B4A", alpha=0.4, label="Anomaly window")
    axes[0].legend(handles=[normal_patch, anom_patch],
                   loc="upper right", fontsize=9, framealpha=0.7)

    plt.tight_layout()
    out = OUTPUT_DIR / "sensor_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] sensor_plot.png    — saved")


# ── Stats summary ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n── Signal statistics (normal windows only) ──")
    normal = df[df["is_anomaly"] == 0]
    for col in ["temperature", "vibration", "voltage"]:
        print(f"  {col:12s}  mean={normal[col].mean():.3f}  "
              f"std={normal[col].std():.3f}  "
              f"min={normal[col].min():.3f}  max={normal[col].max():.3f}")

    print("\n── Anomaly window breakdown ──")
    anom_only = df[df["is_anomaly"] == 1]
    for col in ["temperature", "vibration", "voltage"]:
        if len(anom_only):
            delta = anom_only[col].mean() - normal[col].mean()
            print(f"  {col:12s}  anomaly mean delta = {delta:+.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Simulating robot sensor data...\n")
    t, temp, vib, volt, labels, anomaly_defs = simulate()
    df = save_csvs(t, temp, vib, volt, labels)
    plot_signals(df, anomaly_defs)
    print_summary(df)
    print("\nDone. Next: run sensor_simulator.py, then feed sensor_data.csv into your model.")