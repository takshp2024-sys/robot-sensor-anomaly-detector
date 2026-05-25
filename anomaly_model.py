"""
Robot Sensor Anomaly Detector — Phase 2
LSTM Autoencoder + Isolation Forest baseline.

Loads sensor_data.csv from Phase 1, trains both models on normal data only,
then evaluates on all windows using the injected ground truth labels.

Usage:
    python anomaly_model.py
    -> outputs: model_results.png, model_comparison.csv, lstm_autoencoder.pt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

SENSORS      = ["temperature", "vibration", "voltage"]
SEQ_LEN      = 30          # timesteps per window fed into LSTM
STRIDE       = 1           # sliding window stride
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 1e-3
HIDDEN_DIM   = 64
LATENT_DIM   = 16          # bottleneck size
NUM_LAYERS   = 2
THRESHOLD_PERCENTILE = 95  # reconstruction error percentile → anomaly cutoff
DATA_PATH    = Path("sensor_data.csv")
OUTPUT_DIR   = Path(".")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading & preprocessing ──────────────────────────────────────────────

def load_and_scale(path: Path):
    df     = pd.read_csv(path)
    scaler = MinMaxScaler()
    X      = scaler.fit_transform(df[SENSORS].values)   # shape (N, 3)
    labels = df["is_anomaly"].values                     # ground truth
    return X, labels, scaler, df


def make_windows(X: np.ndarray, labels: np.ndarray, seq_len: int, stride: int):
    """Slide a window across the time series → (num_windows, seq_len, features)."""
    windows, win_labels = [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        windows.append(X[i : i + seq_len])
        # label a window as anomalous if ANY timestep inside it is anomalous
        win_labels.append(int(labels[i : i + seq_len].any()))
    return np.array(windows, dtype=np.float32), np.array(win_labels)


def split_normal(windows: np.ndarray, win_labels: np.ndarray):
    """Return only the normal windows for unsupervised training."""
    mask = win_labels == 0
    return windows[mask], windows[~mask]


# ── LSTM Autoencoder ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)   # h: (num_layers, batch, hidden)
        z = self.fc(h[-1])         # take last layer hidden state → latent vec
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.fc      = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.out     = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # repeat latent vector across time to seed decoder
        h0 = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h0)
        return self.out(out)          # (batch, seq_len, features)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, seq_len=SEQ_LEN, num_layers=NUM_LAYERS):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)

    def forward(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        return xhat


# ── Training ──────────────────────────────────────────────────────────────────

def train_autoencoder(normal_windows: np.ndarray):
    tensor    = torch.tensor(normal_windows).to(DEVICE)
    dataset   = TensorDataset(tensor)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = LSTMAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        train_losses.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg:.6f}")

    return model, train_losses


# ── Scoring ───────────────────────────────────────────────────────────────────

def reconstruction_errors(model: LSTMAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error per window."""
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            batch = torch.tensor(windows[i : i + BATCH_SIZE]).to(DEVICE)
            recon = model(batch)
            mse   = ((batch - recon) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def threshold_classify(errors: np.ndarray, normal_errors: np.ndarray,
                       percentile: int = THRESHOLD_PERCENTILE) -> np.ndarray:
    """Classify windows as anomalous if error > Nth percentile of normal errors."""
    cutoff = np.percentile(normal_errors, percentile)
    print(f"  LSTM threshold @ p{percentile}: {cutoff:.6f}")
    return (errors > cutoff).astype(int), cutoff


# ── Isolation Forest baseline ─────────────────────────────────────────────────

def run_isolation_forest(windows: np.ndarray, win_labels: np.ndarray):
    """Flatten windows → (n, seq_len*features), fit on normal, predict all."""
    flat        = windows.reshape(len(windows), -1)
    normal_flat = flat[win_labels == 0]

    clf = IsolationForest(n_estimators=200, contamination=0.10,
                          random_state=SEED if 'SEED' in dir() else 42)
    clf.fit(normal_flat)

    # IsolationForest: -1 = anomaly, 1 = normal → remap to 0/1
    preds = (clf.predict(flat) == -1).astype(int)
    scores = -clf.score_samples(flat)   # higher = more anomalous
    return preds, scores


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, preds: np.ndarray,
             labels: np.ndarray, scores: np.ndarray = None):
    p  = precision_score(labels, preds, zero_division=0)
    r  = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if scores is not None else float("nan")
    cm  = confusion_matrix(labels, preds)
    print(f"\n── {name} ──")
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"  ROC-AUC   : {auc:.3f}")
    print(f"  Confusion matrix:\n{cm}")
    return dict(model=name, precision=p, recall=r, f1=f1, roc_auc=auc)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(df_raw, win_labels, lstm_errors, lstm_preds, lstm_threshold,
                 if_scores, if_preds, train_losses):

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Robot Sensor Anomaly Detector — Model Results",
                 fontsize=13, fontweight="bold")

    # --- Row 1: raw sensor signals ----------------------------------------
    axes_sensors = [fig.add_subplot(4, 3, i+1) for i in range(3)]
    colors = ["#E85D24", "#378ADD", "#1D9E75"]
    for ax, col, color in zip(axes_sensors, SENSORS, colors):
        ax.plot(df_raw["timestamp"], df_raw[col],
                color=color, linewidth=0.7, alpha=0.85)
        # shade anomaly regions
        anom_idx = np.where(df_raw["is_anomaly"].values == 1)[0]
        if len(anom_idx):
            for start, end in _contiguous_ranges(anom_idx):
                ax.axvspan(df_raw["timestamp"].iloc[start],
                           df_raw["timestamp"].iloc[end],
                           color="#E24B4A", alpha=0.15, linewidth=0)
        ax.set_title(col.capitalize(), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)

    # --- Row 2: training loss + LSTM error distribution -------------------
    ax_loss = fig.add_subplot(4, 3, 4)
    ax_loss.plot(train_losses, color="#378ADD", linewidth=1.5)
    ax_loss.set_title("LSTM training loss", fontsize=10)
    ax_loss.set_xlabel("Epoch", fontsize=8)
    ax_loss.tick_params(labelsize=8)
    ax_loss.spines[["top","right"]].set_visible(False)

    ax_hist = fig.add_subplot(4, 3, 5)
    normal_err  = lstm_errors[win_labels == 0]
    anomaly_err = lstm_errors[win_labels == 1]
    ax_hist.hist(normal_err,  bins=60, color="#378ADD", alpha=0.6, label="Normal")
    ax_hist.hist(anomaly_err, bins=60, color="#E24B4A", alpha=0.6, label="Anomaly")
    ax_hist.axvline(lstm_threshold, color="#BA7517", linewidth=1.5,
                    linestyle="--", label=f"Threshold")
    ax_hist.set_title("LSTM reconstruction error dist.", fontsize=10)
    ax_hist.legend(fontsize=7)
    ax_hist.tick_params(labelsize=8)
    ax_hist.spines[["top","right"]].set_visible(False)

    ax_if = fig.add_subplot(4, 3, 6)
    if_normal  = if_scores[win_labels == 0]
    if_anomaly = if_scores[win_labels == 1]
    ax_if.hist(if_normal,  bins=60, color="#1D9E75", alpha=0.6, label="Normal")
    ax_if.hist(if_anomaly, bins=60, color="#E24B4A", alpha=0.6, label="Anomaly")
    ax_if.set_title("Isolation Forest score dist.", fontsize=10)
    ax_if.legend(fontsize=7)
    ax_if.tick_params(labelsize=8)
    ax_if.spines[["top","right"]].set_visible(False)

    # --- Row 3 & 4: LSTM and IF anomaly scores over time ------------------
    t_win = np.arange(len(win_labels))  # window indices as time proxy

    for row, (name, scores, preds, color) in enumerate([
        ("LSTM reconstruction error",   lstm_errors, lstm_preds, "#378ADD"),
        ("Isolation Forest anomaly score", if_scores, if_preds,  "#1D9E75"),
    ]):
        ax = fig.add_subplot(4, 1, row + 3)
        ax.plot(t_win, scores, color=color, linewidth=0.7, alpha=0.8, label=name)

        # shade true anomaly windows
        anom_win = np.where(win_labels == 1)[0]
        for start, end in _contiguous_ranges(anom_win):
            ax.axvspan(start, end, color="#E24B4A", alpha=0.12, linewidth=0)

        # mark false positives and false negatives
        fp = np.where((preds == 1) & (win_labels == 0))[0]
        fn = np.where((preds == 0) & (win_labels == 1))[0]
        ax.scatter(fp, scores[fp], color="#BA7517", s=8, zorder=3,
                   label=f"False positive ({len(fp)})", alpha=0.6)
        ax.scatter(fn, scores[fn], color="#A32D2D", s=8, zorder=3,
                   label=f"False negative ({len(fn)})", alpha=0.6)

        if row == 0:
            ax.axhline(lstm_threshold, color="#BA7517",
                       linewidth=1, linestyle="--", alpha=0.8)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Window index", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "model_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] model_results.png saved")


def _contiguous_ranges(indices):
    """Convert an array of indices into (start, end) contiguous ranges."""
    if len(indices) == 0:
        return []
    ranges, start = [], indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            ranges.append((start, indices[i-1]))
            start = indices[i]
    ranges.append((start, indices[-1]))
    return ranges


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # 1. Load data
    print("Loading sensor_data.csv...")
    X, labels, scaler, df_raw = load_and_scale(DATA_PATH)
    windows, win_labels = make_windows(X, labels, SEQ_LEN, STRIDE)
    normal_wins, anomaly_wins = split_normal(windows, win_labels)

    print(f"  Total windows   : {len(windows):,}")
    print(f"  Normal windows  : {len(normal_wins):,}")
    print(f"  Anomaly windows : {len(anomaly_wins):,}")
    print(f"  Anomaly rate    : {win_labels.mean()*100:.1f}%\n")

    # 2. Train LSTM Autoencoder (on normal data only)
    print("Training LSTM Autoencoder...")
    model, train_losses = train_autoencoder(normal_wins)
    torch.save(model.state_dict(), OUTPUT_DIR / "lstm_autoencoder.pt")
    print(f"[✓] lstm_autoencoder.pt saved")

    # 3. Score all windows
    print("\nScoring all windows with LSTM...")
    all_errors   = reconstruction_errors(model, windows)
    normal_errors = reconstruction_errors(model, normal_wins)
    lstm_preds, threshold = threshold_classify(all_errors, normal_errors)

    # 4. Isolation Forest baseline
    print("\nTraining Isolation Forest baseline...")
    if_preds, if_scores = run_isolation_forest(windows, win_labels)

    # 5. Evaluate
    lstm_res = evaluate("LSTM Autoencoder", lstm_preds, win_labels, all_errors)
    if_res   = evaluate("Isolation Forest", if_preds,   win_labels, if_scores)

    # 6. Save comparison CSV
    results_df = pd.DataFrame([lstm_res, if_res]).round(4)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"\n[✓] model_comparison.csv saved")
    print(results_df.to_string(index=False))

    # 7. Plot
    plot_results(df_raw, win_labels, all_errors, lstm_preds, threshold,
                 if_scores, if_preds, train_losses)

    print("\n── Summary ─────────────────────────────────────────────")
    better = "LSTM Autoencoder" if lstm_res["f1"] >= if_res["f1"] else "Isolation Forest"
    print(f"  Best F1: {better}")
    print(f"  LSTM  F1={lstm_res['f1']:.3f}  AUC={lstm_res['roc_auc']:.3f}")
    print(f"  IF    F1={if_res['f1']:.3f}  AUC={if_res['roc_auc']:.3f}")
    print("\nPhase 2 complete. Next: wrap model in FastAPI (Phase 3).")


if __name__ == "__main__":
    main()