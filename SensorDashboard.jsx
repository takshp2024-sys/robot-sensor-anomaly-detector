/**
 * Robot Sensor Anomaly Detector — Phase 4
 * Live dashboard: real-time sensor charts, alert feed, model health panel.
 *
 * Setup:
 *   npm install recharts lucide-react
 *
 * Connects to the FastAPI backend from Phase 3 at BACKEND_URL.
 * While the backend is offline, runs in DEMO MODE with simulated data.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Activity, AlertTriangle, CheckCircle, Cpu,
  Zap, Thermometer, Radio, RefreshCw, Circle,
} from "lucide-react";

// ── Config ────────────────────────────────────────────────────────────────────

const BACKEND_URL   = "http://localhost:8000";
const POLL_MS       = 2000;     // how often to poll /health + /history
const MAX_CHART_PTS = 60;       // points kept in each sensor chart
const SEQ_LEN       = 30;

// ── Demo-mode simulator (runs when backend is offline) ───────────────────────

let _demoT = 0;
let _anomalyBurst = 0;

function nextDemoReading() {
  _demoT++;
  const isAnomaly = _anomalyBurst > 0;
  if (!isAnomaly && Math.random() < 0.04) _anomalyBurst = 12 + Math.floor(Math.random() * 10);
  if (_anomalyBurst > 0) _anomalyBurst--;

  const spike = isAnomaly ? 8 + Math.random() * 6 : 0;
  return {
    t:           _demoT,
    temperature: +(70 + 1.5 * Math.sin(_demoT / 15) + (Math.random() - 0.5) * 0.8 + spike).toFixed(3),
    vibration:   +(1.0 + 0.1 * Math.sin(_demoT / 8)  + (Math.random() - 0.5) * 0.05 + (isAnomaly ? 0.9 : 0)).toFixed(4),
    voltage:     +(24 + (Math.random() - 0.5) * 0.15 + (isAnomaly ? -1.2 : 0)).toFixed(3),
    recon_error: isAnomaly ? +(0.0015 + Math.random() * 0.012).toFixed(8) : +(0.000050 + Math.random() * 0.00045).toFixed(8),
    is_anomaly:  isAnomaly,
    severity:    isAnomaly ? (Math.random() > 0.5 ? "high" : "medium") : "normal",
  };
}

function demoHealth() {
  return {
    status: "healthy",
    total_predictions: 1200 + _demoT,
    anomaly_rate_pct: 5.3,
    avg_latency_ms: 3.8,
    p95_latency_ms: 7.2,
    avg_recon_error: 0.00031,
    drift_score: 1.04,
    threshold: 0.000713,
    device: "cpu (demo)",
  };
}

// ── Severity helpers ──────────────────────────────────────────────────────────

const SEV = {
  normal:   { color: "#22c55e", bg: "#052e16",  label: "NORMAL",   icon: CheckCircle },
  low:      { color: "#84cc16", bg: "#1a2e05",  label: "LOW",      icon: Activity    },
  medium:   { color: "#f59e0b", bg: "#2d1b00",  label: "MEDIUM",   icon: AlertTriangle },
  high:     { color: "#f97316", bg: "#2d1000",  label: "HIGH",     icon: AlertTriangle },
  critical: { color: "#ef4444", bg: "#2d0000",  label: "CRITICAL", icon: AlertTriangle },
};

// ── Custom tooltip ────────────────────────────────────────────────────────────

function SensorTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const v = payload[0];
  return (
    <div style={{
      background: "#0f1117", border: "1px solid #1e2330",
      borderRadius: 6, padding: "8px 12px", fontSize: 12,
    }}>
      <div style={{ color: "#6b7280", marginBottom: 4 }}>t = {label}</div>
      <div style={{ color: v.color, fontWeight: 600, fontFamily: "monospace" }}>
        {v.value}
      </div>
    </div>
  );
}

// ── Sensor chart panel ────────────────────────────────────────────────────────

function SensorChart({ label, unit, dataKey, color, data, threshold }) {
  const Icon = dataKey === "temperature" ? Thermometer : dataKey === "vibration" ? Radio : Zap;
  const latest = data[data.length - 1]?.[dataKey];
  const isAnom = data[data.length - 1]?.is_anomaly;

  return (
    <div style={{
      background: "#0a0c10", border: `1px solid ${isAnom ? color + "55" : "#1a1f2e"}`,
      borderRadius: 12, padding: "16px 20px",
      transition: "border-color 0.4s",
      boxShadow: isAnom ? `0 0 20px ${color}18` : "none",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Icon size={14} color={color} />
          <span style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.12em", color: "#6b7280", textTransform: "uppercase" }}>
            {label}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
          <span style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace", color: isAnom ? "#ef4444" : "#e2e8f0", transition: "color 0.3s" }}>
            {latest?.toFixed(2) ?? "—"}
          </span>
          <span style={{ fontSize: 11, color: "#4b5563" }}>{unit}</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={90}>
        <LineChart data={data} margin={{ top: 4, right: 0, left: -32, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#0f1117" vertical={false} />
          <XAxis dataKey="t" hide />
          <YAxis tick={{ fontSize: 9, fill: "#374151" }} tickLine={false} axisLine={false} />
          {threshold && (
            <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="4 2" strokeOpacity={0.4} />
          )}
          <Tooltip content={<SensorTooltip />} />
          <Line
            type="monotone" dataKey={dataKey}
            stroke={color} strokeWidth={1.5}
            dot={false} isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Alert badge ───────────────────────────────────────────────────────────────

function AlertBadge({ severity }) {
  const s = SEV[severity] ?? SEV.normal;
  return (
    <span style={{
      fontSize: 10, fontFamily: "monospace", fontWeight: 700,
      letterSpacing: "0.1em", padding: "2px 7px", borderRadius: 4,
      background: s.bg, color: s.color, border: `1px solid ${s.color}44`,
    }}>
      {s.label}
    </span>
  );
}

// ── Alert feed row ────────────────────────────────────────────────────────────

function AlertRow({ item, fresh }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "8px 0", borderBottom: "1px solid #0f1117",
      animation: fresh ? "fadeSlide 0.3s ease" : "none",
    }}>
      <Circle
        size={6} fill={item.is_anomaly ? "#ef4444" : "#22c55e"}
        color={item.is_anomaly ? "#ef4444" : "#22c55e"}
        style={{ flexShrink: 0 }}
      />
      <span style={{ fontSize: 11, color: "#374151", fontFamily: "monospace", flexShrink: 0, minWidth: 56 }}>
        t={item.t}
      </span>
      <AlertBadge severity={item.severity} />
      <span style={{ fontSize: 11, color: "#4b5563", fontFamily: "monospace", marginLeft: "auto" }}>
        err: {item.recon_error?.toFixed?.(6) ?? "—"}
      </span>
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: "#0a0c10", border: "1px solid #1a1f2e", borderRadius: 10, padding: "14px 16px" }}>
      <div style={{ fontSize: 10, fontFamily: "monospace", color: "#4b5563", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "monospace", color: accent ?? "#e2e8f0" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 10, color: "#374151", marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

// ── Reconstruction error sparkline ────────────────────────────────────────────

function ErrorSparkline({ data, threshold }) {
  return (
    <div style={{ background: "#0a0c10", border: "1px solid #1a1f2e", borderRadius: 12, padding: "16px 20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
        <span style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.1em", color: "#6b7280", textTransform: "uppercase" }}>
          Reconstruction Error
        </span>
        <span style={{ fontSize: 10, color: "#374151", fontFamily: "monospace" }}>
          threshold: {threshold?.toFixed(6) ?? "—"}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={70}>
        <LineChart data={data} margin={{ top: 4, right: 0, left: -32, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#0f1117" vertical={false} />
          <XAxis dataKey="t" hide />
          <YAxis tick={{ fontSize: 9, fill: "#374151" }} tickLine={false} axisLine={false} />
          <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="4 2" strokeOpacity={0.5} />
          <Tooltip content={<SensorTooltip />} />
          <Line
            type="monotone" dataKey="recon_error"
            stroke="#a78bfa" strokeWidth={1.5}
            dot={false} isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────────────

export default function SensorDashboard() {
  const [chartData,   setChartData]   = useState([]);
  const [alerts,      setAlerts]      = useState([]);
  const [health,      setHealth]      = useState(null);
  const [connected,   setConnected]   = useState(false);
  const [freshAlert,  setFreshAlert]  = useState(false);
  const [tickCount,   setTickCount]   = useState(0);
  const tickRef = useRef(0);

  // ── Poll / simulate ───────────────────────────────────────────────────────

  const tick = useCallback(async () => {
    tickRef.current++;
    setTickCount(tickRef.current);

    let reading, healthData;

    try {
      // Try live backend first
      const [predRes, healthRes] = await Promise.all([
        fetch(`${BACKEND_URL}/history?limit=1`),
        fetch(`${BACKEND_URL}/health`),
      ]);
      const [histArr, h] = await Promise.all([predRes.json(), healthRes.json()]);
      const last = histArr[0];
      reading = {
        t:           tickRef.current,
        temperature: last?.temp_mean ?? 70,
        vibration:   last?.vib_mean  ?? 1.0,
        voltage:     last?.volt_mean ?? 24.0,
        recon_error: last?.recon_error ?? 0,
        is_anomaly:  last?.is_anomaly ?? false,
        severity:    last?.is_anomaly
                       ? (last.recon_error > h.threshold * 4 ? "critical"
                          : last.recon_error > h.threshold * 2 ? "high"
                          : "medium")
                       : "normal",
      };
      healthData = h;
      setConnected(true);
    } catch {
      // Demo mode
      reading    = nextDemoReading();
      healthData = demoHealth();
      setConnected(false);
    }

    setChartData(prev => {
      const next = [...prev, reading];
      return next.length > MAX_CHART_PTS ? next.slice(-MAX_CHART_PTS) : next;
    });

    if (reading.is_anomaly) {
      setFreshAlert(true);
      setAlerts(prev => [reading, ...prev].slice(0, 50));
      setTimeout(() => setFreshAlert(false), 600);
    }

    setHealth(healthData);
  }, []);

  useEffect(() => {
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => clearInterval(id);
  }, [tick]);

  // ── Derived ───────────────────────────────────────────────────────────────

  const latestAnomaly   = alerts[0];
  const anomalyCount    = alerts.length;
  const driftOk         = !health || health.drift_score <= 1.5;
  const statusColor     = health?.status === "healthy" ? "#22c55e" : "#ef4444";

  return (
    <div style={{
      minHeight: "100vh", background: "#060810",
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      color: "#e2e8f0", padding: "0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
        @keyframes fadeSlide { from { opacity:0; transform:translateY(-6px); } to { opacity:1; transform:none; } }
        @keyframes pulse     { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes spin      { to{transform:rotate(360deg)} }
        ::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:#060810; }
        ::-webkit-scrollbar-thumb { background:#1a1f2e; border-radius:2px; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom: "1px solid #0f1117", padding: "16px 28px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, zIndex: 10, background: "#060810",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg,#1d4ed8,#6d28d9)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Cpu size={16} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.06em", color: "#f1f5f9" }}>
              ROBOT SENSOR MONITOR
            </div>
            <div style={{ fontSize: 10, color: "#374151", letterSpacing: "0.08em" }}>
              ANOMALY DETECTION v1.0
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          {/* Connection badge */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: connected ? "#22c55e" : "#f59e0b" }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: connected ? "#22c55e" : "#f59e0b",
              animation: "pulse 1.5s infinite",
            }} />
            {connected ? "LIVE" : "DEMO MODE"}
          </div>
          {/* Model status */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: statusColor }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: statusColor }} />
            {health?.status?.toUpperCase() ?? "—"}
          </div>
          {/* Tick counter */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "#374151" }}>
            <RefreshCw size={10} style={{ animation: "spin 2s linear infinite" }} />
            {tickCount}
          </div>
        </div>
      </div>

      <div style={{ padding: "20px 28px", maxWidth: 1200, margin: "0 auto" }}>

        {/* Health stat row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))", gap: 10, marginBottom: 20 }}>
          <StatCard
            label="Total Preds" value={health?.total_predictions?.toLocaleString() ?? "—"}
            sub={`device: ${health?.device ?? "—"}`}
          />
          <StatCard
            label="Anomaly Rate" value={`${health?.anomaly_rate_pct?.toFixed(1) ?? "—"}%`}
            accent={health?.anomaly_rate_pct > 15 ? "#ef4444" : "#22c55e"}
          />
          <StatCard
            label="Avg Latency" value={`${health?.avg_latency_ms?.toFixed(1) ?? "—"}ms`}
            sub={`p95: ${health?.p95_latency_ms?.toFixed(1) ?? "—"}ms`}
          />
          <StatCard
            label="Drift Score" value={health?.drift_score?.toFixed(3) ?? "—"}
            accent={driftOk ? "#22c55e" : "#ef4444"}
            sub={driftOk ? "no drift detected" : "⚠ retrain needed"}
          />
          <StatCard
            label="Alerts (session)" value={anomalyCount}
            accent={anomalyCount > 0 ? "#f97316" : "#22c55e"}
          />
        </div>

        {/* Sensor charts + alert feed */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 16, marginBottom: 16 }}>

          {/* Left: sensor charts */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <SensorChart
              label="Temperature" unit="°C" dataKey="temperature"
              color="#f97316" data={chartData}
            />
            <SensorChart
              label="Vibration" unit="g" dataKey="vibration"
              color="#3b82f6" data={chartData}
            />
            <SensorChart
              label="Voltage" unit="V" dataKey="voltage"
              color="#22c55e" data={chartData}
            />
          </div>

          {/* Right: alert feed */}
          <div style={{
            background: "#0a0c10", border: `1px solid ${freshAlert ? "#ef444455" : "#1a1f2e"}`,
            borderRadius: 12, padding: "16px 18px", display: "flex",
            flexDirection: "column", transition: "border-color 0.3s",
            boxShadow: freshAlert ? "0 0 24px #ef444418" : "none",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.1em", color: "#6b7280", textTransform: "uppercase" }}>
                Alert Feed
              </span>
              <span style={{
                fontSize: 10, fontFamily: "monospace", padding: "2px 8px",
                borderRadius: 4, background: anomalyCount > 0 ? "#2d0000" : "#052e16",
                color: anomalyCount > 0 ? "#ef4444" : "#22c55e",
                border: `1px solid ${anomalyCount > 0 ? "#ef444433" : "#22c55e33"}`,
              }}>
                {anomalyCount} alerts
              </span>
            </div>

            {alerts.length === 0 ? (
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <div style={{ textAlign: "center" }}>
                  <CheckCircle size={22} color="#1a2e1a" style={{ margin: "0 auto 8px" }} />
                  <div style={{ fontSize: 11, color: "#1f2937" }}>No anomalies detected</div>
                </div>
              </div>
            ) : (
              <div style={{ overflowY: "auto", flex: 1, maxHeight: 340 }}>
                {alerts.map((a, i) => (
                  <AlertRow key={`${a.t}-${i}`} item={a} fresh={i === 0 && freshAlert} />
                ))}
              </div>
            )}

            {/* Latest anomaly detail */}
            {latestAnomaly && (
              <div style={{
                marginTop: 14, padding: "10px 12px", borderRadius: 8,
                background: "#0f1117", border: "1px solid #1a1f2e",
              }}>
                <div style={{ fontSize: 10, color: "#374151", marginBottom: 6 }}>LAST ANOMALY</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  {[
                    ["TEMP",  `${latestAnomaly.temperature?.toFixed(2)}°C`],
                    ["VIB",   `${latestAnomaly.vibration?.toFixed(3)}g`],
                    ["VOLT",  `${latestAnomaly.voltage?.toFixed(2)}V`],
                    ["ERR",   latestAnomaly.recon_error?.toFixed(6)],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <div style={{ fontSize: 9, color: "#374151", letterSpacing: "0.1em" }}>{k}</div>
                      <div style={{ fontSize: 12, color: "#ef4444", fontFamily: "monospace", fontWeight: 600 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Reconstruction error chart */}
        <ErrorSparkline data={chartData} threshold={health?.threshold ?? 0.000713} />

        {/* Footer */}
        <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid #0f1117", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "#1f2937", fontFamily: "monospace" }}>
            LSTM AUTOENCODER · SEQ_LEN={SEQ_LEN} · THRESHOLD={health?.threshold?.toFixed(6) ?? "0.000713"}
          </span>
          <span style={{ fontSize: 10, color: "#1f2937", fontFamily: "monospace" }}>
            {connected ? `↑ ${BACKEND_URL}` : "DEMO MODE — start FastAPI to go live"}
          </span>
        </div>
      </div>
    </div>
  );
}