#!/bin/bash
# deploy.sh
# Run this on the EC2 instance after SSHing in.
# Builds the Docker image, starts the container, and confirms it's healthy.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="anomaly-detector:latest"

echo "╔══════════════════════════════════════════════╗"
echo "║   Robot Sensor Anomaly Detector — Deploy     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$APP_DIR"

# ── Verify required files ─────────────────────────────────────────────────────
REQUIRED=(main.py Dockerfile docker-compose.yml requirements.txt \
          lstm_autoencoder.pt scaler_params.json threshold.json)
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[✗] Missing: $f"
    echo "    Run: scp -i your-key.pem $f ubuntu@<EC2-IP>:~/anomaly-detector/"
    exit 1
  fi
  echo "[✓] $f"
done
echo ""

# ── Create data dir for SQLite drift log ─────────────────────────────────────
mkdir -p data

# ── Build image ───────────────────────────────────────────────────────────────
echo "Building Docker image (first run: ~5 min for PyTorch download)..."
docker build -t "$IMAGE" .
echo "[✓] Image built: $IMAGE"
echo ""

# ── Stop any existing container ───────────────────────────────────────────────
docker compose down 2>/dev/null || true

# ── Start container ───────────────────────────────────────────────────────────
docker compose up -d
echo "[✓] Container started"
echo ""

# ── Wait for health check ─────────────────────────────────────────────────────
echo "Waiting for API to be ready..."
for i in {1..20}; do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[✓] API is healthy"
    break
  fi
  if [[ $i -eq 20 ]]; then
    echo "[✗] API did not start. Check logs: docker compose logs"
    exit 1
  fi
  sleep 3
  echo "  ... attempt $i/20"
done

echo ""
echo "══════════════════════════════════════════════"

# ── Print public IP ───────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<your-ec2-ip>")

echo ""
echo "  API live at:  http://$PUBLIC_IP:8000"
echo "  Swagger docs: http://$PUBLIC_IP:8000/docs"
echo "  Health:       http://$PUBLIC_IP:8000/health"
echo ""
echo "  Test it:"
echo "  curl -X POST http://$PUBLIC_IP:8000/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"readings\": $(python3 -c "import json; print(json.dumps([[70.5,1.01,24.0]]*30))")}'"
echo ""
echo "  View logs:    docker compose logs -f"
echo "  Stop:         docker compose down"
echo "══════════════════════════════════════════════"