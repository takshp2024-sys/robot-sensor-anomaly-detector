#!/bin/bash
# ec2_bootstrap.sh
# Paste this as EC2 User Data when launching the instance (runs once on first boot).
# Installs Docker + Docker Compose, then starts the anomaly detector API.

set -euo pipefail
exec > /var/log/bootstrap.log 2>&1   # all output logged here

echo "=== Bootstrap started: $(date) ==="

# ── 1. System update ──────────────────────────────────────────────────────────
apt-get update -y
apt-get upgrade -y

# ── 2. Install Docker ─────────────────────────────────────────────────────────
apt-get install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ── 3. Allow ubuntu user to run Docker without sudo ───────────────────────────
usermod -aG docker ubuntu

# ── 4. Start Docker on boot ───────────────────────────────────────────────────
systemctl enable docker
systemctl start docker

# ── 5. Create app directory ───────────────────────────────────────────────────
mkdir -p /home/ubuntu/anomaly-detector/data
chown -R ubuntu:ubuntu /home/ubuntu/anomaly-detector

echo "=== Bootstrap complete: $(date) ==="
echo "SSH in and run: cd /home/ubuntu/anomaly-detector && ./deploy.sh"