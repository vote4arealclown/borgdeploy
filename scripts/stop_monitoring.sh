#!/usr/bin/env bash
# Stop Prometheus and Grafana.
set -euo pipefail

echo "==> Stopping Prometheus and Grafana"
sudo systemctl stop prometheus || true
sudo systemctl stop grafana-server || true
echo "==> Monitoring stopped"
