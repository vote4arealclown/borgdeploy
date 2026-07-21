#!/usr/bin/env bash
# Start Prometheus and Grafana for bare-metal Borg monitoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

start_prometheus() {
  if command -v prometheus >/dev/null 2>&1; then
    echo "==> Starting Prometheus via systemctl"
    sudo systemctl start prometheus || true
  else
    echo "Prometheus not installed. Run scripts/install_monitoring.sh first."
    exit 1
  fi
}

start_grafana() {
  if command -v grafana-server >/dev/null 2>&1; then
    echo "==> Starting Grafana via systemctl"
    sudo systemctl start grafana-server || true
  else
    echo "Grafana not installed. Run scripts/install_monitoring.sh first."
    exit 1
  fi
}

import_dashboard_if_missing() {
  # Ensure the Borg dashboard is loaded. File provisioning sometimes fails on
  # the very first Grafana start, so fall back to a direct API import.
  if curl -s -u admin:borg http://localhost:3000/api/search >/dev/null 2>&1; then
    if ! curl -s -u admin:borg http://localhost:3000/api/search | grep -q "Borg System Overview"; then
      curl -s -X POST -u admin:borg \
        -H "Content-Type: application/json" \
        -d @/etc/grafana/provisioning/dashboards/borg-overview.json \
        http://localhost:3000/api/dashboards/db >/dev/null 2>&1 || true
    fi
  fi
}

start_prometheus
start_grafana
sleep 2
import_dashboard_if_missing

HOST_IP=$(hostname -I | awk '{print $1}')
echo "==> Monitoring started"
echo "    Prometheus UI: http://$HOST_IP:9090"
echo "    Grafana UI:    http://$HOST_IP:3000  (admin / borg)"
