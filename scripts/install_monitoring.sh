#!/usr/bin/env bash
# Install Prometheus and Grafana for bare-metal Borg monitoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

install_prometheus() {
  if command -v prometheus >/dev/null 2>&1; then
    echo "Prometheus already installed: $(prometheus --version 2>&1 | head -1)"
    return
  fi

  echo "==> Installing Prometheus"
  sudo apt-get update
  sudo apt-get install -y prometheus

  sudo systemctl enable prometheus || true
}

install_grafana() {
  if command -v grafana-server >/dev/null 2>&1; then
    echo "Grafana already installed: $(grafana-server --version 2>&1 | head -1)"
    return
  fi

  echo "==> Installing Grafana"
  sudo apt-get install -y apt-transport-https wget
  sudo mkdir -p /etc/apt/keyrings
  wget -q -O - https://apt.grafana.com/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
  echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | \
    sudo tee /etc/apt/sources.list.d/grafana.list
  sudo apt-get update
  sudo apt-get install -y grafana

  sudo systemctl enable grafana-server || true
}

configure_prometheus() {
  echo "==> Configuring Prometheus to scrape Borg"
  sudo mkdir -p /etc/prometheus
  sudo cp "$PROJECT_ROOT/monitoring/prometheus.yml" /etc/prometheus/prometheus.yml
  sudo sed -i "s|localhost:8000|$(hostname -I | awk '{print $1}'):8000|g" /etc/prometheus/prometheus.yml || true
}

configure_grafana() {
  echo "==> Configuring Grafana provisioning"
  sudo mkdir -p /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards
  sudo cp "$PROJECT_ROOT/monitoring/grafana/datasources/datasource.yml" /etc/grafana/provisioning/datasources/borg.yml
  sudo cp "$PROJECT_ROOT/monitoring/grafana/dashboards/dashboard.yml" /etc/grafana/provisioning/dashboards/borg.yml
  sudo cp "$PROJECT_ROOT/monitoring/grafana/dashboards/borg-overview.json" /etc/grafana/provisioning/dashboards/borg-overview.json

  # Bare-metal: Grafana and Prometheus run on the same host, so point the
  # datasource at localhost. (Docker Compose keeps the prometheus:9090 alias.)
  sudo sed -i 's|http://prometheus:9090|http://localhost:9090|g' /etc/grafana/provisioning/datasources/borg.yml || true

  # Bind Grafana to all interfaces so it's reachable on the LAN.
  sudo sed -i 's|^;http_addr =.*|http_addr =|' /etc/grafana/grafana.ini || true
}

import_dashboard() {
  echo "==> Importing Borg dashboard into Grafana"
  local max_wait=30
  local waited=0
  while ! curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login | grep -q "200\|302"; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$max_wait" ]; then
      echo "WARNING: Grafana did not become ready in time; dashboard may need manual import."
      return
    fi
  done

  # Reset admin password to a known default so the operator can log in immediately.
  sudo grafana cli --homepath /usr/share/grafana admin reset-admin-password borg >/dev/null 2>&1 || true

  # Provisioning usually loads the dashboard automatically, but fall back to the
  # API if it doesn't (some Grafana 13 builds ignore file provisioning on first
  # start).
  curl -s -X POST -u admin:borg \
    -H "Content-Type: application/json" \
    -d @/etc/grafana/provisioning/dashboards/borg-overview.json \
    http://localhost:3000/api/dashboards/db >/dev/null 2>&1 || true
}

install_prometheus
install_grafana
configure_prometheus
configure_grafana

echo "==> Starting monitoring services"
sudo systemctl restart prometheus grafana-server || true
sleep 3

import_dashboard

echo "==> Monitoring stack installed and started."
echo "    Prometheus UI: http://$(hostname -I | awk '{print $1}'):9090"
echo "    Grafana UI:    http://$(hostname -I | awk '{print $1}'):3000  (admin / borg)"
