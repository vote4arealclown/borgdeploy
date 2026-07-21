#!/usr/bin/env bash
set -euo pipefail

# Borg one-shot Debian installer.
# Usage: ./setup.sh [--with-systemd] [--no-ollama] [--no-samba]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
INSTALL_SYSTEMD=false
INSTALL_OLLAMA=true
INSTALL_SAMBA=true
INSTALL_MONITORING=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-systemd) INSTALL_SYSTEMD=true ; shift ;;
    --with-monitoring) INSTALL_MONITORING=true ; shift ;;
    --no-ollama) INSTALL_OLLAMA=false ; shift ;;
    --no-samba) INSTALL_SAMBA=false ; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "==> Updating packages"
sudo apt-get update

echo "==> Installing base packages"
sudo apt-get install -y python3 python3-venv python3-pip build-essential curl gnupg2 lsb-release

if [[ "$INSTALL_SAMBA" == true ]]; then
  echo "==> Installing Samba"
  sudo apt-get install -y samba samba-common-bin
  sudo mkdir -p /borg/input /borg/output
  sudo chmod 777 /borg/input /borg/output
  # Add guest share if not present
  if ! grep -q "\[borg-input\]" /etc/samba/smb.conf 2>/dev/null; then
    sudo tee -a /etc/samba/smb.conf >/dev/null <<'EOF'

[borg-input]
   path = /borg/input
   browsable = yes
   read only = no
   guest ok = yes

[borg-output]
   path = /borg/output
   browsable = yes
   read only = yes
   guest ok = yes
EOF
    sudo systemctl restart smbd || true
  fi
fi

echo "==> Installing PostgreSQL 17 + pgvector"
# Try distribution packages first; fallback to PGDG repo on failure
if ! sudo apt-get install -y postgresql-17 postgresql-17-pgvector 2>/dev/null; then
  echo "==> Adding PostgreSQL APT repository"
  sudo install -d /usr/share/postgresql-common/pgdg
  sudo curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
    https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | \
    sudo tee /etc/apt/sources.list.d/pgdg.list
  sudo apt-get update
  sudo apt-get install -y postgresql-17 postgresql-17-pgvector
fi

sudo systemctl enable postgresql
sudo systemctl start postgresql

# Create borg database/user
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='borg'" | grep -q 1 || \
  sudo -u postgres createuser -s borg
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='borg'" | grep -q 1 || \
  sudo -u postgres createdb -O borg borg

# Set borg password
sudo -u postgres psql -c "ALTER USER borg WITH PASSWORD 'borg';" || true

# Run schema
PGPASSWORD=borg psql -U borg -d borg -h localhost -f "$PROJECT_ROOT/db/schema.sql" || true

echo "==> Creating Python virtual environment"
python3 -m venv "$PROJECT_ROOT/.venv"
"$PROJECT_ROOT/.venv/bin/pip" install --upgrade pip
"$PROJECT_ROOT/.venv/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"

if [[ "$INSTALL_OLLAMA" == true ]]; then
  echo "==> Installing Ollama"
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  sudo systemctl enable ollama || true
  sudo systemctl start ollama || true
  sleep 2
  echo "==> Pulling TinyLlama model (~640 MB)"
  ollama pull tinyllama:latest || true
  echo "==> Pulling nomic-embed-text model"
  ollama pull nomic-embed-text:latest || true
fi

echo "==> Creating local input/output dirs"
mkdir -p "$PROJECT_ROOT/input" "$PROJECT_ROOT/output" "$PROJECT_ROOT/data"

# Create default .env if missing
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "==> Writing sample .env"
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  echo "BORG_PASSWORD=borg" >> "$PROJECT_ROOT/.env"
fi

if [[ "$INSTALL_MONITORING" == true ]]; then
  echo "==> Installing Prometheus + Grafana monitoring"
  bash "$PROJECT_ROOT/scripts/install_monitoring.sh"
  sudo cp "$PROJECT_ROOT/scripts/borg-prometheus.service" /etc/systemd/system/borg-prometheus.service || true
  sudo systemctl daemon-reload
  sudo systemctl enable prometheus || true
  sudo systemctl start prometheus || true
  sudo systemctl enable grafana-server || true
  sudo systemctl start grafana-server || true
fi

if [[ "$INSTALL_SYSTEMD" == true ]]; then
  echo "==> Installing systemd service"
  sudo cp "$PROJECT_ROOT/scripts/borg.service" /etc/systemd/system/borg.service
  sudo sed -i "s|/home/theone/BorgDeploy|$PROJECT_ROOT|g" /etc/systemd/system/borg.service
  sudo sed -i "s|User=theone|User=$USER|g" /etc/systemd/system/borg.service
  sudo systemctl daemon-reload
  sudo systemctl enable borg
  sudo systemctl start borg || true
fi

HOST_IP=$(hostname -I | awk '{print $1}')
echo "==> Setup complete. Run:"
echo "    $PROJECT_ROOT/.venv/bin/python -m borg.main all"
echo "Then open http://$HOST_IP:8000"
if [[ "$INSTALL_MONITORING" == true ]]; then
  echo "Monitoring:"
  echo "    Prometheus UI: http://$HOST_IP:9090"
  echo "    Grafana UI:    http://$HOST_IP:3000  (default admin/admin)"
fi
