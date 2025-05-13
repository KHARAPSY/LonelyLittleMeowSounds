#!/bin/bash

# Exit on any error
set -e

# Default ports
API_PORT="${API_PORT:-8001}"
DOC_PORT="${DOC_PORT:-8002}"
LOG_DIR="/var/log/app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/startup_$TIMESTAMP.log"

# Ensure log directory exists
sudo mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Install dependencies
log "Installing Python dependencies..."
pip3 install --no-cache-dir --default-timeout=100 -r requirements.txt &>> "$LOG_FILE"

# Check required commands
for cmd in nginx pip3 uvicorn sphinx-quickstart; do
    command -v "$cmd" >/dev/null || { log "ERROR: $cmd not installed"; exit 1; }
done

# Start Nginx
log "Starting Nginx..."
sudo nginx &>> "$LOG_FILE"

# Start FastAPI app
log "Starting FastAPI application..."
cd "$HOME/LLMS/api"
uvicorn main_dev:app --host 0.0.0.0 --port "$API_PORT" --reload --log-level info &>> "$LOG_FILE" &
UVICORN_PID=$!

# Wait for background processes
wait $UVICORN_PID
