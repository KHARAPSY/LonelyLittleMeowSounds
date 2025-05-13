#!/bin/bash

# Exit on any error
set -e

# Default ports
API_PORT="${API_PORT:-8001}"
LOG_DIR="/var/log/app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/startup_$TIMESTAMP.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start Nginx
log "Starting Nginx..."
nginx &>> "$LOG_FILE"

# Start FastAPI app
log "Starting FastAPI application..."
cd api/
uvicorn main_prod:app --host 0.0.0.0 --port "$API_PORT" --workers 2 --log-level info &>> "$LOG_FILE" &
UVICORN_PID=$!

# Wait for background processes
wait $UVICORN_PID
