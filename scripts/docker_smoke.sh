#!/usr/bin/env bash
# Lightweight local Docker build + smoke health check mirroring CI docker.yml smoke job.
# Usage: ./scripts/docker_smoke.sh
set -euo pipefail

# Graceful skip if docker unavailable (local dev containers may lack daemon capabilities)
if ! command -v docker >/dev/null 2>&1; then
  echo "[docker-smoke] docker CLI not found; skipping." >&2
  exit 0
fi
if ! docker info >/dev/null 2>&1; then
  echo "[docker-smoke] docker daemon not available; skipping." >&2
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REGISTRY="local"
IMAGE_NAME="trend-model-local"
PORT="8000"
HEALTH_PATH="/health"

# Build (no push)
 echo "Building image ${REGISTRY}/${IMAGE_NAME}:latest" >&2
 docker build -t ${REGISTRY}/${IMAGE_NAME}:latest .

# Run container
echo "Starting container for smoke test" >&2
CID=$(docker run -d -p ${PORT}:${PORT} ${REGISTRY}/${IMAGE_NAME}:latest)
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

# Probe health endpoint with retries
max_attempts=15
attempt=1
while [[ $attempt -le $max_attempts ]]; do
  if curl -fs "http://localhost:${PORT}${HEALTH_PATH}" | grep -q "OK"; then
    echo "Smoke health check passed on attempt $attempt" >&2
    echo "Docker smoke: PASS" >&2
    exit 0
  fi
  sleep 1
  attempt=$((attempt+1))
  echo "Retry $attempt..." >&2
done

echo "Health check failed after ${max_attempts} attempts" >&2
exit 1
