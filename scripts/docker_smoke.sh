#!/usr/bin/env bash
# Lightweight local Docker build + smoke health check mirroring CI pr-12-docker-smoke.yml job.
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
docker build -t "${REGISTRY}/${IMAGE_NAME}:latest" .

# Run container
echo "Starting container for smoke test" >&2
CID=$(docker run -d -p "${PORT}:${PORT}" "${REGISTRY}/${IMAGE_NAME}:latest")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

# Probe health endpoint with retries
max_attempts=15
attempt=1
last_curl_status=0
last_health_response=""
while [[ $attempt -le $max_attempts ]]; do
  curl_output=""
  curl_status=0
  if ! curl_output=$(curl --fail --silent --show-error --max-time 2 "http://localhost:${PORT}${HEALTH_PATH}"); then
    curl_status=$?
  fi
  last_curl_status=$curl_status

  if [[ $curl_status -eq 0 ]]; then
    last_health_response="$curl_output"
    if (
      HEALTH_RESPONSE="$curl_output" python - <<'PY'
import json
import os
import sys

payload = os.environ.get("HEALTH_RESPONSE", "").strip()
if not payload:
    sys.exit(1)

if payload.lower() == "ok":
    sys.exit(0)

try:
    data = json.loads(payload)
except json.JSONDecodeError:
    sys.exit(1)

status = str(data.get("status", "")).lower()
sys.exit(0 if status == "ok" else 1)
PY
    ); then
      echo "Smoke health check passed on attempt $attempt" >&2
      echo "Docker smoke: PASS" >&2
      exit 0
    fi
  else
    last_health_response=""
  fi

  sleep 1
  attempt=$((attempt+1))
  echo "Retry $attempt..." >&2
done

echo "Health check failed after ${max_attempts} attempts (last curl exit: ${last_curl_status})" >&2
if [[ -n "$last_health_response" ]]; then
  printf 'Last health response:%s%s' "\n" "$last_health_response" >&2
  printf '\n' >&2
fi
exit 1
