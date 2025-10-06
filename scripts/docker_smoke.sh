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

# Run container with a unique, traceable name so logs can be captured on failure.
RUN_ID=$(date +%s)
HEALTH_CONTAINER_NAME="docker-smoke-${RUN_ID}-$$"
cleanup() {
  docker rm -f "$HEALTH_CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker rm -f "$HEALTH_CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Starting container $HEALTH_CONTAINER_NAME for smoke test" >&2
CONTAINER_ID=$(docker run -d \
  --name "$HEALTH_CONTAINER_NAME" \
  -p "${PORT}:${PORT}" \
  "${REGISTRY}/${IMAGE_NAME}:latest")
echo "Container id: $CONTAINER_ID" >&2

# Probe health endpoint with retries mirroring the CI workflow behaviour.
HEALTH_PARSER_SCRIPT=$'import json\nimport os\nimport sys\n\npayload = os.environ.get("HEALTH_RESPONSE", "").strip()\nif not payload:\n    sys.exit(1)\n\nnormalized = payload.lower()\nif normalized in {"ok", "healthy", "pass", "passed", "up", "ready", "success"}:\n    sys.exit(0)\n\ntry:\n    data = json.loads(payload)\nexcept json.JSONDecodeError:\n    sys.exit(1)\n\nSUCCESS_VALUES = {"ok", "healthy", "pass", "passed", "true", "1", "up", "ready", "success"}\nKEYS_TO_CHECK = ("status", "state", "health", "message", "detail")\n\ndef is_success(value):\n    if isinstance(value, str):\n        return value.strip().lower() in SUCCESS_VALUES\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, (int, float)):\n        return value == 1\n    if isinstance(value, dict):\n        for key in KEYS_TO_CHECK:\n            if key in value and is_success(value[key]):\n                return True\n        return any(is_success(v) for v in value.values())\n    if isinstance(value, (list, tuple, set)):\n        return any(is_success(item) for item in value)\n    return False\n\nsys.exit(0 if is_success(data) else 1)'

max_attempts=5
attempt=1
last_curl_status=0
last_health_response=""
health_ready=0

while [[ $attempt -le $max_attempts ]]; do
  running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER_ID" 2>/dev/null || echo false)"
  if [[ "$running" != true ]]; then
    echo "Container exited before health check succeeded. Capturing logs..." >&2
    docker logs "$CONTAINER_ID" >&2 || true
    exit 1
  fi

  curl_output=""
  curl_status=0
  if ! curl_output=$(curl --fail --silent --show-error --max-time 2 "http://127.0.0.1:${PORT}${HEALTH_PATH}"); then
    curl_status=$?
  fi
  last_curl_status=$curl_status

  if [[ $curl_status -eq 0 ]]; then
    last_health_response="$curl_output"
    if HEALTH_RESPONSE="$curl_output" python -c "$HEALTH_PARSER_SCRIPT"
    then
      echo "Smoke health check passed on attempt $attempt" >&2
      echo "Docker smoke: PASS" >&2
      health_ready=1
      break
    fi
  else
    last_health_response=""
  fi

  echo "Attempt $attempt/$max_attempts failed, retrying in 2s..." >&2
  sleep 2
  attempt=$((attempt + 1))
done

if [[ $health_ready -ne 1 ]]; then
  echo "Health check failed after ${max_attempts} attempts (last curl exit: ${last_curl_status})" >&2
  if [[ -n "$last_health_response" ]]; then
    printf 'Last health response:%s%s' "\n" "$last_health_response" >&2
    printf '\n' >&2
  fi
  docker logs "$CONTAINER_ID" >&2 || true
  exit 1
fi
