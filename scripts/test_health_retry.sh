#!/bin/bash
# Test script to verify health endpoint retry logic works deterministically
# This simulates the CI/Docker smoke test scenario

set -euo pipefail

cd "$(dirname "$0")/.."

echo "🔍 Testing health endpoint retry logic..."

# Start health wrapper in background
# Activate virtual environment if available (Unix or Windows), else fallback to system Python
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "⚠️  No virtual environment found. Falling back to system Python."
fi

if ! python - <<'PY' >/dev/null 2>&1; then
import importlib
import sys

try:
    importlib.import_module("trend_portfolio_app.health_wrapper")
except ModuleNotFoundError:  # pragma: no cover - shell wrapper guard
    sys.exit(1)
sys.exit(0)
PY
then
    echo "trend-portfolio-app package not installed. Run 'pip install -e .[app]' first." >&2
    exit 1
fi

python -m trend_portfolio_app.health_wrapper &
HEALTH_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    kill $HEALTH_PID 2>/dev/null || true
    wait $HEALTH_PID 2>/dev/null || true
}
trap cleanup EXIT

# Test retry logic with backoff ≤10s total as specified in issue
echo "Testing retry logic with max 10 second total timeout..."
max_attempts=10
attempt=1
start_time=$(date +%s)

while [ $attempt -le $max_attempts ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    # Stop if we've exceeded 10 seconds total
    if [ $elapsed -ge 10 ]; then
        echo "❌ Timeout after ${elapsed}s (max 10s allowed)"
        exit 1
    fi
    
    if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
        end_time=$(date +%s)
        total_time=$((end_time - start_time))
        echo "✅ Health check passed on attempt $attempt after ${total_time}s"
        
        # Verify response content
        response=$(curl -s http://localhost:8000/health)
        if [ "$response" = "OK" ]; then
            echo "✅ Response content correct: '$response'"
        else
            echo "❌ Unexpected response content: '$response'"
            exit 1
        fi
        
        # Test root endpoint too
        root_response=$(curl -s http://localhost:8000/)
        if [ "$root_response" = "OK" ]; then
            echo "✅ Root endpoint also works: '$root_response'"
        else
            echo "❌ Root endpoint failed: '$root_response'"
            exit 1
        fi
        
        echo "✅ All health endpoint tests passed in ${total_time}s"
        exit 0
    else
        echo "Attempt $attempt failed, retrying... (${elapsed}s elapsed)"
        sleep 1
    fi
    attempt=$((attempt+1))
done

echo "❌ Health check failed after $max_attempts attempts"
exit 1
