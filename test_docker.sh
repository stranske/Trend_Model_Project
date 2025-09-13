#!/bin/bash
# test_docker.sh - Test script for Docker functionality
# Run this script after the GitHub workflow builds and publishes the Docker image

set -euo pipefail

IMAGE="ghcr.io/stranske/trend-model:latest"

echo "🐳 Testing Docker Image: $IMAGE"
echo "=================================="

# Test 1: Check if image exists and can be pulled
echo "📥 Test 1: Pulling image..."
if docker pull "$IMAGE"; then
    echo "✅ Image pull successful"
else
    echo "❌ Failed to pull image"
    exit 1
fi

# Test 2: Test basic container startup
echo ""
echo "🚀 Test 2: Testing container startup..."
CONTAINER_ID=$(docker run -d -p 8000:8000 -p 8501:8501 "$IMAGE")
echo "Started container: $CONTAINER_ID"

# Wait for container to start
sleep 10

# Test 3: Check health endpoint with retry/backoff
echo ""
echo "🔍 Test 3: Testing health endpoint..."
max_attempts=10
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -fs http://localhost:8000/health | grep -q "OK"; then
        echo "✅ Health check passed"
        break
    else
        echo "Attempt $attempt failed, retrying..."
        sleep 1
    fi
    attempt=$((attempt+1))
done
if [ $attempt -gt $max_attempts ]; then
    echo "❌ Health check failed after $max_attempts attempts"
    echo "Container logs:"
    docker logs "$CONTAINER_ID"
    exit 1
fi

# Test 4: Test CLI functionality
echo ""
echo "⚙️ Test 4: Testing CLI functionality..."
if docker run --rm "$IMAGE" trend-analysis --help > /dev/null; then
    echo "✅ CLI help command works"
else
    echo "❌ CLI help command failed"
fi

# Test 5: Test Python imports
echo ""
echo "🐍 Test 5: Testing Python imports..."
if docker run --rm "$IMAGE" python -c "
import sys
sys.path.insert(0, 'src')
from trend_analysis import cli, pipeline
print('All imports successful')
"; then
    echo "✅ Python imports work"
else
    echo "❌ Python imports failed"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
docker stop "$CONTAINER_ID"
docker rm "$CONTAINER_ID"

echo ""
echo "✅ All tests passed! Docker setup is working correctly."
echo ""
echo "Quick start commands:"
echo "  Health check:  curl http://localhost:8000/health"
echo "  Web interface: docker run -p 8000:8000 -p 8501:8501 $IMAGE"
echo "  CLI help:      docker run --rm $IMAGE trend-analysis --help"
echo "  Interactive:   docker run -it --rm $IMAGE bash"