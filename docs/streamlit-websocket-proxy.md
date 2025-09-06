# Streamlit WebSocket Proxy

This proxy server solves the issue where Streamlit's frontend requires WebSocket endpoints like `/_stcore/stream` for bidirectional updates that aren't supported by simple HTTP-only proxies.

## Problem Statement

The original issue was that a wrapper proxied requests to Streamlit by forwarding only HTTP verbs via `httpx`. Streamlit's frontend relies on WebSocket endpoints such as `/_stcore/stream` for bidirectional updates; without a WebSocket route the browser will fail to connect and the UI shows a "network connection error" even though `/health` succeeds.

## Solution

This proxy server implements both HTTP and WebSocket forwarding:

1. **HTTP Requests**: Forwarded using `httpx` with full header and body support
2. **WebSocket Connections**: Forwarded directly to Streamlit for real-time updates
3. **Health Checks**: Proper forwarding of `/_stcore/health` and other endpoints

## Installation

Install the required dependencies:

```bash
pip install fastapi uvicorn httpx websockets
```

## Usage

### Command Line

Run the proxy from the command line:

```bash
# Default configuration (proxy on :8500, Streamlit on localhost:8501)
python -m trend_analysis.proxy

# Custom configuration
python -m trend_analysis.proxy \
    --streamlit-host localhost \
    --streamlit-port 8501 \
    --proxy-host 0.0.0.0 \
    --proxy-port 8500 \
    --log-level INFO
```

### Programmatic Usage

Use the proxy in your Python code:

```python
from trend_analysis.proxy import StreamlitProxy
import asyncio

async def main():
    proxy = StreamlitProxy(streamlit_host="localhost", streamlit_port=8501)
    try:
        await proxy.start(host="0.0.0.0", port=8500)
    finally:
        await proxy.close()

asyncio.run(main())
```

### Docker Usage

Update your Docker setup to use the proxy:

```dockerfile
# Start Streamlit on internal port
EXPOSE 8500
CMD ["python", "-m", "trend_analysis.proxy", "--proxy-port", "8500"]
```

Or with docker-compose:

```yaml
services:
  streamlit:
    image: your-streamlit-image
    command: streamlit run app.py --server.port=8501
    expose:
      - "8501"
  
  proxy:
    image: your-proxy-image  
    command: python -m trend_analysis.proxy --streamlit-host streamlit
    ports:
      - "8500:8500"
    depends_on:
      - streamlit
```

## How It Works

### HTTP Forwarding

All HTTP requests (GET, POST, PUT, DELETE, etc.) are forwarded to Streamlit using `httpx`:

1. Request headers are copied (excluding `host`)
2. Request body is read and forwarded
3. Response is streamed back to the client
4. Response headers are filtered appropriately

### WebSocket Forwarding

WebSocket connections are established to both the client and Streamlit:

1. Accept incoming WebSocket connection
2. Connect to Streamlit WebSocket endpoint
3. Set up bidirectional message forwarding
4. Handle connection cleanup on disconnect

### Supported Endpoints

- `/_stcore/stream` - Primary WebSocket endpoint for real-time updates
- `/_stcore/health` - Health check endpoint
- All other Streamlit endpoints (static files, API calls, etc.)

## Configuration

### Environment Variables

```bash
export STREAMLIT_HOST=localhost
export STREAMLIT_PORT=8501
export PROXY_HOST=0.0.0.0
export PROXY_PORT=8500
```

### CLI Arguments

- `--streamlit-host`: Host where Streamlit is running (default: localhost)
- `--streamlit-port`: Port where Streamlit is running (default: 8501)
- `--proxy-host`: Host to bind the proxy server (default: 0.0.0.0)
- `--proxy-port`: Port to bind the proxy server (default: 8500)
- `--log-level`: Logging level (default: INFO)

## Testing

Run the test suite:

```bash
# Run proxy-specific tests
pytest tests/test_proxy.py -v

# Run all tests
pytest
```

## Troubleshooting

### Common Issues

**"Required dependencies not available"**
- Install proxy dependencies: `pip install fastapi uvicorn httpx websockets`

**"Connection refused" errors**
- Ensure Streamlit is running on the configured host/port
- Check firewall settings
- Verify network connectivity

**WebSocket connection fails**
- Check that Streamlit's WebSocket endpoints are accessible
- Verify proxy can connect to Streamlit WebSocket URLs
- Check browser developer tools for WebSocket errors

### Debug Mode

Enable debug logging to see detailed proxy activity:

```bash
python -m trend_analysis.proxy --log-level DEBUG
```

### Health Checks

Test the proxy health:

```bash
# Test HTTP forwarding
curl http://localhost:8500/_stcore/health

# Test main Streamlit page
curl http://localhost:8500/
```

## Architecture

```
Client Browser
     |
     | HTTP/WebSocket
     v
Streamlit Proxy (:8500)
     |
     | HTTP (httpx) / WebSocket (websockets)
     v  
Streamlit App (:8501)
```

The proxy acts as a transparent bridge, ensuring that all Streamlit functionality works correctly while adding WebSocket support that was missing in HTTP-only proxies.