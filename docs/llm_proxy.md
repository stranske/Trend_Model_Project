# LLM Proxy for Streamlit Deployments

Use the LLM proxy to keep API keys server-side while allowing Streamlit users to access LLM features.

## Why
- API keys never reach the browser or client machines.
- One shared key powers all usage.
- Optional proxy token restricts access without billing logic.

## Requirements
Install the app extras:

```
pip install -e ".[app]"
```

## Configuration
Set environment variables on the proxy host:

- `TS_STREAMLIT_API_KEY`: upstream OpenAI API key (preferred)
- `TS_OPENAI_STREAMLIT`: legacy upstream key name (also accepted)
- `OPENAI_API_KEY` or `TREND_LLM_API_KEY`: fallback key names (optional)
- `TS_LLM_PROXY_TOKEN`: optional shared token for access control
- `TS_LLM_PROXY_UPSTREAM`: optional override for upstream base URL (default: https://api.openai.com)

## Run the proxy

```
trend-llm-proxy --host 0.0.0.0 --port 8799
```

⚠️ Binding to `0.0.0.0` exposes the proxy on all interfaces. For local-only use, set `--host 127.0.0.1`.

## Configure Streamlit
Set these environment variables on the Streamlit server:

- `TS_LLM_PROXY_URL`: e.g. `https://your-llm-proxy.example.com`
- `TS_LLM_PROXY_TOKEN`: must match the proxy token if enabled

The app will route OpenAI calls through the proxy automatically.
