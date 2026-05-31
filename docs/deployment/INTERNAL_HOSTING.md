# Internal / on-prem hosting for proprietary data

This recipe runs the Trend Model Streamlit app on an **internal host** so that
proprietary returns and the deterministic engine stay entirely inside the org
perimeter, while LLM features (which are optional) are routed only through an
**authorized no-train endpoint** via the bundled proxy. It is the compliant
alternative to Streamlit Community Cloud, which is external SaaS and therefore
unsuitable for real proprietary data.

For external sharing with **synthetic** data instead, use the stlite browser
demo (issue #5343 / `README_APP.md`) — that path never touches real data and is
out of scope here.

## What stays in-perimeter

| Concern | Where it runs |
| --- | --- |
| Deterministic analysis engine + uploaded data | the `app-internal` container, bound to an internal interface |
| LLM API key | the `llm-proxy` sidecar only (never the browser, never the app env in plain form) |
| LLM prompt/response traffic | egresses **only** through `llm-proxy` to the operator-supplied upstream |

The only outbound LLM egress is from `llm-proxy` to the upstream you configure.
Point it at an authorized **no-train** endpoint. An on-prem
[Ollama](https://ollama.com) server (the `langchain-ollama` extra is already in
the `[llm]` group) is the lowest-egress option because prompts never leave the
perimeter at all.

## Components this builds on

- `trend-llm-proxy` console script (`pyproject.toml`) →
  `src/trend_analysis/llm_proxy/server.py`, an OpenAI-compatible proxy that
  keeps the API key server-side and binds `0.0.0.0:8799` by default.
- `TREND_LLM_BASE_URL` plumbing in
  `streamlit_app/components/llm_settings.py` →
  `src/trend_analysis/llm/providers.py` (`base_url` override), so the app talks
  to the proxy instead of the public API.
- `TREND_LLM_ZONE` switch (new): `internal_authorized` routes via the proxy;
  `disabled` hides every LLM panel so a zone with no authorized endpoint still
  runs the deterministic engine.

## Run it (Docker Compose)

The internal services live behind the `internal` compose profile, so the
default `docker compose up` is unchanged.

```bash
# 1. Build the image (once).
docker compose build app

# 2. Set the authorized no-train upstream and (optionally) a proxy token.
export TS_LLM_PROXY_UPSTREAM="https://your-authorized-no-train-endpoint"   # or http://ollama:11434 for on-prem
export TS_LLM_PROXY_TOKEN="$(openssl rand -hex 24)"        # optional shared secret
export TREND_LLM_API_KEY="..."                              # upstream key, stays in the proxy
# Optional data-egress ceiling (bytes) — bound what can leave the perimeter:
export TS_LLM_PROXY_MAX_BODY_BYTES=1048576

# 3. Start the internal app + proxy sidecar.
docker compose --profile internal up app-internal llm-proxy
```

The app is published on `127.0.0.1:8501` (loopback only) by default. For
LAN-internal access, change the host portion of the `app-internal` port mapping
in `docker-compose.yml` to your internal NIC address — never `0.0.0.0` on an
internet-facing host.

### Running a proprietary zone with no LLM

If a zone has no authorized LLM endpoint, run with the LLM features disabled.
The deterministic engine is unaffected:

```bash
TREND_LLM_ZONE=disabled docker compose --profile internal up app-internal
```

## What is and isn't sent upstream

- **Sent upstream (only when zone is `internal_authorized`):** the LLM
  request bodies the app constructs for the Explain Results / LLM Comparison
  panels, forwarded by `llm-proxy` to `TS_LLM_PROXY_UPSTREAM`.
- **Never sent:** uploaded returns files, the analysis cache, and the API key
  (the proxy injects the key server-side; the browser and app never hold the
  upstream key when a proxy token is used).
- **Bounded:** when `TS_LLM_PROXY_MAX_BODY_BYTES` is set, the proxy rejects any
  request body larger than that ceiling with HTTP 413 before forwarding, so an
  oversized payload cannot silently egress. Unset means no limit (default
  behavior).

> The proxy forwards the request body verbatim within the size ceiling; it does
> not field-redact prompt contents. Treat prompt construction in the app as the
> redaction boundary and keep the upstream a no-train endpoint. A richer
> field-allowlist redaction hook can extend `request_body_within_limit` in
> `llm_proxy/server.py`.

## Live-verification gate (operator checklist)

Run these on the internal host before declaring the deployment good. This is
the acceptance gate for issue #5344.

1. **App loads on the internal URL.** Browse to `http://127.0.0.1:8501` (or your
   internal NIC address) and confirm the Streamlit UI renders.
   ```bash
   curl --fail http://127.0.0.1:8501/_stcore/health && echo OK
   ```
2. **A real run completes in-perimeter.** Upload a real sample returns CSV and
   run a single-period analysis; confirm metrics/charts render. No outbound
   call leaves the host for this step (deterministic engine only).
3. **LLM fails closed when the proxy is down (zone `internal_authorized`).**
   Stop `llm-proxy` (`docker compose --profile internal stop llm-proxy`), then
   trigger an Explain Results / LLM Comparison action. It must error against the
   proxy URL and make **no direct call** to `api.openai.com` (verify with host
   egress monitoring / firewall logs). Restart the proxy to restore.
4. **Deterministic features still work with zone `disabled`.** Restart with
   `TREND_LLM_ZONE=disabled`; confirm the LLM panels are hidden (a single
   "LLM features are disabled" notice appears) and that analysis runs still
   complete.

## Security notes

- The compose config in this repo contains **no public-SaaS hostnames and no
  embedded real API key**; the LLM upstream and key are operator-supplied at
  deploy time via environment variables.
- Set `TS_LLM_PROXY_TOKEN` so only the app (which presents the token) can use
  the proxy.
- Keep the app bound to an internal interface; do not expose `8501` publicly.
