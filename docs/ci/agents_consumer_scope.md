# Agents Consumer Workflow (Manual Dispatch)

`.github/workflows/agents-consumer.yml` provides a manual dispatch wrapper around
[`reusable-70-agents.yml`](../../.github/workflows/reusable-70-agents.yml). Use it
when you want to bypass the orchestrator schedule and invoke the reusable
workflow directly with bespoke settings.

## Dispatch guidance

1. Navigate to **Actions → Agents Consumer → Run workflow**.
2. Toggle the high-level switches exposed in the UI (readiness, preflight,
   diagnostics, verify issue, watchdog, keepalive, bootstrap, draft PR). For
   bespoke overrides—custom readiness lists, Codex command phrases, bootstrap
   labels, diagnostic dry-run flags—use the dedicated string inputs (values
   such as `'true'` / `'false'` forward verbatim) or provide an `options_json`
   payload.
3. Review the dispatched job named **Dispatch reusable agents toolkit** to
   confirm downstream behaviour and capture outputs.

The workflow enforces a concurrency group of `agents-consumer-${ref}`. Triggering
another run on the same branch cancels any in-flight execution and prevents
manual re-trigger storms.

For scheduled or automated routing prefer the
[`agents-70-orchestrator.yml`](../../.github/workflows/agents-70-orchestrator.yml)
entry point. It fans out to the same reusable toolkit while handling the
recurring keepalive cadence.
