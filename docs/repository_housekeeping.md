# Repository housekeeping

This guide centralizes the archival rules and quarterly hygiene checks for the
Trend Model repository. It is meant to keep automation artifacts, retired
content, and audit trails organized so they remain discoverable without
cluttering the active contributor surface.

## Archiving rules and folder owners
| Area | What to archive | When to archive | Owner |
| --- | --- | --- | --- |
| Agents automation (`agents/`, `.github/workflows/agents-*`) | Obsolete orchestration experiments, temporary shims, and logs that have shipped into a published runbook. Keep active dispatch surfaces intact. | After replacement workflows are live for a full release cycle and the allowlist in `docs/AGENTS_POLICY.md` is updated. | Automation maintainers (`@stranske` for workflow entrypoints). |
| Retired content (`retired/`, `archives/`) | Superseded notebooks, deprecated guides, and frozen datasets. Preserve a changelog entry or pointer explaining the retirement. | Immediately after a successor guide or dataset publishes so duplicated guidance is not maintained in parallel. | Documentation maintainers (ops rotation). |
| Logs and keepalive traces (`docs/keepalive`, `coverage_keepalive_status.md`, `issues-3260-*`) | Weekly/monthly probes, keepalive summaries, and issue traces once they age out of the active triage window. Compress oversized logs before moving. | At the end of each quarter after closing outstanding incidents. Keep the current quarter in place for context. | Reliability lead on call.
| Root files (top-level `README*.md`, `Issues.txt`, `gate-summary.md`, `keepalive_status.md`) | Snapshot copies of root files when replaced by new entrypoints, plus any one-off audits tied to incident timelines. | When the canonical navigation changes (for example, new doc index) or when an audit is finished and only the summary is needed. | Repo stewardship (docs/code owners).

## Quarterly housekeeping checklist
Follow this checklist during the first week of each quarter to keep the
repository organized and searchable.

1. **Inventory** – Sweep `agents/`, `archives/`, `retired/`, and `docs/keepalive` for items older than the last release. Confirm root files match `docs/INDEX.md` and flag duplicates. *(Owner: docs/ops rotation).* 
2. **Archive** – Move superseded notebooks, logs, and shims to `archives/` or `retired/` with a short README that points to the successor. Update `docs/AGENTS_POLICY.md` or related runbooks when agents workflows shift. *(Owner: area lead listed above).* 
3. **Link audit** – Validate cross-references in `docs/INDEX.md`, root READMEs, and any playbooks that mention archived items. Remove dead links and add redirects where necessary. *(Owner: docs/ops rotation with verification by automation maintainers for agents references).*

Document the outcome in the quarterly ops notes so future sweeps can track what
changed and why.
