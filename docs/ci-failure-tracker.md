# CI Failure Tracker (Phase-2 Enhancements)

This document summarises the behaviour and configuration of the enhanced failure tracking workflow.

## Overview
`maint-33-check-failure-tracker.yml` listens to completed runs of the primary CI workflows (`CI`, `Docker`, and the manual `CI Selftest`). On failure it:

1. Enumerates failed jobs and the first failing step.
2. Optionally extracts a stack token (first exception or error line) per failed job.
3. Builds a deterministic signature: `workflow|sha256(job::step::stackToken...)[:12]`.
4. Opens or updates a single GitHub Issue per signature (labels: `ci-failure`, `workflows`, `devops`, `priority: medium`).
5. Maintains metadata header: Occurrences, Last seen, Healing threshold.
6. Appends a failure comment (rate-limited) with job + stack token tables.
7. On successful runs, scans for inactive issues and auto-closes those with no reoccurrence for the configured inactivity window.

## Configuration (Environment Variables)
| Variable | Purpose | Default |
|----------|---------|---------|
| `RATE_LIMIT_MINUTES` | Minimum minutes between new comments for same issue | 15 |
| `STACK_TOKENS_ENABLED` | Toggle stack token hashing (`true`/`false`) | true |
| `STACK_TOKEN_MAX_LEN` | Max chars retained from a stack/error line | 160 |
| `AUTO_HEAL_INACTIVITY_HOURS` | Hours of stability before success path auto-heal closes issue | 24 |
| `FAILURE_INACTIVITY_HEAL_HOURS` | (Reserved) Close during failure path if inactive for this many hours | 0 (disabled) |

## Signature Evolution
- Phase-1: job + first failing step.
- Phase-2: adds first stack/error line token (or `no-stack` / `stacks-off`).

## Rate Limiting
A run comment is suppressed if:
- The run URL already appears in an existing comment, OR
- The last comment is younger than `RATE_LIMIT_MINUTES`.

## Auto-Heal (Success Path)
On any successful monitored workflow run, open `ci-failure` issues are scanned. If `Last seen` is older than `AUTO_HEAL_INACTIVITY_HOURS`, the issue is commented on and closed.

## JSON Snapshot Artifact
Each successful run uploads `ci_failures_snapshot.json` containing an array of current open failure issues (number, occurrences, last_seen, timestamps). Use this for dashboards or external monitoring.

## Occurrence History
Each failure issue maintains an internal, capped (10 rows) occurrence history table between HTML comment markers:
```
<!-- occurrence-history-start -->
| Timestamp | Run | Sig Hash | Failed Jobs |
|---|---|---|---|
| 2025-09-23T12:34:56Z | run link | a1b2c3d4e5f6 | 2 |
<!-- occurrence-history-end -->
```
New failures prepend a row; table truncated at 10 to keep issues readable.

## Deterministic Signature Self-Test
Utility script: `tools/test_failure_signature.py`

Example:
```bash
python tools/test_failure_signature.py \
	--jobs '[{"name":"Tests","step":"pytest","stack":"ValueError: boom"}]' \
	--expected 0123456789ab
```
Integrate into local pre-flight checks to ensure signature algorithm adjustments are deliberate.

## Signature Guard Workflow
Workflow: `maint-40-ci-signature-guard.yml` runs on pushes / PRs and validates that a canonical fixture (`.github/signature-fixtures/basic_jobs.json`) hashes to the expected value stored in `basic_hash.txt`. Any intentional algorithm change should update both fixture and expected hash in the same commit.

## Manual Self-Test
You can manually validate behaviour:
1. Dispatch `CI Selftest` (will create a failing issue).
2. Rerun it but edit the `ci-selftest.yml` to succeed (or manually re-run jobs) to test auto-heal logic after adjusting `INACTIVITY_HOURS`.

## Future Extensions
- Persist aggregated metrics (failure frequency) as JSON artifact.
- Add PR comment summary for new signatures encountered in a PR context.
- Integrate stack token similarity clustering for noisy crash variants.

## Post CI Status Summary Aggregation
`maint-30-post-ci-summary.yml` replaced the legacy matrix summary workflow. It
listens to both the **CI** and **Docker** workflow completions and pushes all of
the high-value signals directly into the PR comment that reviewers already use.

**Signals merged into the comment**
- CI + Docker job table with failure prioritisation.
- Required-lane rollups (tests, automation, style, gate, Docker) with badges.
- Latest coverage averages and worst-job metrics plus deltas vs the previous
  recorded run (sourced from `coverage-trend.json` and history artifacts).
- Rendered `coverage_summary.md` content for file-level hotspots whenever the
  artifact exists.

**Artifacts consumed** (downloaded opportunistically; missing files are skipped)
- `coverage-summary`
- `coverage-trend`
- `coverage-trend-history`

Extending the comment is straightforward: add an additional download step for
the new artifact, parse it in the "Derive coverage stats" (or adjacent) step,
and thread the result into the comment body builder. Keep each computation
quick (<100 ms) because the workflow runs after every CI/Docker completion.

## Maintenance Checklist
- If new workflows should be monitored, add their names to the `workflows:` array under the `workflow_run` trigger.
- Keep labels consistent with project taxonomy.
- When refining stack heuristics, ensure deterministic fallback values to preserve signature stability.
