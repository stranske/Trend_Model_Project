# Repository Health Self-Check Workflow (Implements #1175)

## Overview
The `repo-health-self-check.yml` workflow provides a scheduled, automated governance audit covering secrets, labels, and branch protection. It creates (or updates) a single tracking issue when deficiencies are detected and closes that issue automatically once all checks pass.

| Aspect | What It Verifies | Failure Handling |
| ------ | ---------------- | ---------------- |
| Secrets | Presence of required secrets (configurable list inline in workflow) | Missing names appended to tracking issue body |
| PAT Probe | Validity/permissions of service bot token (if configured) | Notes failure cause (403/404/timeout) |
| Labels | Required label set exists (e.g., `autofix`, `ci: quarantine`) | Missing labels enumerated |
| Branch Protection | Protection rules for `phase-2-dev` & `main` retrievable | Logs API retrieval failure or absence |

## Trigger Modes
- `schedule`: Daily cron (early UTC window) for unattended governance.
- `workflow_dispatch`: Manual on-demand validation.

## Issue Lifecycle
- Opens (or updates) a single canonical issue titled: `Repository Health Failing Checks`.
- Adds or replaces a machine section delimited by HTML comments for idempotent updates.
- Closes the issue automatically when all checks pass (adds a success summary comment first).

## Extending Checks
1. Add a new step producing a JSON or line-delimited output file under `.health/`.
2. Append its parsing to the aggregation script block that builds the issue body segment.
3. Keep parsing logic deterministic and side-effect free.

## Local Dry Run
You can replicate most logic locally using the `actions/github-script` fragments converted to a Node.js script with a `GITHUB_TOKEN` environment variable.

## Relationship to Other Workflows
- Complements reusable CI by ensuring structural prerequisites (labels, secrets) are continuously present.
- Differs from feature-specific watchdogs by focusing on static repository hygiene rather than dynamic runtime states.

## Closure of #1175
This document plus the merged workflow address the previously identified gap: lack of a unified scheduled self-check. After first successful scheduled run (no failures), close #1175 referencing commit SHAs introducing the workflow and this documentation file.

_Last updated: 2025-09-19_
