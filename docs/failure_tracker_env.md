# Failure Tracker Environment Reference

Central reference for all environment variables used by the `failure-tracker` job inside `.github/workflows/maint-46-post-ci.yml`.

| Variable | Default | Type | Scope | Purpose / Behaviour | Notes |
|----------|---------|------|-------|---------------------|-------|
| `RATE_LIMIT_MINUTES` | 15 | int | Existing issue comment gate | Minimum minutes between automated issue comments for the same failure issue | Prevents noisy duplicate comments when multiple jobs fail close together |
| `STACK_TOKENS_ENABLED` | `true` | bool | Signature enrichment | Enables extraction + hashing of the first meaningful stack/exception line per failed job | Disabling reverts to coarse signature (job + step only) |
| `STACK_TOKEN_MAX_LEN` | 160 | int | Stack token normalization | Truncates normalized token to bounded length | Keeps signature stable + avoids gigantic tokens |
| `STACK_TOKEN_RAW` | `false` | bool | Stack token normalization | If `true`, skips normalization (only truncation) | Raw mode may fragment signatures across similar failures |
| `AUTO_HEAL_INACTIVITY_HOURS` | 24 | int/float | Success path (healer job) | Hours of no recurrence before closing failure issue | Applied only in `success` job – separates detection from healing |
| `FAILURE_INACTIVITY_HEAL_HOURS` | 0 | int/float | (Reserved) failure path | Would allow healing during failure workflow if >0 | Currently unused placeholder for future inline healing |
| `NEW_ISSUE_COOLDOWN_HOURS` | 12 | int/float | New issue creation path | If >0, within window try to append to an existing issue instead of creating a new one | Tuned lower to curb duplicate issues while staying responsive |
| `COOLDOWN_SCOPE` | `global` | enum (`global`,`workflow`,`signature`) | Cooldown selection | Controls which existing issue is eligible for append under cooldown | `global` → most recent; `workflow` → same workflow name; `signature` → only exact signature |
| `COOLDOWN_RETRY_MS` | 3000 | int (ms) | Race mitigation | Wait then re-check candidate issues before creating a new one | Reduces TOCTOU window between parallel runs |
| `DISABLE_FAILURE_ISSUES` | `false` | bool | All failure tracking | If `true` skips create/update (summary still written) | Use for dry-runs or temporary silence |
| `OCCURRENCE_ESCALATE_THRESHOLD` | 3 | int | Existing issue update | If >0 and occurrences ≥ threshold → escalation branch | Escalate on the third occurrence |
| `ESCALATE_LABEL` | `priority: high` | string | Escalation | Label added once threshold crossed | Auto-created if missing |
| `ESCALATE_COMMENT` | (empty) | string | Escalation | Optional custom escalation comment body | Falls back to auto-generated message |

## Default Labels
The workflow seeds each failure issue with the following labels to aid triage:

- `ci-failure`
- `ci`
- `devops`
- `priority: medium`

All labels are created on-demand if they are missing from the repository so the automation remains resilient across new forks.

## Signature Construction
Signature hash = SHA-256( sorted failed jobs mapped to `jobName::firstFailingStep::stackToken`) truncated to 12 hex chars. Workflow name + hash produce the issue title: `Workflow Failure (<workflow>|<hash>)`.

## Cooldown Flow
1. Failure detected and no existing signature issue.
2. If `NEW_ISSUE_COOLDOWN_HOURS > 0` attempt append using `COOLDOWN_SCOPE`:
   - `global`: newest open `ci-failure` issue (time window)
   - `workflow`: newest with matching workflow prefix in title
   - `signature`: open issue for identical signature (rare here because search already failed)
3. Retry after `COOLDOWN_RETRY_MS` if first attempt finds none.
4. Create new issue only if both attempts fail.

## Occurrence Tracking & History
Each issue body maintains:
```
Occurrences: <n>
Last seen: <ISO8601>
Healing threshold: Auto-heal after <AUTO_HEAL_INACTIVITY_HOURS>h stability (success path)
<!-- occurrence-history-start -->
| Timestamp | Run | Sig Hash | Failed Jobs |
|---|---|---|---|
| 2025-09-23T10:12:34Z | [run](...) | abc123def456 | 2 |
... up to 10 rows ...
<!-- occurrence-history-end -->
```
History is capped at 10 latest rows (most recent first).

## Escalation Mechanics
When an existing issue is updated:
- Increment occurrence counter
- If `OCCURRENCE_ESCALATE_THRESHOLD > 0` and reached:
  - Ensure `ESCALATE_LABEL` exists (create if needed)
  - Apply label if absent
  - Post `ESCALATE_COMMENT` (or default) only once

## Normalization Rules (when `STACK_TOKEN_RAW != true`)
Applied in order to the first detected error/exception line:
1. Strip leading ISO8601 timestamps
2. Remove progress bracket tokens (e.g. `[12%]`)
3. Collapse whitespace
4. Truncate to first `Token: Message` segment if pattern matches
5. Fallback to `no-stack` if empty
6. Truncate to `STACK_TOKEN_MAX_LEN`

## Disable Modes
- Set `DISABLE_FAILURE_ISSUES=true` to fully suppress create/update while retaining summary output. Useful during refactors.

## Local / Sandbox Testing
Trigger artificial failures by introducing a controlled failing step in a temporary branch and observing issue evolution:
```yaml
- name: Force fail (sandbox)
  run: |
    echo "Simulated failure" >&2
    exit 1
```
Shorten windows for rapid iteration:
```yaml
NEW_ISSUE_COOLDOWN_HOURS: 0.001   # ~3.6 seconds
AUTO_HEAL_INACTIVITY_HOURS: 0.01  # ~36 seconds
OCCURRENCE_ESCALATE_THRESHOLD: 2
```
(Use only on non-default branches.)

## Recommended Production Settings
| Goal | Suggested Overrides |
|------|---------------------|
| Reduce duplicate noise | `NEW_ISSUE_COOLDOWN_HOURS: 12` & `COOLDOWN_SCOPE: workflow` |
| High-severity escalation | `OCCURRENCE_ESCALATE_THRESHOLD: 3`, custom `ESCALATE_LABEL` |
| Temporarily silence | `DISABLE_FAILURE_ISSUES: true` |
| Increase aggregation | `COOLDOWN_SCOPE: global` |
| Precise grouping | `COOLDOWN_SCOPE: signature` |

## Future Enhancements (Ideas)
- Inline `FAILURE_INACTIVITY_HEAL_HOURS` support
- Metrics emission (JSON / Prometheus) per run
- Webhook dispatch on escalation
- Daily digest summarizing open signatures & occurrence deltas

---
_Last updated: 2025-09-23_
