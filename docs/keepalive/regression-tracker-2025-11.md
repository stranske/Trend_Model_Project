# Keepalive Regression Tracker — November 2025

## Problem Statement
- **Symptom:** Codex keepalive comments stopped posting, and initial PR comments no longer mirror the entire originating issue template.
- **Immediate Cause:** `scripts/keepalive-runner.js` currently demands `Scope`, `Tasks`, and `Acceptance Criteria` headings in order; when `Scope` is absent (which is expected per design), the extraction fails and the whole keepalive cycle aborts. In parallel, Agents 63 intake no longer copies the full issue body into the first PR comment, so even the required Tasks + Acceptance sections disappear from the timeline.
- **Impact:** Keepalive automation cannot re-engage Codex on PR #3827 or related branches. The orchestrator claims success even though no actionable instructions reach the agent, eroding trust in status updates.

## Recent Failed Attempts (past 3 days)
| Date (UTC) | Workflow / PR | Observed Failure | Notes |
|------------|---------------|------------------|-------|
| 2025-11-28 | PR #3827 keepalive sweep | Runner logs "scope/tasks/acceptance block unavailable" and skips commenting. | Confirms hard dependency on `Scope` section; no remediation applied yet. |
| 2025-11-27 | Agents 63 Issue Intake bootstrap | First PR comment lacks the issue's Tasks + Acceptance content. | Intake workflow no longer copies the issue template when creating the bootstrap PR description. |
| 2025-11-26 | Manual keepalive retry | Keepalive comment posted without actionable scope block; Codex ignored due to missing tasks. | Shows that partial fixes went out but were incomplete; notes referenced as "fixed" in commit message. |

*(Add new rows here as additional failures are confirmed.)*

## Merge Audit Queue
Review the last ~8 merges touching keepalive / intake automation to avoid reapplying ineffective fixes.

| Order | Commit / PR | Files of Interest | Status |
|-------|-------------|-------------------|--------|
| 1 | `71b3b8b1` / PR #3828 — *Improve status summary parsing from issue scope* | `.github/scripts/issue_scope_parser.js`, `.github/workflows/agents-pr-meta.yml` | Still rejects PRs that only carry Tasks + Acceptance; parser insists on `Scope` heading so keepalive extraction keeps failing. |
| 2 | `a87bc394` / PR #3826 — *Improve PR issue context summary extraction* | `.github/scripts/issue_scope_parser.js`, `.github/workflows/reusable-agents-issue-bridge.yml` | Added plain-heading parsing but never re-inserts Tasks/Acceptance into PR body, so bootstrap comments remain empty. |
| 3 | `66923c6b` / PR #3814 — *Restore issue details in Agents 63 PR bootstrap* | `.github/workflows/reusable-agents-issue-bridge.yml`, `pr-00-gate.yml` | Claimed to restore issue scope copy, yet PR #3827 still lacks the Task/Acceptance block, meaning the workflow regression persists. |
| 4 | `f22d84c1` / PR #3823 — *Fix issue context comment and status summary population* | `.github/workflows/reusable-agents-issue-bridge.yml` | Adjusted comment wiring but never propagated the issue template into the opening PR comment. |
| 5 | `599a5631` / PR #3804 — *Improve keepalive scope extraction* | `scripts/keepalive-runner.js`, `tests/test_keepalive_workflow.py` | Introduced the hard requirement for Scope/Tasks/Acceptance ordering, creating today’s blocker when Scope is intentionally omitted. |
| 6 | `9b8fb932` / PR #3794 — *Align keepalive markers and run-cap defaults* | `scripts/keepalive_instruction_segment.js`, `.github/scripts/keepalive_gate.js`, related tests | Updated markers but continued to assume all three headings exist; no fallback for Tasks-only payloads. |
| 7 | `8481db34` / PR #3829 — *Refresh PR3827 keepalive checklist* | `docs/keepalive/status/PR3827_Status.md` | Documentation-only merge that stated keepalive status was refreshed even though automation still skipped due to missing scope. |
| 8 | `4a281aa9` / PR #3815 — *Replace Issues.txt with keepalive branch-sync work* | `docs/keepalive/BranchSyncAutomationGaps.md`, `Issues.txt` | Logged branch-sync gaps but shipped no code changes, so the same regression kept resurfacing. |

*(Fill entries with actual SHAs/PR numbers while auditing.)*

## Remediation Plan (active)
1. **Keep the canonical Scope / Tasks / Acceptance template working**: ensure the parser + bridge expect those headings and never regress on the long-standing issue template. If a section is missing, surface an actionable warning (in intake summary or PR comment) so the user fixes the issue before automation fails.
2. **Restore bootstrap comment copy** in Agents 63 intake so the entire issue body (with Scope/Task List/Acceptance Criteria) lands in the first PR comment; this keeps the task checklist alive even if Scope is intentionally empty.
3. **Manual backfill for PR #3827** to keep Codex unblocked while automation rolls out.
4. **Regression tests**: extend the keepalive runner and intake bridge tests to cover the canonical template plus the warning path when sections are missing.
5. **Workflow cross-check**: audit other consumers (agents-pr-meta, issue bridge, keepalive contract/gate) to ensure none still require the removed headings or silently drop required sections.
6. **Documentation & runbook updates** here and in `agent_codex_troubleshooting.md` so future regressions (missing template sections) are surfaced early with exact remediation steps.

### Completion Gate (must be true before declaring the issue fixed)
- [ ] Agents 63 intake dry-run shows the opening PR comment populated with Tasks + Acceptance for a new issue.
- [ ] Keepalive workflow posts a comment on PR #3827 even when only Tasks/Acceptance are present, and Codex acknowledges the instructions.
- [ ] All updated workflows/tests pass in CI (`dev_check`, `validate_fast`, `check_branch --fast`).
- [ ] Documentation updated with remediation steps plus a link to this tracker.
- [ ] At least one follow-up keepalive run (real data) completes without "scope block unavailable" warnings.

## Progress Log
- 2025-11-28 14:10 UTC — Logged regression details and created this tracker per user request.
- 2025-11-28 16:05 UTC — Updated `scripts/keepalive-runner.js` so Tasks + Acceptance-only payloads are accepted, extended detection regex, added new scope parser tests, and ran `node --test .github/scripts/__tests__/keepalive-runner-scope.test.js` plus `./scripts/dev_check.sh --changed --fix`.
- 2025-11-28 18:42 UTC — Added checklist normalization so plain bullet lists render as `- [ ]` items, injected the auto-status summary block into existing PR bodies (even in invite mode), ensured the issue-context comment always posts, and re-ran `node --test .github/scripts/__tests__/issue_scope_parser.test.js` followed by `./scripts/dev_check.sh --changed --fix`.
- 2025-11-28 18:55 UTC — Deflaked the helper-sync harness by extending the `create_pr` TTL fixture, verified with `pytest tests/test_keepalive_post_work.py -k create_pr` (py311 + venv), and re-ran the full suite via `./scripts/run_tests.sh` (all 3,696 tests green).
- 2025-11-28 19:10 UTC — Confirmed Issue-label → PR bootstrap works again (`agent:codex` on Issue #3821 spawned PR #3833 automatically). Keepalive state comment now renders even when scope/tasks are missing, but the completion gate remains **open** until a real issue populates Tasks + Acceptance and we observe a follow-up keepalive round.
- 2025-11-28 20:05 UTC — Added `analyzeSectionPresence()` helper plus workflow wiring so Agents 63 surfaces warnings whenever Scope/Tasks/Acceptance are missing. Missing sections now trigger PR + issue comments **and** the Automated Status Summary now emits a human-readable explanation when it cannot populate the block, detailing which headings it expects and what failed. Validated via `node --test .github/scripts/__tests__/issue_scope_parser.test.js` and `./scripts/dev_check.sh --changed --fix`.
- 2025-11-28 20:32 UTC — Reworked the bridge so the Automated Status Summary refuses to show placeholder checklists when any canonical section is missing. Instead, it now emits a `⚠️ Summary Unavailable` block that lists the expected headings and calls out the missing sections, while the PR comment still carries placeholders so keepalive can run. Re-tested with `./scripts/dev_check.sh --changed --fix`.
- _(add more entries as steps complete)_

## Validation Checklist (to run after fixes)
- [ ] `scripts/dev_check.sh --changed --fix`
- [ ] `./scripts/validate_fast.sh --fix`
- [ ] `./scripts/check_branch.sh --fast --fix`
- [ ] Re-run Agents 63 intake in dry-run mode; confirm first PR comment copies Tasks + Acceptance.
- [ ] Trigger keepalive workflow (dry-run + live) and verify instructions post even without Scope section.
- [ ] Confirm Codex responds to keepalive comment on PR #3827.

## Open Questions
- Do any other workflows assume the `Scope` heading exists? Need confirmation before changing parsers.
- Are there outstanding PRs that already contain manual scope/tasks blocks we can reuse as fixtures?
