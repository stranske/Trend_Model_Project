# Codex Bootstrap Simulation & Verification Guide

This document describes the simulation controls, workflow dispatch inputs, and verification scenarios for the hardened **Codex Assign Minimal** workflow and associated composite action `codex-bootstrap`.

## Overview
The production workflow `codex-assign-minimal.yml` remains intentionally lean while supporting a comprehensive verification matrix through:

- Manual dispatch inputs (for deterministic scenario execution)
- Synthetic failure & behavior simulation labels
- Enforced command validation and fallback gating
- Artifact-based observability (`codex_bootstrap_result.json`, `codex_bootstrap_summary.json`)

A separate verification harness (`scripts/verify_codex_bootstrap.py`) drives scenarios (T01–T15) used in CI matrix testing.

## Workflow Dispatch Inputs
When triggering `Codex Assign Minimal` via `workflow_dispatch`, the following inputs are available:

| Input | Required | Example | Purpose |
|-------|----------|---------|---------|
| `test_issue` | false | `123` | Treat an existing issue ID as the event target (manual simulation). |
| `simulate_label` | false | `agent:codex,codex-sim:manual` | Comma‑separated labels to simulate; include `agent:codex` to force bootstrap. |
| `allow_fallback` | false | `false` | Override fallback policy (blocks if PAT absent and set to `false`). |
| `codex_command` | false | `codex: review` | Override the Codex activation command (validated). |

## Simulation Labels
Labels (real or simulated via `simulate_label`) drive enhanced test paths:

| Label | Effect |
|-------|--------|
| `agent:codex` | Primary trigger enabling bootstrap behavior. |
| `codex-sim:primary-fail` | Forces a synthetic 403 on the primary branch creation attempt. |
| `codex-sim:dual-fail` | Forces both primary and fallback branch creation failures (no artifact). |
| `codex-sim:manual` | Sets PR mode to manual (branch + marker without opening a PR). |
| `codex-sim:suppress` | Suppresses the initial activation comment on the PR. |

> Simulation labels are **case-insensitive** in detection logic.

## Command Validation & Enforcement
The composite action accepts an input `codex_command` and validates it against an allowlist:
```
codex: start
codex: test
codex: lint
codex: build
codex: review
codex: docs
```
Invalid overrides are sanitized to `codex: start` with a warning. When `fail_on_invalid_command: true` (now set by the minimal workflow), the run is marked failed if an invalid command was provided and enforcement is enabled.

Artifact fields record both the original and final command:
```
command_original: "codex: rm -rf /"
command_final: "codex: start"
invalid_command: true
```

## Fallback Policy Override
`allow_fallback=false` + missing `SERVICE_BOT_PAT` causes an early block (exit code 86) and suppresses artifact creation. This is validated by scenario T09.

## Forced Failure Inputs (Test Only)
The composite provides test-only inputs wired from detection outputs:

| Input | Source | Behavior |
|-------|--------|----------|
| `force_branch_fail` | `codex-sim:primary-fail` | Injects synthetic primary branch creation failure (status 403). |
| `force_dual_fail` | `codex-sim:dual-fail` | Suppresses fallback success path → overall branch creation fails. |

## Manual Mode
When `codex-sim:manual` is present, `pr_mode=manual`:
- Branch created
- Marker file committed
- No PR opened yet (user manually opens PR later)
- Artifact indicates `manual: true` (and no `pr` number) in `codex_bootstrap_result.json`.

## Suppression Mode
`codex-sim:suppress` suppresses the auto activation comment while still creating the draft PR (unless manual mode). The artifact includes `suppression: true`.

## Key Artifacts
| File | Purpose |
|------|---------|
| `codex_bootstrap_result.json` | Primary machine-readable result (branch, PR, flags, tokens, command info). |
| `codex_bootstrap_summary.json` | Lightweight summary written at workflow end. |
| `marker_snapshot.json` | (On reuse) Snapshot of existing marker file for comparison. |

## Important Artifact Fields
| Field | Description |
|-------|-------------|
| `reused` | `true` if previous bootstrap reused (no new branch/PR). |
| `branch_created` | `true` if branch created during this run. |
| `pr` | Draft PR number (missing/null in manual mode). |
| `fallback_used` | Indicates fallback token path taken. |
| `token_source` | Token ultimately used (`SERVICE_BOT_PAT`, `GITHUB_TOKEN`, `ACTIONS_DEFAULT_TOKEN`). |
| `suppression` | Activation comment suppression flag. |
| `command_original` | Raw command override provided. |
| `command_final` | Sanitized/allowed command used. |
| `invalid_command` | Boolean; true if override rejected. |

## Verification Scenarios (T01–T15)
| ID | Focus | Summary Pass Condition |
|----|-------|-----------------------|
| T01 | Basic bootstrap | New PR + branch, reused=false |
| T02 | Reuse existing | reused=true, no new branch |
| T03 | Re-bootstrap closed PR | New PR number differs from original |
| T04 | Missing label | No artifact |
| T05 | Manual dispatch + simulate label | Artifact with manual mode or PR depending on config |
| T06 | Manual dispatch without simulate label | No artifact |
| T07 | Invalid manual issue id | Detection handles invalid input; no artifact |
| T08 | PAT missing fallback allowed | Artifact with fallback or non-PAT token_source |
| T09 | Fallback disallowed | No artifact unless PAT present |
| T10 | Forced primary failure | Artifact, fallback_used=true |
| T11 | Forced dual fail | No artifact |
| T12 | Manual mode label on labeled issue | Artifact; pr may be absent/manual |
| T13 | Suppressed activation | Artifact; suppression flag |
| T14 | Invalid command override | Artifact with sanitized command OR no artifact (enforced failure) |
| T15 | Corrupt marker reuse | Reuse still works after marker corruption |

## Common Failure Codes
| Exit Code | Origin | Meaning |
|-----------|--------|---------|
| 75 | Network preflight | Exhausted retry attempts hitting GitHub API. |
| 86 | Token gate | PAT missing and `allow_fallback=false`. |

## Dispatch Examples
Manual simulation with fallback disallowed:
```
workflow_dispatch:
  test_issue: 123
  simulate_label: agent:codex
  allow_fallback: false
```
Force dual failure simulation:
```
workflow_dispatch:
  test_issue: 123
  simulate_label: agent:codex,codex-sim:dual-fail
```
Invalid command test (enforced):
```
workflow_dispatch:
  test_issue: 123
  simulate_label: agent:codex
  codex_command: codex: rm -rf /
```

## Extending the Matrix
When adding new behaviors:
1. Add a simulation label or dispatch input—keep production defaults untouched.
2. Expose detection output (lowercase, dashless) mirroring the input.
3. Pass through to composite as a quoted input.
4. Add scenario function + expectation in `verify_codex_bootstrap.py`.
5. Update this document.

## Safety & Hygiene
- All unsafe command overrides are sanitized; no shell metacharacters are ever executed.
- Branch naming is deterministic: `agents/codex-issue-<num>`.
- Marker files are idempotent; reuse path short-circuits heavy operations.

## Quick Triage Checklist
1. Artifact missing unexpectedly? Check detection summary (workflow job output) for reason code.
2. Fallback unexpectedly used? Inspect `token_source` + `fallback_used` flags.
3. Invalid command not flagged? Ensure `fail_on_invalid_command` is set to `true` in the workflow step.
4. Fallback block not working? Confirm dispatch included `allow_fallback=false` and PAT is truly absent.

---
Maintainers: Update this guide with any new simulation flag, artifact field, or scenario ID so CI and documentation remain aligned.
