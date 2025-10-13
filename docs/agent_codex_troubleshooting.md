# Codex Agent Labeling & Bootstrap Troubleshooting

> Related: For CI failure tracker environment variable reference (cooldown, escalation, signature hashing) see `failure_tracker_env.md` in this directory.

This guide helps diagnose why labeling an issue with `agent:codex` did **not** create (or update) a Codex bootstrap branch / draft PR.

## Quick Expectations

| Action | Expected Result |
|--------|-----------------| 
| Add label `agent:codex` to issue | `Assign to Agent` workflow runs: `assign_or_backfill` job sets `needs_codex_bootstrap=true` |
| Same run | `codex_bootstrap` job starts (needs == true) |
| Job output | Branch `agents/codex-issue-<n>` created; draft PR (auto mode) or comment w/ manual instructions (manual mode) |
| Re-label issue | Workflow runs again; detects marker & reuses state (no duplicate PR) |

## Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `codex_bootstrap` job skipped | `needs_codex_bootstrap` output was `false` | Ensure label exactly `agent:codex` (lowercase) and event type is issue label (not PR) |
| Error: cannot create branch | GITHUB_TOKEN lacks `contents: write` in repo settings | Enable ACTIONS permissions: Allow write, or configure `SERVICE_BOT_PAT` |
| Exit code 86 early in job | PAT missing & fallback disallowed | Add `SERVICE_BOT_PAT` secret or set repo/org var `CODEX_ALLOW_FALLBACK=true` |
| PR created but not assigned to Codex | Bot not assignable / missing permission | Manually add `chatgpt-codex-connector` and automation companion user |
| Activation comment missing | Suppression token present or suppression env/input set | Remove `[codex-suppress-activate]` or unset suppression vars |
| Marker exists but PR missing | Marker written in manual mode before PR made | Open a draft PR from the branch; re-run labeling |

## Token & Permission Matrix

| Scenario | SERVICE_BOT_PAT | Behavior |
|----------|-----------------|----------|
| PAT present (recommended) | Yes | Human-authored comments & branch creation via PAT |
| PAT absent (fallback allowed) | No | Falls back to `GITHUB_TOKEN` (PR authored by github-actions[bot]) |
| PAT absent (fallback disallowed) | No + `CODEX_ALLOW_FALLBACK` unset/false | Bootstrap fails early (exit 86); add PAT or enable fallback |
| PAT present but PR author is `github-actions[bot]` | Yes (mismatch) | Logged warning; enable full repo access for the PAT |

Minimum scopes for PAT: `repo` (contents + issues + pull requests). Optional: `workflow` if you later add dynamic dispatch.

## Manual Mode vs Auto Mode

`CODEX_PR_MODE=manual` (org/repo var or workflow_dispatch input):
- Branch + marker created
- No PR automatically opened
- Comment instructs user to open draft PR

`auto` (default):
- Draft PR created or reused
- Body mirrors issue contents
- Labels + assignment + activation comment applied (unless suppressed)

## Suppression Resolution

Activation is suppressed if ANY of:
- Issue body contains `[codex-suppress-activate]`
- Repo/org var `CODEX_SUPPRESS_ACTIVATE` set
- Workflow input `suppress_activate` set to true

## Reproduction Checklist

1. Create a test issue: "Test Codex bootstrap".
2. Add label `agent:codex`.
3. Open workflow run logs: confirm two jobs executed.
4. Verify branch `agents/codex-issue-<n>` pushed.
5. (Auto mode) Confirm draft PR opened with body header `### Source Issue #<n>`.
6. Re-run: remove label, add again → second run logs `reused: true` in summary.
7. (Fallback test) Remove PAT, set `CODEX_ALLOW_FALLBACK=true`, re-label → bootstrap proceeds with warning.
8. (Gating test) Remove PAT, unset fallback → run fails at preflight token gate (exit 86).

## Verification Workflow (Optional)

You can manually verify a bootstrap via the `Verify Codex Bootstrap` workflow:

1. Navigate to Actions → `Verify Codex Bootstrap`.
2. Provide:
	- `issue`: the issue number you labeled with `agent:codex`.
	- `expect_pr`: `true` (default) unless in manual mode.
3. Run workflow → Summary table lists branch, PR, marker presence.
4. Fails fast if branch missing or expected PR absent.

Use this when debugging uncertain bootstrap behavior without re-triggering labels.

## Debugging A Failed Run

Collect these from the run summary:
- `needs_codex_bootstrap`
- `codex_issue`
- Any warnings about PAT / token mismatch
- Presence of `CODEX_BOOTSTRAP_RESULT:` line in logs
 - Exit code 86 indicates PAT gating prevented bootstrap; confirm secret or fallback variable.

If missing, the initial job likely short‑circuited before setting outputs.

## Log Search Strings

Use these in workflow logs:
- `CODEX_BOOTSTRAP_RESULT:` (structured JSON line)
- `Marker exists – bootstrap already performed.`
- `Token mismatch – SERVICE_BOT_PAT` (permission misconfiguration)
- `Codex bootstrap blocked: unable to create branch` (contents permission)
 - `SERVICE_BOT_PAT missing – proceeding with GITHUB_TOKEN` (fallback path)
 - `exit 86` (token gating fast-fail)

## Safe Local Experiments

Trigger a dry run via workflow dispatch with inputs:
```text
scope: issues
agents: codex
codex_pr_mode: manual
```
Then label an issue and compare manual vs auto sequences.

### Network Preflight Retry
Configure (repo/org variables) to mitigate transient API/network hiccups before bootstrap side-effects:

| Variable | Example | Effect |
|----------|---------|--------|
| `CODEX_NET_RETRY_ATTEMPTS` | 3 | Number of lightweight `/rate_limit` probes before proceeding |
| `CODEX_NET_RETRY_DELAY_S` | 3 | Seconds between probes |

Failures after preflight indicate persistent issues (permissions, logic) rather than transient connectivity.

## Future Hardening Ideas

- Use `agents-70-orchestrator.yml` (manual `workflow_dispatch` or wait for the 20-minute schedule) to replay bootstrap or run targeted diagnostics against branch + marker invariants.
- Emit a machine-readable JSON summary comment (reaction toggles rerun).
- Add metrics export (counts of reused vs new bootstraps) to an org dashboard.

---
*Last updated: 2026-10-12 (Issue #2190 cleanup)*
