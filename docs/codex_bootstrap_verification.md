# Codex Bootstrap Workflow – Failure Points & Verification Matrix

## 1. Failure Point Enumeration

| ID | Scenario | Trigger Condition | Expected Current Behaviour | Mitigation / Handling | Recommended Test |
|----|----------|------------------|-----------------------------|-----------------------|------------------|
| FP1 | No codex label | Issue event without `agent:codex` | detect sets needs=false; no bootstrap job | Clear reason output: issue-without-label | Remove label, re-run |
| FP2 | Manual dispatch missing simulated label | Dispatch with simulate_label not containing `agent:codex` | needs=false; reason=manual-dispatch-missing-simulated-label | Explicit reason for transparency | Dispatch test |
| FP3 | Invalid manual issue id | Non-numeric test_issue | needs=false; reason=invalid-manual-issue | Guard warns & skips | Dispatch test with `ABC` |
| FP4 | Missing PAT & fallback disabled | PAT empty, allow_fallback=false | Token gate exits (code 86) fail | Early explicit failure prevents wasted API calls | Run with CODEX_ALLOW_FALLBACK=false |
| FP5 | Primary branch create 403 | PAT attempt fails 403 | Fallback fetch attempt (if allowed) else failure | Fallback uses GITHUB_TOKEN; dual failure setFailed | Simulate restricted PAT |
| FP6 | Fallback branch create HTTP error | GITHUB_TOKEN restricted or ruleset blocks | setFailed (if fail_on_dual_branch_failure=true) | Clear guidance comment added | Temporarily restrict token perms |
| FP7 | Marker present (idempotent) | Branch+marker exist | Reuse path; codex_reused=true | Avoids duplicate PR churn | Relabel same issue |
| FP8 | Marker present but PR closed & rebootstrap enabled | Existing PR closed | New draft PR created; reused=false | Automated re-engagement | Close PR then relabel |
| FP9 | Marker present PR closed & rebootstrap disabled | Input sets rebootstrap_closed_pr=false | Reuse path still; will not create new PR | Conservative mode | Configure input override |
| FP10 | Manual mode | pr_mode=manual | Branch + marker only; no PR | Allows staged manual PR creation | Input test |
| FP11 | Activation suppression token | Issue body contains [codex-suppress-activate] | Activation comment skipped | Respect suppression directive | Add token to issue body |
| FP12 | Command validation | Unsupported codex_command | Falls back to safe default | Prevents unsafe commands | Pass invalid command |
| FP13 | Network rate-limit transient | Preflight curl fails until attempts exhausted | Retry then exit 75 | Prevents partial bootstrap w/o API budget | Temporarily lower rate limit |
| FP14 | Closed PR rebootstrap error | New PR creation fails | Warning, action continues (branch already exists) | Surface error, idempotent marker unaffected | Simulate with restricted PR creation |
| FP15 | JSON marker parse error | Corrupt marker file | Warnings; bootstrap continues using defaults | Defensive try/catch | Manually corrupt marker |

## 2. Verification Matrix

| Test Name | Inputs / Pre-state | Assertions |
|-----------|--------------------|------------|
| T01 Basic Create | New issue + label | branch_created=true; codex_reused=false; codex_pr != '' |
| T02 Idempotent Reuse | Relabel same issue | branch_created=false; codex_reused=true; codex_pr same as first |
| T03 Closed PR Rebootstrap | Close PR then relabel | new codex_pr != old; codex_reused=false; marker rebootstrap:true |
| T04 Missing Label | Issue without label | detect.reason=issue-without-label; no bootstrap job |
| T05 Manual Dispatch Simulated | dispatch test_issue=N simulate_label=agent:codex | branch_created=true; codex_pr set |
| T06 Manual Dispatch Missing Sim | dispatch test_issue=N simulate_label=foo | reason=manual-dispatch-missing-simulated-label |
| T07 Invalid Manual Issue | dispatch test_issue=ABC simulate_label=agent:codex | reason=invalid-manual-issue |
| T08 PAT Missing Fallback Allowed | Remove PAT; allow_fallback=true | branch_created=true (or failure surfaced); token_source=GITHUB_TOKEN in artifact |
| T09 PAT Missing Fallback Disabled | Remove PAT; allow_fallback=false | Job fails token gate (exit 86) |
| T10 Primary 403 + Fallback Success | Restrict PAT; fallback allowed | Primary fail log + fallback success log |
| T11 Dual Failure | Restrict both PAT & GITHUB_TOKEN branch perms | Job fails; fail_on_dual_branch_failure true |
| T12 Manual Mode | pr_mode=manual | codex_pr empty; marker mode:manual |
| T13 Suppressed Activation | Add suppression token to issue body | No activation comment on PR |
| T14 Invalid Command | codex_command=codex: unknown | Warning + default command used |
| T15 Corrupt Marker | Manually edit marker invalid JSON then relabel | Warning about parse; reuse logic still sets outputs |

## 3. Inputs Reference

| Input | Purpose | Typical Values |
|-------|---------|----------------|
| allow_fallback | Permit GITHUB_TOKEN fallback (default now true for resilience) | true / false |
| pr_mode | Manual vs auto PR creation | auto / manual |
| rebootstrap_closed_pr | Recreate PR if closed | true / false |
| fail_on_dual_branch_failure | Fail job if both branch attempts fail | true / false |
| codex_command | Initial Codex command (validated) | codex: start, codex: test |
| debug_mode | Verbose logging | true / false |

## 4. Operational Notes
- Always test create (T01) and idempotent reuse (T02) after any action.yml change.
- Run dual failure (T11) before relaxing protections to ensure fail path clarity.
- Keep artifact JSON (`codex_bootstrap_summary.json`) as source of truth for outputs vs console logs.

## 5. Suggested Future Enhancements
- (DONE) Structured JSON summary artifact (`codex_bootstrap_summary.json`).
- Add optional reviewer assignment list input.
- Implement stale PR detection (time-based rebootstrap) separate from closed state.

---
Generated as part of Codex bootstrap verification tasks.
