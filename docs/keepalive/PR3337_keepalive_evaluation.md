# Keepalive Evaluation – PR 3337 (2025-11-07)

This report assesses the keepalive automation on pull request [#3337](https://github.com/stranske/Trend_Model_Project/pull/3337)
against the requirements laid out in:

- `docs/keepalive/GoalsAndPlumbing.md`
- `docs/keepalive/SyncChecklist.md`
- `docs/keepalive/Agents.md`

Evidence sources include GitHub Actions runs `19156998987`, `19157730366`, `19157741929`, Gate run `19157753285`, and the
public PR comment stream (`gh pr view 3337 --comments`).

## 1. Activation Guardrails (Goals & Plumbing §1)

| Requirement | Evidence | Status |
| --- | --- | --- |
| `agents:keepalive` label present | `gh pr view 3337 --json labels` lists `agent:codex`, `agents:activated`, `autofix`, `from:codex`; **`agents:keepalive` absent** while rounds 17–27 continue posting instructions. | ❌ Violated |
| Human kickoff via @mention | Owner comment [`#3500330776`](https://github.com/stranske/Trend_Model_Project/pull/3337#issuecomment-3500330776) directly pinged `@codex` with the keepalive instructions before Round 17. | ✅ |
| Gate success at current head | Orchestrator run `19156998987` (Round 26) step “Keepalive gate satisfied; proceeding with instruction” confirms the gate check passed prior to posting. | ✅ |

**Impact:** Missing `agents:keepalive` means every subsequent round is running while the activation contract is broken.

## 2. Repeat Contract (Goals & Plumbing §2)

Each orchestrator invocation re-executes the guardrails and run-cap logic. Because the required label never reappears, the
contract is technically violated on every repeat round even though the gate check succeeds.

❌ Overall
## 3. Run Cap Enforcement (Goals & Plumbing §3)

No overlapping orchestrator traces were observed (single trace per run, e.g. `mhoa8vmzmlxru9`, `mho9ztzhag89fl`). There is no
positive proof of enforcement, but no cap breach evidence either. | ⚠️ No breach observed; continue monitoring.

## 4. Pause & Stop Controls (Goals & Plumbing §4)

`agents:pause` is absent and the workflow did not inject extra comments when guardrails were missing. However, automation should
have halted once the `agents:keepalive` label disappeared. | ❌ Respect pause contract by reinstating the label before continuing.

## 5. No-Noise Policy (Goals & Plumbing §5)

Guardrail failures (e.g. dispatch error) produce run-summary notices only; no superfluous PR comments were emitted. | ✅

## 6. Instruction Comment Contract (Goals & Plumbing §6)

Instruction comment [`#3500415982`](https://github.com/stranske/Trend_Model_Project/pull/3337#issuecomment-3500415982) includes
all required hidden markers, was authored by `stranske`, and Round 26 logs confirm 👀/🚀 acknowledgements. | ✅

## 7. Detection & Dispatch Flow (Goals & Plumbing §7)

- PR-meta reacts to instruction comments (`Keepalive 27 … worker success` etc.).
- The orchestrator dispatch succeeds.
- The connector dispatch for `update-branch` fails with `github.getOctokit is not available for keepalive dispatch` (run
	`19156998987`, job `54759658383`), so the primary branch-update attempt never fires.

Status: ⚠️ Partial — restore Octokit credentials so `action: "update-branch"` dispatches.

## 8. Branch-Sync Gate (Goals & Plumbing §8 / Sync Checklist)

| Step | Checklist Expectation | Evidence | Status |
| ---- | --------------------- | -------- | ------ |
| 1. Snapshot before instruction | Record head/base before commenting. | Run `19156998987`, step “Capture keepalive head snapshot” stored `head=839aec9c79e6`, `head_ref=codex/issue-3333`, `base_ref=phase-2-dev`. | ✅ |
| 2. Short poll (≤120 s) | Poll head after worker success. | Comment [`#3500385963`](https://github.com/stranske/Trend_Model_Project/pull/3337#issuecomment-3500385963) notes “ack not observed in TTL,” confirming the short poll expired. | ✅ |
| 3. Update-branch dispatch | Emit `action: update-branch`. | Job `54759658383` aborted with `github.getOctokit is not available…`; no dispatch sent. | ❌ |
| 4. Create-pr fallback | Dispatch `action: create-pr`, auto-merge connector PR. | Comment [`#3500386425`](https://github.com/stranske/Trend_Model_Project/pull/3337#issuecomment-3500386425) records worker success with new head `839aec9c79e6`; automation summary [`#3500415258`](https://github.com/stranske/Trend_Model_Project/pull/3337#issuecomment-3500415258) lists the connector merge. | ✅ |
| 5. Escalate when stale | Apply `agents:sync-required` + escalation comment if branch still unchanged. | Label absent and branch advanced, so escalation not needed. | ✅ |

## 9. Orchestrator Invariants (Goals & Plumbing §9)

Runs exit gracefully on failures, logging the missing Octokit warning without cancelling in-progress work. Assignee hygiene is
respected (logs explicitly skip non-human assignees). | ✅

## 10. Restart & Success Conditions (Goals & Plumbing §10)

Keepalive remains active because tasks/acceptance criteria remain unchecked. Once the label issue is fixed and tests are green,
workflow can stand down as designed. | ⚠️ Pending completion.

## Connector Payload Contract (Sync Checklist)

- `update-branch` dispatch: **not sent** because Octokit was unavailable. ❌
- `create-pr` dispatch: implied via successful fallback merge with trace `mho9ztzhag89fl`. ✅

## Pause Label Clearance

New commits landed (`839aec9c79e6 → 3f189062f521 → 666fd30a0f72`), so the pause label was never applied and did not require
manual clearance. ✅

## Outstanding Actions

1. **Restore `agents:keepalive` label** (or update the contract) before running additional rounds; current behaviour breaches
	 Goals §1–2.
2. Provide a usable token/Octokit instance to `scripts/keepalive-runner.js` so `action: "update-branch"` dispatches succeed.
3. Address Gate failures on head `666fd30a0f72` (`python ci / python 3.11` and `python 3.12`) to let keepalive close out the task list.

## Remediation Plan

1. **Reinstate Guardrail Labels**
	- Confirm PR #3337 requirements with stakeholders and reapply `agents:keepalive` immediately.
	- Update the orchestrator playbook so label removal triggers an explicit pause comment and halts future rounds until restored.
	- Add a unit test (or workflow check) ensuring instruction emission aborts if the label is missing.

2. **Restore Octokit Dispatch Capability**
	- Investigate the credential wiring for `scripts/keepalive-runner.js`; recent logs show Octokit is `None` when `update-branch` fires.
	- Patch the workflow to inject a valid `GITHUB_TOKEN` or alternate app token into the runner environment and document the requirement in `docs/keepalive/GoalsAndPlumbing.md`.
	- Re-run the keepalive orchestrator in dry-run mode to verify `action: "update-branch"` payload emits successfully before re-enabling production rounds.

3. **Validate Branch-Sync on Staging PR**
	- Once the dry-run orchestrator pass succeeds, open (or reuse) a staging pull request that mirrors the production target branch.
	- Dispatch the `agents-keepalive-branch-sync.yml` workflow against the staging PR using the successful dry-run trace metadata so the update-branch path is exercised end-to-end.
	- Confirm the staging PR head advances (or the branch-sync workflow reports success) before re-enabling production rounds; capture the run link in the recovery log.

4. **Stabilise Gate Failures**
	- Pull the failing workflow artifacts for `python ci / python 3.11` and `python 3.12` on commit `666fd30a0f72`; catalogue test and lint errors.
	- Land fixes on `phase-2-dev`, re-run the full gate (`./scripts/check_branch.sh --fast --fix` locally, then GitHub Actions) until both jobs pass.
	- After successful gates, trigger a fresh keepalive cycle to confirm the automation completes without nudges.

5. **Regression Safeguards**
	- Add a synthetic workflow check that fails if the required guardrail labels are absent during keepalive dispatch.
	- Extend the keepalive integration tests to assert both primary and fallback dispatches succeed when credentials are available and fail loudly when not.

## Execution Notes (2025-11-07)

- Label audit: `gh pr view 3337 --json labels` confirms `agents:keepalive` remains absent; manual intervention still required.
- Octokit remediation: added a constructor fallback in `scripts/keepalive-runner.js` so the dispatcher can build an authenticated Octokit even when `github.getOctokit` is unavailable in `actions/github-script`.
- Gate diagnostics: inspected Gate run `19157753285`; both Python 3.11/3.12 jobs hit the `Auto-fix missing dependencies` step, which reported missing stdlib module `gc`. Patched `scripts/sync_test_dependencies.py` to whitelist `gc` as stdlib; need a follow-up Gate run to confirm the fix.
- Guardrail enforcement: updated `scripts/keepalive-runner.js` to flag missing `agents:keepalive` (and peer labels) as a hard failure when a keepalive sentinel is active, preventing further rounds until the label is restored.
- Regression coverage: added `tests/fixtures/keepalive/missing_label.json` and `test_keepalive_fails_when_required_labels_missing` to ensure guardrail violations fail the sweep and surface in the step summary.
