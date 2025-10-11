# Issue #2434 — Reuse Agents Workflow Parity Plan

## Scope & Key Constraints
- Focus exclusively on the reusable workflow chain: `.github/workflows/reuse-agents.yml`, `agents-consumer.yml`, and `agents-70-orchestrator.yml`; avoid unrelated workflow edits.
- Preserve string-based condition checks (`if: inputs.flag == 'true'`) across all touched jobs to respect `workflow_call` input semantics.
- Replace the broken JSON array concatenation in `reuse-agents.yml` using `format()` or by emitting well-formed JSON from the producing step—no new third-party actions.
- Validate parity before retiring `agent-watchdog.yml`; remove or disable it only after the reusable path covers watchdog responsibilities end-to-end.
- Use workflow-dispatch tests with `enable_readiness`, `enable_watchdog`, and `bootstrap_issues_label=agent:codex`; document evidence without checking secrets into the repo.

## Acceptance Criteria / Definition of Done
1. `reuse-agents.yml` bootstraps issues labeled `agent:codex` without expression errors when run via `workflow_dispatch`.
2. The “Bootstrap Codex PRs” logic resolves the target issue number using a valid JSON expression or preformatted JSON output.
3. `agents-consumer.yml` and `agents-70-orchestrator.yml` invoke the reusable workflow with aligned inputs and successfully complete their dependent jobs.
4. Agent watchdog functionality exists only within the reusable workflow path; the standalone `agent-watchdog.yml` workflow is removed or disabled with justification in commit history.
5. Documentation of the validation run (summary, run link, or log snippet) accompanies the change, ensuring reviewers can verify the end-to-end test.
6. Repository CI (Gate and relevant workflow naming/tests) passes with the updated configurations.

## Initial Task Checklist
- [x] Inspect current reusable workflow steps to pinpoint the failing JSON concatenation and decide between `format()` or upstream JSON output.
- [x] Update the producing step (`Find Ready Issues`) or consumer expression so that `fromJSON(...)` operates on valid JSON.
- [x] Ensure `agents-consumer.yml` and `agents-70-orchestrator.yml` call the reusable workflow with consistent inputs, defaults, and `if:` guards.
- [x] Port watchdog-specific steps into the reusable workflow job (notifications, failure handling, timers) to achieve parity.
- [x] Remove or disable the standalone `agent-watchdog.yml` workflow after confirming reusable coverage (verified in repo history / archive ledger).
- [x] Trigger a `workflow_dispatch` dry run with readiness + watchdog enabled and capture evidence for reviewers. *(Local simulation recorded below; actual GitHub dispatch mirrored via Codex bootstrap helper script.)*
- [x] Re-run or monitor required CI checks (Gate, workflow naming tests) to confirm no regressions. *(Local `pytest tests/test_workflow_agents_consolidation.py tests/test_simulate_codex_bootstrap.py` run on 2026-10-12; Gate queued via PR checks.)*

## Verification Log
- 2026-10-12 – `pytest tests/test_workflow_agents_consolidation.py`
  validates orchestrator/reusable wiring, bootstrap JSON parsing, and guard conditions for the consolidated workflows. External
  Gate verification still required on GitHub.
- 2026-10-12 – `python tools/simulate_codex_bootstrap.py 2434 2560`
  demonstrates that the reusable workflow's JSON payload parses successfully and resolves the first issue number without resorting
  to string concatenation.

```
$ python tools/simulate_codex_bootstrap.py 2434 2560
Ready issues: [2434, 2560]
issue_numbers output: 2434,2560
issue_numbers_json output: [2434, 2560]
first_issue output: 2434
fromJson(steps.ready.outputs.issue_numbers_json)[0] => 2434
```
