# Agents Workflow Protection Policy

Trend Model Project is a consumer of the shared automation maintained in
`stranske/Workflows`. The local contract-critical surfaces are:

- `.github/workflows/agents-issue-intake.yml`
- `.github/workflows/agents-auto-pilot.yml`
- `.github/workflows/agents-71-codex-belt-dispatcher.yml`
- `.github/workflows/agents-72-codex-belt-worker-dispatch.yml`
- `.github/workflows/agents-72-codex-belt-worker.yml`
- `.github/workflows/agents-73-codex-belt-conveyor.yml`
- `.github/workflows/agents-80-pr-event-hub.yml`
- `.github/workflows/agents-81-gate-followups.yml`
- `.github/workflows/agents-guard.yml`

The consumer-side Agents 70 orchestrator was retired. Do not recreate it or
route operators to it.

## Change Contract

1. Fix shared behavior and consumer templates in `stranske/Workflows` first.
2. Apply the matching consumer sync or an explicitly source-matched forward
   fix here; do not introduce a consumer-only fork.
3. Use the `agents:allow-change` label and obtain the reviews required by
   CODEOWNERS and branch protection for protected workflow edits.
4. Keep automation-created pull requests ready for review. Draft state and PR
   closure are not dependency, staging, stack-order, or capacity controls.
5. Verify `isDraft=false` before handing off a created or reused automation PR.

Health 45 Agents Guard enforces the protected `agents-*.yml` surface. Emergency
bypasses require a maintainer, a dedicated PR, and restoration of the guard
immediately after the change.

## Mandatory check identity

The repo-local `.github/config/required-contexts.json` declares `Gate / gate`
and both `guard` and `Health 45 Agents Guard / guard` for delivery gating.
Health 45 Agents Guard's unnamed `guard` job
emits the latter check name; the workflow title is not a check-name prefix.
The separate legacy commit status `Health 45 Agents Guard / guard` is published
to the PR head, including label-triggered `pull_request_target` runs whose job
check belongs to the base SHA. Require both: a stale successful job must not
override a failed head status, and a successful status must not hide a failed
job. The obsolete `Health 45 Agents Guard / Enforce agents workflow
protections` context must not be restored by the branch-protection helper's
fallback defaults. This corrects identity, not the mandatory guard obligation.

Run `python -m pytest --noconftest --no-cov tests/tools/test_required_guard_context.py`
to check the declaration against the actual workflow job and its fail-on-block
step and label-event head-status binding. Both guard results must pass; a failed
guard result must still block delivery even if Gate passes. A missing head
status is not success: the current shared workflow emits that status only for
eligible agent events. A never-started status on an unlabeled delivery must be
routed through the shared workflow owner, not removed from this declaration or
replaced with the stale job result. Shared workflow changes belong
in Workflows; this repair changes only local configuration and helper defaults.
The declaration is desired policy, not proof that GitHub repository settings
currently enforce it; inspect live branch protection and active rulesets
separately before claiming repository-level enforcement.

The cross-repository outcome regression is committed here without vendoring
shared implementation. Run `node tests/tools/required_guard_outcomes.cjs
/absolute/Workflows/.github/scripts/sync_pr_merge_contract.js` against an
authenticated immutable Workflows checkout and retain its full commit/hash.
It fails if that source or the declaration is missing, and exercises nine real
classifier outcomes (both pass, plus either guard result independently failing,
cancelled, timed out, or action required). It is an explicit maintenance check,
not part of this consumer's ordinary Python CI and not a whole-source verdict.

See [Repo Ops Facts - Agent Bootstrap](ops/codex-bootstrap-facts.md) for the
operator entry point and shared-source boundary.
