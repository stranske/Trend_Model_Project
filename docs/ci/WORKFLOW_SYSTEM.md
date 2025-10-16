# Workflow System Overview

**Purpose.** Keep PR feedback fast and reliable, maintain repo hygiene automatically, let issues spawn agent work where appropriate, and automate the boring error checking (lint, types, cosmetic test fixes) without enterprise pageantry.

## Buckets and Canonical Workflows

### PR checks
- **Gate**: `.github/workflows/pr-00-gate.yml`
  Always runs on PRs. A lightweight `detect_doc_only` job classifies changes (Markdown anywhere, `docs/`, and `assets/` mirror the former PR‑14 filters). Docs-only or empty diffs short-circuit the heavy Python CI (`reusable-10-ci-python.yml`) and Docker smoke (`reusable-12-ci-docker.yml`) jobs; Gate posts the friendly skip notice and reports success immediately. Code changes continue to fan out to the full matrix, aggregate results, and return a single required status. The job explicitly requests `pull-requests: write` and `statuses: write` scopes so it can publish the comment and commit status with the correct fast-path message.
- **Autofix (optional)**: _Deprecated as a default runner_. `maint-46-post-ci.yml` handles small hygiene fixes post‑CI. `pr-02-autofix.yml` may be kept as **opt‑in** via the `autofix` label only.

### Maintenance & repo health
- **Post‑CI summary & hygiene**: `.github/workflows/maint-46-post-ci.yml` consolidates results, applies “small self‑contained” fixes safely, and updates the failure tracker.
- **Cosmetic repair helper**: `.github/workflows/maint-45-cosmetic-repair.yml` runs pytest + guardrail fixers on demand and opens a labelled PR if adjustments are needed.
- **Health checks**: repo pulse (`health-40-repo-selfcheck.yml`), weekly health sweep (`health-41-repo-health.yml`), actionlint (`health-42-actionlint.yml`), CI signature guard (`health-43-ci-signature-guard.yml`), branch‑protection verifier (`health-44-gate-branch-protection.yml`), and agents workflow guard (`health-45-agents-guard.yml`).

- **Single entry point**: `agents-70-orchestrator.yml`. All consumer
  dispatches route through the orchestrator; the Agents 61/62 shims remain
  retired and must not return.
- **Issue bridge**: `agents-63-codex-issue-bridge.yml` opens branches/PRs from `agent:codex` issues.
- **Assignment verifier**: `agents-64-verify-agent-assignment.yml` exposes the assignment audit path and feeds the orchestrator.
- **ChatGPT topic sync**: `agents-63-chatgpt-issue-sync.yml` turns curated topic files (e.g. `Issues.txt`) into labelled GitHub issues on demand.
- **Immutable guardrail**: the orchestrator and both `agents-63-*` workflows are protected by CODEOWNERS, branch protection, the `Agents Critical Guard` CI check, and a repository ruleset. See the [Agents Workflow Protection Policy](./AGENTS_POLICY.md) for guardrails, allowlisted change reasons, and override steps.
- **Issue template**: [Agent task](https://github.com/stranske/Trend_Model_Project/issues/new?template=agent_task.yml) pre-labels
  issues with `agents` and `agent:codex` so the bridge triggers immediately.
- **Consumer note**: manual shims were removed; all automation now dispatches
  the orchestrator directly. Agents 63 (bridge + ChatGPT sync) and Agents 70
  are the supported topology.

### Reusable composites
- Python CI: `reusable-10-ci-python.yml` (ruff + mypy + pytest).
- Docker CI: `reusable-12-ci-docker.yml`.
- Agents: `reusable-16-agents.yml`.
- Autofix: `reusable-18-autofix.yml`.

### Self‑tests
- `selftest-81-reusable-ci.yml` remains the reusable matrix.
- `selftest-runner.yml` is the single manual entry point. Inputs:
  - `mode`: `summary`, `comment`, or `dual-runtime` (controls reporting surface and Python matrix).
  - `post_to`: `pr-number` or `none` (comment target when `mode == comment`).
  - `enable_history`: `true` or `false` (download the verification artifact for local inspection).
  - Optional niceties for comment/summary titles plus the dispatch reason.

## Policy

- **Required checks**: Gate and Health 45 Agents Guard are required on every PR. Gate aggregates the CI matrix, while the guard runs on each PR (even when agents workflows are untouched) and only blocks when protected files change without the allow label + CODEOWNER approval.
- **Doc‑only rule**: Doc‑only detection lives inside Gate. No separate docs‑only workflow.
- **Autofix**: Centralized under `maint-46-post-ci.yml`. For forks, upload patch artifacts and post links instead of pushing. Any pre‑CI autofix (`pr-02-autofix.yml`) must be label-gated and cancel duplicate runs in flight.
- **Branch protection**: Default branch must require both **Gate / gate** and **Health 45 Agents Guard / Enforce agents workflow protections**. Health 44 resolves the default branch via the REST API, then enforces/verifies that the required status list matches that pair. Provide a `BRANCH_PROTECTION_TOKEN` secret with admin scope when you want the job to apply the setting automatically; otherwise it will fail fast when the contexts are missing.
- **Code Owner reviews**: Enable **Require review from Code Owners** on the default branch. This keeps the `agents-63-chatgpt-issue-sync.yml`, `agents-63-codex-issue-bridge.yml`, and `agents-70-orchestrator.yml` workflows gated on maintainer approval in addition to the deletion/rename rules described in the protection policy.
- **Types**: Run mypy where the config is pinned. If types are pinned to a specific version, run mypy in that leg only (to avoid stdlib stub drift across Python versions). Our `pyproject.toml` sets `python_version = "3.11"`, so `reusable-10-ci-python.yml` resolves the value directly from the file, publishes it as a step output, and guards the mypy step with `matrix.python-version == steps.mypy-pin.outputs.python-version`. Ruff and pytest still execute on every configured interpreter.
- **Labels used by automation**:  
  `workflows`, `ci`, `devops`, `docs`, `refactor`, `enhancement`, `autofix`, `priority: high|medium|low`, `risk:low`, `status: ready|in-progress`, `agents`, `agent:codex`.

## Final topology (keep vs retire)

- **Keep**: `pr-00-gate.yml`, `maint-46-post-ci.yml`, health 42/43/44/45, agents 70/63, `agents-critical-guard.yml`, reusable 10/12/16/18, `selftest-81-reusable-ci.yml`, `selftest-runner.yml`.
- **Retire**: `pr-14-docs-only.yml`, `maint-47-check-failure-tracker.yml`, the
  retired Agents 61/62 consumer workflows (removed from the Actions catalogue),
  and the legacy `selftest-*` wrappers replaced by `selftest-runner.yml`.

## Verification checklist
- Gate runs and passes on a docs‑only PR and is visible as a required check.
- Health‑45 runs on the same PR (fast path) and passes when no agent workflows are touched.
- Health‑44 confirms branch protection requires both Gate and the agents guard on the default branch.
- Health‑45 runs on the same PR (fast path) and passes when no agent workflows are touched.
- Health‑44 confirms branch protection requires both Gate and the agents guard on the default branch.
- Maint‑46 posts a single consolidated summary; autofix artifacts or commits are attached where allowed.

## Branch protection playbook

1. **Confirm the default branch.**
   - Health‑44 now emits a `Determine default branch` step that resolves the branch name through `repos.get`. No manual input is required for scheduled runs.
   - For ad-hoc verification, run `gh api repos/<owner>/<repo> --jq .default_branch` or browse the repository settings to confirm the value (currently `phase-2-dev`).
2. **Verify enforcement credentials.**
   - Create a fine-grained personal access token with `Administration: Read and write` on the repository.
   - Store it as the `BRANCH_PROTECTION_TOKEN` Actions secret. When present, Health‑44 applies the branch protection before verifying. Without it the workflow performs a read-only check, surfaces an observer-mode summary, and still fails if Gate is not yet required.
3. **Run the enforcement script locally when needed.**
   - `python tools/enforce_gate_branch_protection.py --repo <owner>/<repo> --branch <default-branch> --check` reports the current status.
   - Add `--require-strict` when you want the command to fail if the workflow token cannot confirm “Require branches to be up to date” (requires admin scope).
   - Add `--apply` to enforce the rule locally (requires admin token in `GITHUB_TOKEN`/`GH_TOKEN`). Use `--snapshot path.json` to capture before/after state for change control.
4. **Audit the result.**
   - Health‑44 uploads JSON snapshots (`enforcement.json`, `verification.json`) that mirror the script output and writes a step summary when it must fall back to observer mode.
   - In GitHub settings, confirm that both **Gate / gate** and **Health 45 Agents Guard / Enforce agents workflow protections** appear under required status checks and that “Require branches to be up to date before merging” is enabled.
5. **Trigger Health‑44 on demand.**
   - Kick a manual run with `gh workflow run "Health 44 Gate Branch Protection" --ref <default-branch>` whenever you change branch-protection settings.
   - Scheduled executions run daily at 06:00 UTC; a manual dispatch lets you confirm the fix immediately after applying it.
6. **Verify with a test PR.**
   - Open a throwaway PR against the default branch and ensure that the Checks tab shows both **Gate / gate** and **Health 45 Agents Guard / Enforce agents workflow protections** under “Required checks”.
   - Close the PR after verification to avoid polluting history.

### Recovery scenarios

- **Health‑44 fails because a required check is missing.**
  1. Confirm you have access to an admin-scoped token (see step 2 above) and re-run the workflow with the token configured.
  2. If the failure persists, run `python tools/enforce_gate_branch_protection.py --check` locally to inspect the status and `--apply` to restore both required contexts.
  3. Re-dispatch Health‑44 to record the remediation snapshots and attach them to the incident report.
- **Required check accidentally removed during testing.**
  1. Restore the branch-protection snapshot from the most recent successful Health‑44 run (download from the workflow artifact, then feed into `--apply --snapshot` to replay).
  2. Notify the on-call in `#trend-ci` so they can watch the next scheduled job for regressions.
  3. Open a short-lived PR targeting the default branch to confirm that Gate and Agents Guard are again listed as required before declaring recovery complete.
