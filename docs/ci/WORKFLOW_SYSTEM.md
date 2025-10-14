# Workflow System Overview

**Purpose.** Keep PR feedback fast and reliable, maintain repo hygiene automatically, let issues spawn agent work where appropriate, and automate the boring error checking (lint, types, cosmetic test fixes) without enterprise pageantry.

## Buckets and Canonical Workflows

### PR checks
- **Gate**: `.github/workflows/pr-00-gate.yml`
  Always runs on PRs. A lightweight `detect_doc_only` job classifies changes (Markdown anywhere, `docs/`, and `assets/` mirror the former PR‑14 filters). Docs-only or empty diffs short-circuit the heavy Python CI (`reusable-10-ci-python.yml`) and Docker smoke (`reusable-12-ci-docker.yml`) jobs; Gate posts the friendly skip notice and reports success immediately. Code changes continue to fan out to the full matrix, aggregate results, and return a single required status. The job explicitly requests `pull-requests: write` and `statuses: write` scopes so it can publish the comment and commit status with the correct fast-path message.
- **Autofix (optional)**: _Deprecated as a default runner_. `maint-46-post-ci.yml` handles small hygiene fixes post‑CI. `pr-02-autofix.yml` may be kept as **opt‑in** via the `autofix` label only.

### Maintenance & repo health
- **Post‑CI summary & hygiene**: `.github/workflows/maint-46-post-ci.yml` consolidates results, applies “small self‑contained” fixes safely, and updates the failure tracker.
- **Health checks**: actionlint (`health-42-actionlint.yml`), CI signature guard (`health-43-ci-signature-guard.yml`), and branch‑protection verifier (`health-44-gate-branch-protection.yml`).

### Agents & Issues
- **Single entry point**: `agents-70-orchestrator.yml`.
- **Issue bridge**: `agents-63-codex-issue-bridge.yml` opens branches/PRs from `agent:codex` issues.
- **Issue template**: [Agent task](https://github.com/stranske/Trend_Model_Project/issues/new?template=agent_task.yml) pre-labels
  issues with `agents` and `agent:codex` so the bridge triggers immediately.
- **Consumer note**: manual shims were removed; all automation now dispatches the orchestrator directly.

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

- **Required check**: Gate is the only required PR check. It must always produce a status, including docs‑only PRs (fast no‑op).
- **Doc‑only rule**: Doc‑only detection lives inside Gate. No separate docs‑only workflow.
- **Autofix**: Centralized under `maint-46-post-ci.yml`. For forks, upload patch artifacts and post links instead of pushing. Any pre‑CI autofix (`pr-02-autofix.yml`) must be label-gated and cancel duplicate runs in flight.
- **Branch protection**: Default branch must require Gate. Health job first resolves the repository's current default branch via the REST API, then enforces and/or verifies that **Gate / gate** is the sole required status check. Provide a `BRANCH_PROTECTION_TOKEN` secret with admin scope when you want the job to apply the setting automatically; otherwise it will fail fast when the check is missing.
- **Types**: Run mypy where the config is pinned. If types are pinned to a specific version, run mypy in that leg only (to avoid stdlib stub drift across Python versions). Our `pyproject.toml` sets `python_version = "3.11"`, so `reusable-10-ci-python.yml` only executes mypy on the Python 3.11 matrix entry.
- **Labels used by automation**:  
  `workflows`, `ci`, `devops`, `docs`, `refactor`, `enhancement`, `autofix`, `priority: high|medium|low`, `risk:low`, `status: ready|in-progress`, `agents`, `agent:codex`.

## Final topology (keep vs retire)

- **Keep**: `pr-00-gate.yml`, `maint-46-post-ci.yml`, health 42/43/44, agents 70/63, reusable 10/12/16/18, `selftest-81-reusable-ci.yml`, `selftest-runner.yml`.
- **Retire**: `pr-14-docs-only.yml`, `maint-47-check-failure-tracker.yml`, agents 61/62, and the legacy `selftest-*` wrappers replaced by `selftest-runner.yml`.

## Verification checklist
- Gate runs and passes on a docs‑only PR and is visible as the required check.
- Health‑44 confirms branch protection requires Gate on the default branch.
- Maint‑46 posts a single consolidated summary; autofix artifacts or commits are attached where allowed.

## Branch protection playbook

1. **Confirm the default branch.**
   - Health‑44 now emits a `Determine default branch` step that resolves the branch name through `repos.get`. No manual input is required for scheduled runs.
   - For ad-hoc verification, run `gh api repos/<owner>/<repo> --jq .default_branch` or browse the repository settings to confirm the value (currently `phase-2-dev`).
2. **Verify enforcement credentials.**
   - Create a fine-grained personal access token with `Administration: Read and write` on the repository.
   - Store it as the `BRANCH_PROTECTION_TOKEN` Actions secret. When present, Health‑44 applies the branch protection before verifying. Without it the workflow performs a read-only check and fails if Gate is not yet required.
3. **Run the enforcement script locally when needed.**
   - `python tools/enforce_gate_branch_protection.py --repo <owner>/<repo> --branch <default-branch> --check` reports the current status.
   - Add `--apply` to enforce the rule locally (requires admin token in `GITHUB_TOKEN`/`GH_TOKEN`). Use `--snapshot path.json` to capture before/after state for change control.
4. **Audit the result.**
   - Health‑44 uploads JSON snapshots (`enforcement.json`, `verification.json`) that mirror the script output.
   - In GitHub settings, confirm that **Gate** is listed under required status checks and that “Require branches to be up to date before merging” is enabled.
