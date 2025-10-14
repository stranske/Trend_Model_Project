# Workflow System Overview

**Purpose.** Keep PR feedback fast and reliable, maintain repo hygiene automatically, let issues spawn agent work where appropriate, and automate the boring error checking (lint, types, cosmetic test fixes) without enterprise pageantry.

## Buckets and Canonical Workflows

### PR checks
- **Gate**: `.github/workflows/pr-00-gate.yml`  
  Always runs on PRs. Detects doc-only changes and fast‑passes. Fans out to Python CI (`reusable-10-ci-python.yml`) and Docker smoke (`reusable-12-ci-docker.yml`), aggregates results, and returns a single required status.
- **Autofix (optional)**: _Deprecated as a default runner_. `maint-46-post-ci.yml` handles small hygiene fixes post‑CI. `pr-02-autofix.yml` may be kept as **opt‑in** via the `autofix` label only.

### Maintenance & repo health
- **Post‑CI summary & hygiene**: `.github/workflows/maint-46-post-ci.yml` consolidates results, applies “small self‑contained” fixes safely, and updates the failure tracker.
- **Health checks**: actionlint (`health-42-actionlint.yml`), CI signature guard (`health-43-ci-signature-guard.yml`), and branch‑protection verifier (`health-44-gate-branch-protection.yml`).

### Agents & Issues
- **Single entry point**: `agents-70-orchestrator.yml`.  
- **Issue bridge**: `agents-63-codex-issue-bridge.yml` opens branches/PRs from `agent:codex` issues.  
- **Deprecated shims**: `agents-61-consumer-compat.yml`, `agents-62-consumer.yml` slated for removal.

### Reusable composites
- Python CI: `reusable-10-ci-python.yml` (ruff + mypy + pytest).
- Docker CI: `reusable-12-ci-docker.yml`.
- Agents: `reusable-16-agents.yml`.
- Autofix: `reusable-18-autofix.yml`.

### Self‑tests
- Keep `selftest-81-reusable-ci.yml` as the only matrix. Replace the multiple wrappers with **one** parameterized `selftest-runner.yml` (inputs for posting behavior and runtime selection).

## Policy

- **Required check**: Gate is the only required PR check. It must always produce a status, including docs‑only PRs (fast no‑op).
- **Doc‑only rule**: Doc‑only detection lives inside Gate. No separate docs‑only workflow.
- **Autofix**: Centralized under `maint-46-post-ci.yml`. For forks, the workflow uploads a `.patch` artifact and links it in the
  status comment instead of attempting to push. `pr-02-autofix.yml` survives only as an opt-in runner behind the `autofix` label
  with its own concurrency guard.
- **Branch protection**: Default branch must require Gate. Health job enforces and/or verifies.
- **Types**: Run mypy where the config is pinned. If types are pinned to a specific version, run mypy in that leg only (to avoid stdlib stub drift across Python versions).
- **Labels used by automation**:  
  `workflows`, `ci`, `devops`, `docs`, `refactor`, `enhancement`, `autofix`, `priority: high|medium|low`, `risk:low`, `status: ready|in-progress`, `agents`, `agent:codex`.

## Final topology (keep vs retire)

- **Keep**: `pr-00-gate.yml`, `maint-46-post-ci.yml`, health 42/43/44, agents 70/63, reusable 10/12/16/18, `selftest-81-reusable-ci.yml`.
- **Retire**: `pr-14-docs-only.yml`, `maint-47-check-failure-tracker.yml`, agents 61/62, and all `selftest-*` wrappers except 81.

## Verification checklist
- Gate runs and passes on a docs‑only PR and is visible as the required check.
- Health‑44 confirms branch protection requires Gate on the default branch.
- Maint‑46 posts a single consolidated summary; autofix artifacts or commits are attached where allowed.
