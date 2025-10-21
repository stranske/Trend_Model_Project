# CI Workflow Layout

This repository splits automation into four pillars so contributors know where
checks originate and how follow-up jobs relate to each run.

```text
pull request ──▶ Gate (pr-00-gate.yml)
                 │
                 ├─▶ Reusable CI matrix (reusable-10-ci-python.yml)
                 ├─▶ Docker smoke (reusable-12-ci-docker.yml)
                 └─▶ Maint Post-CI (maint-46-post-ci.yml)
                              │
                              ├─▶ Autofix flows
                              └─▶ Coverage trend + summary artifacts

agents orchestrator ─▶ queue ─▶ workers
scheduled health checks ─▶ maintenance reports
```

## Pull Request Checks

- **Gate** – [`pr-00-gate.yml`](../.github/workflows/pr-00-gate.yml) fans out to the
  Python matrix, Docker smoke, and summarises coverage results. It also emits the
  gate status commit check that blocks merges until green.
- **Reusable CI (Python)** – [`reusable-10-ci-python.yml`](../.github/workflows/reusable-10-ci-python.yml)
  enforces formatting, linting, typing, tests, and coverage uploads for each
  configured Python version.
- **Reusable CI (Docker)** – [`reusable-12-ci-docker.yml`](../.github/workflows/reusable-12-ci-docker.yml)
  runs the lightweight container build/smoke sequence Gate invokes when Docker
  files change.
- **Gate summary consumers** – `maint-46-post-ci.yml` and `maint-coverage-guard.yml`
  subscribe to Gate artifacts to refresh PR comments, coverage deltas, and
  follow-up alerts.

## Autofix Path

- **Maint Post-CI** – [`maint-46-post-ci.yml`](../.github/workflows/maint-46-post-ci.yml)
  retrieves the latest Gate run, evaluates whether autofix can execute, applies
  fixes or patches, updates the consolidated PR comment, and uploads enriched
  reports for maintainers.
- **Cosmetic repair** – [`maint-45-cosmetic-repair.yml`](../.github/workflows/maint-45-cosmetic-repair.yml)
  runs on a schedule to tidy lingering formatting issues that escaped the main
  autofix path.
- **Coverage guard** – [`maint-coverage-guard.yml`](../.github/workflows/maint-coverage-guard.yml)
  compares the most recent Gate coverage bundle against the baseline to surface
  regressions or alert labels.

## Agents Control Plane

Automation hand-off between Codex and ChatGPT lives in the agents workflows:

- **Assign / bridge** – [`agents-63-codex-issue-bridge.yml`](../.github/workflows/agents-63-codex-issue-bridge.yml) and
  [`agents-63-chatgpt-issue-sync.yml`](../.github/workflows/agents-63-chatgpt-issue-sync.yml)
  create bootstrap PRs, propagate labels, and keep issues in sync.
- **Orchestration** – [`agents-70-orchestrator.yml`](../.github/workflows/agents-70-orchestrator.yml)
  schedules work to dispatchers and workers defined in
  [`agents-71`](../.github/workflows/agents-71-codex-belt-dispatcher.yml),
  [`agents-72`](../.github/workflows/agents-72-codex-belt-worker.yml), and
  [`agents-73`](../.github/workflows/agents-73-codex-belt-conveyor.yml).
- **Guards** – [`agents-guard.yml`](../.github/workflows/agents-guard.yml) enforces
  assignment policies and ensures the correct automation account owns the task.

## Health Checks & Maintenance

- **Repository self-check** – [`health-40-repo-selfcheck.yml`](../.github/workflows/health-40-repo-selfcheck.yml)
  validates workflow permissions, PAT availability, and scheduled entrypoints.
- **Repo health summary** – [`health-41-repo-health.yml`](../.github/workflows/health-41-repo-health.yml)
  publishes project metrics and status tables to the repository README summary.
- **Actionlint** – [`health-42-actionlint.yml`](../.github/workflows/health-42-actionlint.yml)
  runs static analysis across all workflows nightly.
- **Signature & branch guards** – [`health-43-ci-signature-guard.yml`](../.github/workflows/health-43-ci-signature-guard.yml)
  and [`health-44-gate-branch-protection.yml`](../.github/workflows/health-44-gate-branch-protection.yml)
  confirm required protections stay active and signatures remain valid.

Refer to [`docs/ci/WORKFLOW_SYSTEM.md`](WORKFLOW_SYSTEM.md) for deeper historical
context; this page is the quick reference for the current wiring.
