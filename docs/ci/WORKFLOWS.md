# CI Workflow Layout

This page captures the target layout for the automation that protects pull requests, heals small issues, and keeps the repository health checks aligned. Each section links directly to the workflow definitions so future changes can trace how the pieces fit together.

```
pull_request ──▶ Gate ──▶ Maint Post-CI summary
                    │              │
                    │              └─▶ Autofix / failure tracking (conditional)
                    └─▶ Reusable test suites (Python matrix & Docker smoke)
```

## Pull Request Gate

* [`Gate`](../../.github/workflows/pr-00-gate.yml) orchestrates the fast-path vs full CI decision, evaluates coverage artifacts, and reports commit status back to the PR.
* [`Reusable CI (Python)`](../../.github/workflows/reusable-10-ci-python.yml) drives the primary test matrix (lint, type-check, tests, coverage) for PR builds.
* [`Reusable CI (Docker)`](../../.github/workflows/reusable-12-ci-docker.yml) executes the container smoke test whenever Docker-related files change.

The gate uses the shared `.github/scripts/detect-changes.js` helper to decide when documentation-only changes can skip heavy jobs and when Docker smoke tests must run.

## Maint Post-CI Summary & Failure Handling

* [`maint-46-post-ci`](../../.github/workflows/maint-46-post-ci.yml) consumes the completed Gate run, normalises coverage artifacts with `.github/scripts/coverage-normalize.js`, publishes the consolidated PR summary, and manages failure tracker issues.
* [`maint-coverage-guard`](../../.github/workflows/maint-coverage-guard.yml) periodically verifies that the latest Gate run meets baseline coverage expectations.

The summary workflow updates its PR comment via `.github/scripts/comment-dedupe.js`, ensuring a single authoritative status thread per pull request.

## Autofix & Cosmetic Repair

* [`reusable-18-autofix`](../../.github/workflows/reusable-18-autofix.yml) provides the shared jobs used by autofix callers to stage, classify, and report automatic fixes.
* [`maint-45-cosmetic-repair`](../../.github/workflows/maint-45-cosmetic-repair.yml) invokes the reusable autofix pipeline on a schedule to keep cosmetic issues in check.
* [`maint-keepalive`](../../.github/workflows/maint-keepalive.yml) ensures Codex/autofix configuration stays fresh and pings for outstanding tasks.

## Agents Control Plane

The agent workflows coordinate Codex and chat orchestration across topics:

* [`agents-70-orchestrator`](../../.github/workflows/agents-70-orchestrator.yml) and [`agents-73-codex-belt-conveyor`](../../.github/workflows/agents-73-codex-belt-conveyor.yml) manage task distribution.
* [`agents-71-codex-belt-dispatcher`](../../.github/workflows/agents-71-codex-belt-dispatcher.yml) and [`agents-72-codex-belt-worker`](../../.github/workflows/agents-72-codex-belt-worker.yml) handle dispatching and execution.
* [`agents-guard`](../../.github/workflows/agents-guard.yml) applies repository-level guardrails before agent workflows run.

## Repository Health Checks

Scheduled health jobs keep the automation ecosystem aligned:

* [`health-40-repo-selfcheck`](../../.github/workflows/health-40-repo-selfcheck.yml) synthesises a repo-wide self-check report.
* [`health-41-repo-health`](../../.github/workflows/health-41-repo-health.yml) compiles dependency and hygiene signals.
* [`health-42-actionlint`](../../.github/workflows/health-42-actionlint.yml) enforces workflow syntax quality.
* [`health-43-ci-signature-guard`](../../.github/workflows/health-43-ci-signature-guard.yml) verifies signed workflow runs when required.
* [`health-44-gate-branch-protection`](../../.github/workflows/health-44-gate-branch-protection.yml) ensures branch protection stays aligned with Gate expectations.

Together these workflows define the CI surface area referenced by the Gate and Maint Post-CI jobs, keeping the automation stack observable, testable, and easier to evolve.
