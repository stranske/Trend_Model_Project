# Keepalive Status — PR #3790

> **Status:** Completed — fork-safe checkout hardening and resiliency updates landed.

## Scope
- [x] Audit workflow jobs that read PR head SHAs (Gate summary helpers, orchestrator workflow_run handlers, belt conveyor/meta workflows) and ensure checkouts target the contributing repository when available.
- [x] Add fallback protections so workflow_run events gracefully handle missing pull_request metadata or inaccessible SHAs instead of aborting.

## Tasks
- [x] Identify every actions/checkout step that pulls a PR head SHA and add repository overrides that prefer pull_request.head.repo.full_name (or equivalent) with safe fallbacks to github.repository.
- [x] Update workflow_run handlers to skip or warn when head metadata is absent rather than failing the run, keeping automation resilient across event types.
- [x] Add lightweight validation (unit tests or dry-run jobs) demonstrating forked PR head checkouts succeed for the audited workflows.

## Acceptance criteria
- [x] All PR-head checkouts succeed on forked contributions across Gate/orchestrator/meta workflows, with no missing-commit errors.
- [x] workflow_run jobs log warnings instead of failing when head data is unavailable, and default to the base repo safely.
- [x] New coverage confirms fork-aware repository selection is exercised in CI.
