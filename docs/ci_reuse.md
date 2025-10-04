# Reusable CI & Automation Workflows

This repository exposes three reusable GitHub Actions workflows (workflow_call) so other repos – or thin consumer workflows inside this repo – can standardise on a single CI / automation implementation.

| Reusable Workflow | File | Purpose |
| ------------------ | ---- | ------- |
| Python CI          | `.github/workflows/reusable-ci-python.yml` | Tests + coverage gate + (optional) quarantine set |
| Autofix            | `.github/workflows/reusable-autofix.yml`   | Formatting / lint autofix on PRs (opt‑in label) |
| Agents Automation  | `.github/workflows/agents-41-assign-and-watch.yml` | Unified agent assignment, Codex bootstrap, watchdog diagnostics, stale sweep |

## 1. Python CI (`reusable-ci-python.yml`)
Trigger via a consumer workflow:
```yaml
# .github/workflows/pr-10-ci-python.yml (consumer example)
name: ci
on:
  pull_request:
  push:
    branches: [ phase-2-dev, main ]

jobs:
  core:
    uses: ./.github/workflows/reusable-ci-python.yml
    with:
      python_matrix: '"3.11"'          # JSON-ish string parsed by the workflow
      cov_min: 70                       # Coverage threshold percent
      run_quarantine: false             # Set true to also run slow / flaky set
```

### Inputs
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `python_matrix` | string | `"3.11"` | Comma or JSON-like string interpreted as matrix versions. |
| `cov_min` | number | `70` | Minimum overall coverage percentage (branch-aware). |
| `run_quarantine` | boolean | `false` | If true, runs quarantined / slow tests job. |

### Behaviour
1. Checks out repository with full history.
2. Sets up matrix Python versions.
3. Installs dependencies via `pip install -r requirements.txt` (adjust inside reusable file if pyproject lock strategy added later).
4. Runs pytest with coverage: `pytest --cov trend_analysis --cov-branch`.
5. Extracts total coverage; fails if below `cov_min`.
6. Optionally executes a quarantine job (future expansion; input present for forward compatibility).

### Required Repository Settings
None mandatory beyond standard Actions permissions. If using Codecov or artifact upload, extend consumer workflow after the `uses:` job with dependent jobs.

## 2. Autofix (`reusable-autofix.yml`)
Applies code formatting / lint autofixes only when an opt‑in label is present (avoids surprise pushes).

### Typical Consumer
```yaml
name: autofix
on:
  pull_request:
    types: [labeled, synchronize]

jobs:
  autofix:
    uses: ./.github/workflows/reusable-autofix.yml
    with:
      opt_in_label: bot:autofix
      commit_prefix: "autofix(ci):"
```

### Inputs
| Name | Default | Description |
| ---- | ------- | ----------- |
| `opt_in_label` | `bot:autofix` | Label required on the PR to activate autofix. |
| `commit_prefix` | `autofix:` | Prefix for generated commit messages. |

### Behaviour
1. Skips entirely if label not present.
2. Runs formatting & linting fix scripts (currently `./scripts/validate_fast.sh --fix`).
3. Commits changes back to the PR branch (uses GitHub token). Fork safety: logic avoids committing if permissions insufficient.

## 3. Agents Automation (`agents-41-assign-and-watch.yml`)
Issue #1662 consolidated the assigner/watchdog pair into a single orchestrator. The reusable building block still lives at `reusable-90-agents.yml`, but `agents-41-assign-and-watch.yml` now owns:

- Label-triggered assignment (via `agents-41-assign.yml` wrapper) with Codex bootstrap + fallback handling.
- Manual watchdog requests (via `agents-42-watchdog.yml` wrapper) and cross-reference polling for fresh PRs.
- A scheduled stale sweep that pings idle agents or escalates when availability probes fail.

### Trigger Strategy
- Forward label/unlabel events from `agents-41-assign.yml` into the unified workflow to keep compatibility with existing notifications.
- Manual watchdog runs call `agents-42-watchdog.yml`, which delegates to the unified workflow with `mode: watch`.
- The unified workflow schedules its own stale sweep (default every 30 minutes) and shares readiness checks through `reusable-90-agents.yml` to avoid duplicating GraphQL probes.

### Notes
- Bootstrap logic still honours PAT priority (`OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`) and posts `@codex start` on newly created PRs.
- Adjust watchdog behaviour (timeout, expected PR) by supplying overrides during manual dispatch.
- Readiness, preflight, and verification probes from `reuse-agents.yml` remain available via `reusable-90-agents.yml` (historical variants can be recovered from git history; see `ARCHIVE_WORKFLOWS.md`).

## 4. Adoption Guide (External Repos)
1. Copy the three reusable files verbatim or add this repo as a submodule / template reference.
2. Create thin consumer workflows calling each `uses: ./.github/workflows/<file>.yml`.
3. Adjust inputs to match language versions and coverage policy.
4. (Optional) Add status badges (see below).

### Example Badges
```markdown
![CI](https://github.com/<org>/<repo>/actions/workflows/pr-10-ci-python.yml/badge.svg)
![Autofix](https://github.com/<org>/<repo>/actions/workflows/maint-32-autofix.yml/badge.svg)
```

## 5. Customisation Points
| Area | How to Extend | Notes |
| ---- | ------------- | ----- |
| Dependency install | Edit reusable CI workflow to introduce caching or lockfiles | Keep interface stable (inputs) when possible. |
| Coverage tooling | Add Codecov upload job after core test job | Use `needs: core`. |
| Autofix steps | Replace script call with ruff/black invocation | Maintain exit codes; keep label gate. |
| Agents watchdog | Add steps under conditional `if: inputs.enable_watchdog == 'true'` | Avoid long-running tasks; add timeouts. |

## 6. Security & Permissions
- Minimal default `permissions: contents: read` in CI; elevate only where required (e.g. `contents: write` for autofix commits).
- Autofix now runs via `workflow_run` follower (`maint-32-autofix.yml`) and pushes with `SERVICE_BOT_PAT`; avoid falling back to `GITHUB_TOKEN`.

## 7. Migration Checklist (Existing Repo)
- [ ] Identify old CI workflows to retire.
- [ ] Introduce new consumer pointing at `reusable-ci-python.yml`.
- [ ] Validate coverage gate matches previous policy.
- [ ] Enable the autofix follower (`maint-32-autofix.yml`) if policy allows automated commits.
- [ ] Add agents consumer (if using Codex automation).
- [ ] Remove redundant workflows.

## 8. FAQ
**Q:** Why not expose these as remote reusable workflows via `owner/repo/.github/workflows/file.yml@ref`?
**A:** Keeping them in-tree eases iteration while they stabilise. Once stable, tag versions and switch `uses:` to a remote ref in downstream repos.

**Q:** How do I pass multiple Python versions?
**A:** Provide a JSON-like string: `"[\"3.11\", \"3.12\"]"`. The workflow parses it into a matrix; see file comments.

**Q:** Where is quarantine implemented?
**A:** Placeholder input now for forward compatibility; implement a second job keyed off `run_quarantine == 'true'` later.

## 9. Status
These docs accompany PR #1257 (Issue #1166). Optional future enhancements: remote versioned usage, quarantine job, extended watchdog metrics.

---
Last updated: 2026-02-15
