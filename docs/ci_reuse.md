# Reusable CI & Automation Workflows

This repository exposes three reusable GitHub Actions workflows (workflow_call) so other repos – or thin consumer workflows inside this repo – can standardise on a single CI / automation implementation.

| Reusable Workflow | File | Purpose |
| ------------------ | ---- | ------- |
| Python CI          | `.github/workflows/reuse-ci-python.yml` | Tests + coverage gate + (optional) quarantine set |
| Autofix            | `.github/workflows/reuse-autofix.yml`   | Formatting / lint autofix on PRs (opt‑in label) |
| Agents Automation  | `.github/workflows/reuse-agents.yml`    | Codex issue bootstrap + watchdog checks |

## 1. Python CI (`reuse-ci-python.yml`)
Trigger via a consumer workflow:
```yaml
# .github/workflows/ci.yml (consumer example)
name: ci
on:
  pull_request:
  push:
    branches: [ phase-2-dev, main ]

jobs:
  core:
    uses: ./.github/workflows/reuse-ci-python.yml
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

## 2. Autofix (`reuse-autofix.yml`)
Applies code formatting / lint autofixes only when an opt‑in label is present (avoids surprise pushes).

### Typical Consumer
```yaml
name: autofix
on:
  pull_request:
    types: [labeled, synchronize]

jobs:
  autofix:
    uses: ./.github/workflows/reuse-autofix.yml
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

## 3. Agents Automation (`reuse-agents.yml`)
Centralises Codex bootstrap plus optional modes (readiness probe, preflight, diagnostic, verify) and a watchdog pass to ensure agent issues / PRs remain healthy.

### Consumer Example
```yaml
name: agents-automation
on:
  schedule:
    - cron: "15 * * * *"
  workflow_dispatch:

jobs:
  agents:
    uses: ./.github/workflows/reuse-agents.yml
    with:
      issue_query: 'is:open is:issue label:codex-bootstrap'
      draft_pr: false
      enable_watchdog: true
```

### Inputs
| Name | Default | Description |
| ---- | ------- | ----------- |
| `enable_readiness` | `false` | Run agent assignability (readiness) probe. |
| `readiness_agents` | `copilot,codex` | Comma list of agent keys for readiness step. |
| `enable_preflight` | `false` | Run Codex preflight (assignability + optional command post). |
| `codex_user` | `` | Override Codex connector login. |
| `codex_command_phrase` | `` | Command phrase to post in preflight issue. |
| `enable_diagnostic` | `false` | Run bootstrap diagnostic environment / token probe. |
| `diagnostic_attempt_branch` | `false` | Attempt branch creation during diagnostic. |
| `diagnostic_dry_run` | `true` | Skip mutating branch creation unless set false. |
| `enable_verify_issue` | `false` | Verify a specific issue has an agent assignee. |
| `verify_issue_number` | `` | Issue number used when verification enabled. |
| `enable_watchdog` | `true` | Run repository watchdog sanity checks. |
| `bootstrap_issues_label` | `agent:codex` | Label used to select issues for Codex bootstrap. |
| `draft_pr` | `false` | Open PR as draft when bootstrapping. |

### Behaviour
Each mode is gated by its `enable_*` boolean input.

Order of potential jobs (independent; may run in parallel):
1. Readiness probe: GraphQL assignability + temp issue test.
2. Preflight: Codex single-issue assignability + optional command comment.
3. Diagnostic: Environment & token visibility + optional branch create attempt.
4. Verify issue: Ensures specified issue has at least one agent assignee.
5. Codex bootstrap: Finds first open issue with matching label and bootstraps PR.
6. Watchdog: Lightweight repo sanity (extensible future hook).

Only jobs with their enable flag set to `true` execute.

### Legacy Workflow Consolidation
The following legacy workflows are now superseded by input‑gated modes:
| Legacy Workflow | Replacement Mode |
| ---------------- | ---------------- |
| `agent-readiness.yml` | `enable_readiness` |
| `codex-preflight.yml` | `enable_preflight` |
| `codex-bootstrap-diagnostic.yml` | `enable_diagnostic` |
| `verify-agent-task.yml` | `enable_verify_issue` |
| `agent-watchdog.yml` | (partially) `enable_watchdog` (full parity pending) |

Deprecation headers were added; removal can occur after a stability window.

## 4. Adoption Guide (External Repos)
1. Copy the three reusable files verbatim or add this repo as a submodule / template reference.
2. Create thin consumer workflows calling each `uses: ./.github/workflows/<file>.yml`.
3. Adjust inputs to match language versions and coverage policy.
4. (Optional) Add status badges (see below).

### Example Badges
```markdown
![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)
![Autofix](https://github.com/<org>/<repo>/actions/workflows/autofix-consumer.yml/badge.svg)
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
- Avoid `pull-request-target` unless strictly necessary; current design uses standard `pull_request` to reduce attack surface.

## 7. Migration Checklist (Existing Repo)
- [ ] Identify old CI workflows to retire.
- [ ] Introduce new consumer pointing at `reuse-ci-python.yml`.
- [ ] Validate coverage gate matches previous policy.
- [ ] Add autofix consumer if policy allows automated commits.
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
Last updated: 2025-09-19
