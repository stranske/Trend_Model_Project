# Contributing Guide

Thank you for contributing to the Trend Model Project.

Before diving into automation specifics, read the [Workflow System Overview](docs/ci/WORKFLOW_SYSTEM.md). It captures the
required merge policy, keep vs retire roster, and how Gate, Maint 46, and the agents orchestrator collaborate. Refer back to the
[workflow catalog](docs/ci/WORKFLOWS.md) when you need per-workflow triggers, permissions, or naming guidance.

## CI & Automation

Pull requests flow through a single required check and a consolidated
post-processing workflow:

- Passing the Gate check is required to merge to the default branch. Branch
  protection also enables GitHub's "Require branches to be up to date"
  toggle, so refresh your branch with the latest default-branch commits before
  merging if Gate reports staleness.
- **Required check** – `Gate / gate` (defined in
  [`.github/workflows/pr-00-gate.yml`](.github/workflows/pr-00-gate.yml)) must
  pass before merges. Branch protection blocks the default branch until
  this check succeeds; treat the gate status as the final merge blocker.
  It fans out to the Python 3.11/3.12 test lanes and the Docker smoke
  job.
- **Autofix lane** – The
  [Autofix workflow](.github/workflows/pr-02-autofix.yml) runs on every
  non-draft PR event. Drafts are ignored unless you opt in by adding the
  `autofix` label; convert the PR back to draft (and drop the label) to
  pause automation, then mark it ready when you want autofix to resume.
- **Maint 46 Post-CI follower** – When Gate finishes, the
  [`maint-46-post-ci.yml`](.github/workflows/maint-46-post-ci.yml) workflow
  posts a single PR summary comment (Gate status + coverage), attempts
  the same autofix sweep using the composite action, and files tracker
  issues when hygiene regressions persist. It also updates the rolling
  "CI failures in last 24 h" issue labelled `ci-failure` so the current
  breakages stay easy to find. Treat that consolidated comment and
  issue as the canonical health dashboards; rerun Gate or Maint 46 Post-CI
  if you need either refreshed.
- **Agent automation** – The
  [`agents-70-orchestrator.yml`](.github/workflows/agents-70-orchestrator.yml)
  workflow is the single dispatch point for scheduled Codex automation and the
  preferred manual entry path. It invokes
  [`reusable-16-agents.yml`](.github/workflows/reusable-16-agents.yml) directly
  to run readiness checks, watchdogs, and Codex bootstrapping. Applying the
  `agent:codex` label flags an issue for bootstrap handling in the next
  run; remove the label to opt out before the dispatcher cycles. Keep
  downstream automation pointed at the orchestrator so every entry route shares
  the same guardrails and permissions surface.

### Manual workflow_dispatch quickstart

- **Maintenance helper (Maint 45 Cosmetic Repair)** – Actions → **Maint 45 Cosmetic Repair** → **Run workflow**. Choose the base branch, interpreter, and whether to run in dry-run mode using the provided inputs before the workflow hydrates the cosmetic repair script.
- **Agent automation (Agents 70 Orchestrator)** – Actions → **Agents 70 Orchestrator** → **Run workflow**. Supply booleans as strings (`true`/`false`) for readiness, watchdog, bootstrap, verification, and keepalive toggles or pass an advanced payload through `options_json` when you need to flip several paths at once.
- **Verify agent assignment (Agents 64)** – Actions → **Agents 64 Verify Agent Assignment** → **Run workflow**. Provide the `issue_number` you want to audit and optional comma-separated `valid_assignees`; the workflow writes its status table to the Actions run summary and returns JSON outputs for downstream automation.

### Health self-check run summaries

The `Health 40 Repo Selfcheck` job writes its status table into the GitHub Actions run summary so you can review label coverage, branch protection, and token usage without downloading artifacts. Open the workflow run and read the generated Markdown table in the **Summary** tab; the latest status always lives there—no artifact download required.

## Manual Self-Test Workflows

Self-test workflows exist as manual examples so they do not clutter everyday CI noise. Launch them from the **Actions** tab by selecting the desired self-test, choosing **Run workflow**, and optionally editing the default "manual test" reason so future readers know why the run exists. Leave the remaining inputs at their defaults or adjust `python-versions` when chasing interpreter-specific drift. Each run writes a summary and uploads a `selftest-report` artifact that confirms the reusable CI scenarios still emit the expected inventory, so any mismatch points to a scenario or artifact contract regression.

## Quick Checklist (Before Every Push)

1. Fast dev validation (changed files only):
   ```bash
   ./scripts/dev_check.sh --changed --fix
   ```
2. Local CI style job mirror (pinned Black/Ruff/Mypy):
   ```bash
   ./scripts/style_gate_local.sh
   ```
   See [`scripts/style_gate_local.sh`](scripts/style_gate_local.sh) for the
   full sequence that mirrors the Gate style lane.
3. Optional combined quality gate:
   ```bash
   ./scripts/quality_gate.sh          # style + fast validate
   ./scripts/quality_gate.sh --full   # adds comprehensive branch checks
   ```
   The [`--full` quality gate](scripts/quality_gate.sh) matches the Gate
   workflow fan-out.
4. (First time) Install pre-push hook:
   ```bash
   ./scripts/install_pre_push_style_gate.sh
   ```
5. Run focused tests for changed areas + full suite when touching core modules:
   ```bash
   pytest -q
   ```

## Validation Tiers
| Tier | Script | Typical Time | Scope |
|------|--------|--------------|-------|
| 1 | `dev_check.sh --changed --fix` | 2–5s | Syntax/imports (changed files) |
| 2 | `validate_fast.sh --fix` | 5–30s | Adaptive (incremental or comprehensive) |
| 3 | `check_branch.sh --fast --fix` | 30–120s | Full formatting, lint, type, imports |
| Full | `run_tests.sh` | 15–25s | Entire test suite + coverage |

## Style & Type Enforcement
The CI style job pins versions in `.github/workflows/autofix-versions.env`. Always rely on `scripts/style_gate_local.sh` to avoid drift; it runs the same Black, Ruff, and mypy checks used in CI. The quality gate script also runs mypy, and pre-commit enforces mypy on `src/trend_analysis` so type regressions are caught pre-push.

### Pull Request Autofix Workflow

Every pull request triggers the `autofix` GitHub workflow. It performs a very small, deterministic cleanup pass so reviewers do not need to chase trivial nits:

- Formats Python files with `ruff format` (no Black/isort/docformatter sweep).
- Runs `ruff check --fix --unsafe-fixes` limited to `F401`, `F841`, and the safe `E1/E2/E3/E4/E7/W1/W2/W3` families.
- Re-runs `ruff check --output-format json` to record any remaining diagnostics.
- Applies the changes directly to the PR branch when the source repository matches this repo; otherwise it pushes a follow-up branch and opens an `autofix` PR.
- Updates labels on the originating PR: `autofix` always, `autofix:applied` when edits land, and either `autofix:clean` or `autofix:debt` depending on whether Ruff still reports issues.

What the workflow **will not** do:

- Introduce formatting from Black or import sorting from isort.
- Attempt broad Ruff rule families outside the codes listed above.
- Force push to contributor forks—the follow-up branch/PR flow preserves the original branch untouched.

If the workflow opens a follow-up PR, reviewers can merge that helper PR to pull in the formatting fixes and keep the original contribution focused on functional changes.

## Pre-Push Hook (Optional but Recommended)
Install once per clone:
```bash
./scripts/install_pre_push_style_gate.sh
```
Blocks pushes if style gate fails.

## Commit Hygiene
- Use conventional commit prefixes: `feat:`, `fix:`, `chore:`, `test:`, `docs:`, `refactor:`.
- Group mechanical formatting changes separately from logic changes when feasible.
- Avoid committing large binary artifacts (prefer CSV/JSON fixtures for tests).

## Tests
- Add unit tests for new pure functions (vectorised finance calculations, selectors, weighting, etc.).
- Maintain column order invariants in exported DataFrames (tests should assert exact `list(df.columns)`).
- When refactoring, keep a temporary compatibility shim (like list-like wrappers) only if tests depend on legacy access patterns—remove in a controlled follow-up.

## Performance Expectations
- Code paths in `trend_analysis` should remain vectorised (NumPy/pandas). Any per-row Python loops must include a comment justifying necessity.
- Use caching toggles (e.g. metric memoization) conservatively; default behaviour must remain deterministic.

## Export Layer Guard-Rails
- Use canonical exporters in `trend_analysis.export` only.
- Preserve sheet order and formatting (summary sheet first unless multi-period run requires per-period tabs + summary). Do not rename columns arbitrarily.

## Filing Issues / PRs
- Reference relevant design doc updates (Phase‑1 / Phase‑2 Agents files) when adding capabilities (e.g., new ranking mode, weighting class).
- Keep PRs scope-focused; large refactors should coordinate via an issue first.
- Use the [agent task issue template](.github/ISSUE_TEMPLATE/agent-task.md) when queuing Codex automation so goals, constraints, outputs, and success criteria are captured up front; the template auto-applies the `agents` and `agent:codex` labels that the orchestrator workflow expects.

## Security & Dependency Hygiene
- Avoid introducing new dependencies without discussion—prefer the existing stack (pandas, numpy, scipy, pydantic, streamlit).
- Run `pip install -e .[dev]` to sync pinned tools.

## Getting Help
Open a draft PR early for structural feedback or tag maintainers in issues. Provide reproduction steps, config snippet, and failing test name when reporting bugs.

Happy contributing!

## Typing Aliases (_typing.py)
To standardise NumPy annotations and eliminate repetitive verbose generics, the module `src/trend_analysis/_typing.py` defines float64-focused aliases:

- `FloatArray` / `VectorF` / `MatrixF`: canonical `np.ndarray` float64 shapes (shape not enforced statically)
- `AnyArray`: unconstrained ndarray when dtype/shape truly agnostic

Guidelines:
- Prefer these aliases over raw `np.ndarray` or `np.floating` unions in new code.
- Do not expand aliases unless a clear cross-module need emerges (keep surface minimal).
- If you introduce structured (e.g. 2D-specific) semantics, document them via a dedicated alias instead of inline comments.
- Avoid mixing old and new styles in the same diff—refactor locally if you touch a function signature.

Rationale: cleans mypy output, stabilises CI typing surface, and accelerates future dtype specialization (e.g., int arrays) without sweeping edits.
## Workflow & Docker Parity (Added Post Style Gate Enhancements)

To prevent CI‑only failures (workflow lint, container smoke, type drift), the following helper scripts replicate CI jobs locally:

| Purpose | CI Job | Local Script |
|---------|--------|--------------|
| Black/Ruff/Mypy pinned | `pr-00-gate.yml` jobs `core tests (3.11/3.12)` | `scripts/style_gate_local.sh` |
| Full quality (style + type + adaptive tests) | aggregate | `scripts/quality_gate.sh` |
| Workflow syntax/semantic lint | `workflow lint (actionlint)` / `actionlint` | `scripts/workflow_lint.sh` |
| Docker build + health smoke | `Docker` (smoke) | `scripts/docker_smoke.sh` |

### Required GitHub check

Branch protection requires the `Gate / gate` workflow to succeed on every pull request and enforces GitHub's "Require branches to be up to date" option. The gate reuses Python 3.11, Python 3.12, and Docker smoke jobs, so investigate any failure in those legs before asking for review.

### Pinned Mypy
The CI now pins mypy via `MYPY_VERSION` in `.github/workflows/autofix-versions.env`. Local scripts consume this env to avoid version drift. If you see differing results, ensure the env file includes the same version and re-run:
```bash
./scripts/style_gate_local.sh
```

### Running Workflow Lint Locally
```bash
./scripts/workflow_lint.sh
```
Run this before pushing when editing any file under `.github/workflows/`.

### Optional Local Docker Smoke
```bash
./scripts/docker_smoke.sh
```
Triggers a minimal build and health endpoint probe matching CI expectations. Recommended when changing the `Dockerfile`, `requirements.lock`, or health handler.

### Quality Gate Change Detection
`scripts/quality_gate.sh` auto-runs workflow lint if workflow files changed and docker smoke if `Dockerfile` / `requirements.lock` changed (requires the scripts to be present and executable).

### Pre-commit (Optional) Actionlint Hook
You may add a local repo stanza (not committed) to `.pre-commit-config.yaml`:
```yaml
   - repo: local
      hooks:
         - id: actionlint
            name: actionlint
            entry: scripts/workflow_lint.sh
            language: system
            files: ".github/workflows/.*\\.yml$"
```
This is optional to avoid slowing unrelated commits.

---
With these parity scripts in place, any style, workflow, or container issue should be discoverable pre-push.
