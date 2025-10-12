# Contributing Guide

Thank you for contributing to the Trend Model Project.

## CI & Automation

Pull requests flow through a single required check and a consolidated
post-processing workflow:

- **Required check** – `Gate / gate` (defined in
  [`.github/workflows/pr-gate.yml`](.github/workflows/pr-gate.yml)) must
  pass before merges. It fans out to the Python 3.11/3.12 test lanes and
  the Docker smoke job.
- **Autofix lane** – The
  [Autofix workflow](.github/workflows/autofix.yml) runs on every
  non-draft PR event. Drafts are ignored unless you opt in by adding the
  `autofix` label; convert the PR back to draft (and drop the label) to
  pause automation, then mark it ready when you want autofix to resume.
- **Maint Post-CI follower** – When Gate finishes, the
  [`maint-post-ci.yml`](.github/workflows/maint-post-ci.yml) workflow
  posts a single PR summary comment (Gate status + coverage), attempts
  the same autofix sweep using the composite action, and files tracker
  issues when hygiene regressions persist. Treat that consolidated
  comment as the canonical health dashboard; rerun Gate or Maint
  Post-CI if you need the summary refreshed.
- **Recommended local mirrors** – Run
  [`./scripts/style_gate_local.sh`](scripts/style_gate_local.sh) for the
  exact formatter/type checks used by Gate, and
  [`./scripts/quality_gate.sh --full`](scripts/quality_gate.sh) to
  mirror the full gate (style + fast validate + branch checks) before
  requesting review.
- **Agent automation** – Scheduled (cron) and on-demand runs of
  [`agents-70-orchestrator.yml`](.github/workflows/agents-70-orchestrator.yml)
  invoke the reusable agents toolkit for readiness checks, diagnostics, and
  Codex bootstrapping. Applying the `agent:codex` label flags an issue for
  bootstrap handling in the next run; remove the label to opt out before the
  dispatcher cycles. Manual dispatch lives under **Actions → Agents 70
  Orchestrator → Run workflow** – supply `enable_bootstrap: true` and an
  optional `bootstrap_issues_label` in the inputs when seeding Codex PRs.

## Quick Checklist (Before Every Push)

1. Fast dev validation (changed files only):
   ```bash
   ./scripts/dev_check.sh --changed --fix
   ```
2. Local CI style job mirror (pinned Black/Ruff/Mypy):
   ```bash
   ./scripts/style_gate_local.sh
   ```
3. Optional combined quality gate:
   ```bash
   ./scripts/quality_gate.sh          # style + fast validate
   ./scripts/quality_gate.sh --full   # adds comprehensive branch checks
   ```
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
| Black/Ruff/Mypy pinned | `pr-gate.yml` jobs `core tests (3.11/3.12)` | `scripts/style_gate_local.sh` |
| Full quality (style + type + adaptive tests) | aggregate | `scripts/quality_gate.sh` |
| Workflow syntax/semantic lint | `workflow lint (actionlint)` / `actionlint` | `scripts/workflow_lint.sh` |
| Docker build + health smoke | `Docker` (smoke) | `scripts/docker_smoke.sh` |

### Required GitHub check

Branch protection requires the `Gate / gate` workflow to succeed on every pull request. The gate reuses Python 3.11, Python 3.12, and Docker smoke jobs, so investigate any failure in those legs before asking for review.

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
