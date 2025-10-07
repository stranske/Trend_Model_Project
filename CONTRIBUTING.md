# Contributing Guide

Thank you for contributing to the Trend Model Project.

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

Pull requests also trigger `.github/workflows/autofix.yml`, which delegates to the reusable autofix composite. When safe fixes are found it pushes a `chore(autofix): …` commit (or publishes a patch artifact for forked PRs) and tags the PR with `autofix`, `autofix:applied`, plus the relevant clean/debt label. Always fetch/rebase before adding new commits if the bot amends your branch.

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
| Black/Ruff/Mypy pinned | `pr-10-ci-python.yml` job `style` | `scripts/style_gate_local.sh` |
| Full quality (style + type + adaptive tests) | aggregate | `scripts/quality_gate.sh` |
| Workflow syntax/semantic lint | `workflow lint (actionlint)` / `actionlint` | `scripts/workflow_lint.sh` |
| Docker build + health smoke | `Docker` (smoke) | `scripts/docker_smoke.sh` |

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
