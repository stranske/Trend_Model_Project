# Contributing Guide

Thank you for contributing to the Trend Model Project.

## Quick Checklist (Before Every Push)

1. Fast dev validation (changed files only):
   ```bash
   ./scripts/dev_check.sh --changed --fix
   ```
2. Local Style Gate (match CI pinned Black & Ruff) + types:
   ```bash
   ./scripts/style_gate_local.sh
   python -m mypy --config-file pyproject.toml src/trend_analysis  # (or rely on quality_gate)
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
CI Style Gate pins versions in `.github/workflows/autofix-versions.env`. Always rely on `scripts/style_gate_local.sh` to avoid drift between local and CI tools. The quality gate script also runs mypy; pre-commit now enforces mypy on `src/trend_analysis` so type regressions are caught pre-push.

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
