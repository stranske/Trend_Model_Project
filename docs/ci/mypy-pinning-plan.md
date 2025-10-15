# Issue #2654 — Align MyPy execution with pinned Python version

## Scope and Key Constraints
- Update `.github/workflows/reusable-10-ci-python.yml` so the MyPy job only executes on the interpreter version pinned via `[tool.mypy].python_version` in `pyproject.toml`.
- Preserve existing coverage for `ruff` linting and `pytest` test execution across every Python version in the CI matrix.
- Introduce minimal logic to detect the pinned version directly from `pyproject.toml`; avoid duplicating the version string in multiple workflow steps.
- Ensure any new workflow logic remains compatible with GitHub-hosted runners and does not require additional secrets or self-hosted runners.
- Document the rationale for gating MyPy to the pinned interpreter within the CI workflow documentation so future changes understand the behavior.

## Acceptance Criteria / Definition of Done
1. MyPy executes only on the matrix leg whose interpreter matches the pinned version from `pyproject.toml`, and that job completes successfully.
2. All other matrix legs continue to run `ruff` linting and `pytest` tests without alteration and pass consistently.
3. The CI workflow documentation clearly explains how the pinned MyPy interpreter is detected and why only that leg runs static type checks.
4. Workflow changes include safeguards (e.g., conditional checks) to prevent MyPy from running when the pinned version is absent or mismatched, emitting clear diagnostics in such cases.

## Initial Task Checklist
- [ ] Parse the pinned MyPy Python version from `pyproject.toml` within the CI workflow and expose it as a reusable output.
- [ ] Guard the MyPy step/job so it runs solely when the current matrix interpreter matches the pinned version output.
- [ ] Verify that `ruff` and `pytest` steps remain unconditioned and still execute on all matrix legs.
- [ ] Update the workflow system documentation to describe the new MyPy gating behavior and maintenance expectations.
- [ ] Validate the modified workflow locally (via `act` or dry-run reasoning) or through CI to confirm the gating behaves as intended.
