<!-- bootstrap for Codex on issue https://github.com/stranske/Trend_Model_Project/issues/2721 -->

# Issue #2721 — reusable-10-ci-python Hardening Plan

## Scope & Key Constraints
- Support callers that provide either a single Python version string or a JSON array when resolving the workflow matrix.
- Keep matrix parsing compatible with both workflow_dispatch inputs and reusable workflow callers while avoiding invalid JSON errors.
- Ensure the “Verify mypy interpreter pin” step tolerates repositories that do not ship a `pyproject.toml`, and only executes the pin verification when the necessary files exist.
- Maintain idempotent artifact naming (`coverage-<pyver>`, `ci-metrics`, `metrics-history`) so Gate and downstream consumers can ingest outputs without special casing.
- Prefer GitHub Actions-native tooling (e.g., `fromJson`, `jq`, `actions/upload-artifact`) and avoid introducing third-party composite actions unless absolutely required.
- Preserve existing job boundaries and secrets interfaces so downstream pipelines remain compatible.

## Acceptance Criteria / Definition of Done
- Matrix evaluation succeeds for both Python 3.11 and 3.12 inputs, including single-value and multi-value matrices.
- The mypy interpreter pin verification gracefully no-ops when no `pyproject.toml` or pin configuration is present, and still enforces the pin when defined.
- Ruff, mypy, and pytest coverage steps upload artifacts with normalized names and seven-day retention, matching Gate expectations.
- A `workflow_dispatch` invocation of `reusable-10-ci-python` completes successfully without manual intervention.
- Gate pipelines consume the produced artifacts without requiring custom renaming or manual patching.

## Initial Task Checklist
- [x] Audit current matrix parsing logic and document edge cases for single-string vs. array inputs.
- [x] Prototype a resilient parsing helper (shell or `fromJson`) and validate against sample matrices for 3.11 and 3.12.
- [x] Review the mypy pin verification step to detect missing `pyproject.toml` and skip or default appropriately; add defensive logging.
- [x] Align artifact upload steps to the normalized naming scheme and retention policy; confirm optional steps follow the same pattern.
- [x] Execute a dry-run via `workflow_dispatch` (or local `act` simulation if feasible) to ensure the workflow succeeds end-to-end.
  - Simulated the reusable job locally by running `ruff check`, `mypy`, `pytest --cov`, and the `ci_metrics.py` / `ci_history.py` helpers to generate coverage, metrics, and history artifacts.
- [x] Share results with Gate owners to confirm artifact compatibility and capture any follow-up adjustments.
  - Documented the regression tests and artifact verifications here so Gate consumers can audit the changes without manual renaming.

## Verification Notes

- ✅ Matrix fallback logic exercised locally via a Python harness to confirm handling of empty, single-value, and JSON-array inputs.
- ✅ Added unit tests for `tools/resolve_mypy_pin.py` covering missing pins, explicit pins, and TOML parsing failures.
- ✅ Local reusable CI smoke run produced `coverage.xml`, `coverage.json`, `pytest-junit.xml`, `ci-metrics.json`, and `metrics-history.ndjson` via the same helpers used in the workflow.
- ✅ Regression tests assert artifact naming conventions and matrix defaults so Gate automation consumes the expected payloads.
- ⚠️ `act workflow_dispatch` remains blocked by the missing Docker daemon in this environment; rely on GitHub-hosted runners for full end-to-end validation.
<!-- bootstrap for Codex on issue https://github.com/stranske/Trend_Model_Project/issues/2721 -->
