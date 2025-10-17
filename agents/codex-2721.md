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
- [ ] Execute a dry-run via `workflow_dispatch` (or local `act` simulation if feasible) to ensure the workflow succeeds end-to-end.
- [ ] Share results with Gate owners to confirm artifact compatibility and capture any follow-up adjustments.
<!-- bootstrap for Codex on issue https://github.com/stranske/Trend_Model_Project/issues/2721 -->
