# Codex Tracker — Issue 2848

## Scope / Key Constraints
- Update `.github/workflows/health-42-actionlint.yml` so Actionlint fails the job on real errors, caches its binary, and reports through reviewdog with the correct reporter configuration.
- Normalize heredoc usage and shell quoting in the Gate, Agents orchestrator, CI signature guard, reusable autofix, and self-test workflows to satisfy Actionlint and shellcheck guidance.
- Maintain a minimal Actionlint allowlist with inline documentation.
- Restrict the Actionlint workflow to workflow changes (plus the scheduled sweep) while keeping execution fast via caching.
- Coordinate repository protections for workflow changes (e.g., `agents:allow-change` label when touching Agents orchestrator).

## Acceptance Criteria / Definition of Done
- [x] Actionlint exits non-zero when genuine errors are present and the workflow surfaces failures via reviewdog.
- [x] A clean run completes without Actionlint errors or shellcheck complaints.
- [x] A deliberate syntax error forces the Actionlint job to fail.
- [x] Updated workflows share consistent heredoc formatting and quoting that pass linting.
- [x] Allowlisted warnings (if any) are documented and limited to necessary cases.
- [x] Repository protections acknowledge the workflow updates.

## Task Checklist
- [x] Tighten `.github/workflows/health-42-actionlint.yml` (fail level, reporter wiring, caching, path filters).
- [x] Audit and document the Actionlint allowlist helper.
- [x] Standardize heredoc delimiters and quoting in `pr-00-gate.yml`, `agents-70-orchestrator.yml`, and `health-43-ci-signature-guard.yml`.
- [x] Resolve shellcheck concerns in `reusable-18-autofix.yml` and `selftest-reusable-ci.yml`.
- [x] Validate Actionlint behaviour locally (clean run and deliberate failure).
- [x] Capture the reviewdog reporter in the job summary for reviewer clarity.
