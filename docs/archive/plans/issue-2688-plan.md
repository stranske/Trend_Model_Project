# Issue #2688 — Repo Health Self-Check Reliability Plan

## Scope and Key Constraints
- Focus changes on `.github/workflows/health-40-repo-selfcheck.yml`, keeping the scheduled run, optional PR comment input, and GitHub Script collection logic intact so existing automation hooks keep working.【F:.github/workflows/health-40-repo-selfcheck.yml†L1-L198】
- Maintain the minimal permission surface (`contents: read`, `issues: write`, `actions: write`) while ensuring privileged API calls are gated behind the optional `SERVICE_BOT_PAT` token to avoid failures when it is absent.【F:.github/workflows/health-40-repo-selfcheck.yml†L13-L55】【F:.github/workflows/health-40-repo-selfcheck.yml†L88-L155】
- Preserve the workflow outputs documented for operators: Markdown summary, JSON artifact, and the optional “How to fix” PR checklist so downstream runbooks remain accurate.【F:docs/repo_health_self_check.md†L43-L51】
- Document behavioural changes within the CI workflow guide and self-check reference so the “green” definition and remediation guidance stay aligned with the automation surface.【F:docs/ci/WORKFLOW_SYSTEM.md†L12-L18】【F:docs/repo_health_self_check.md†L43-L51】

## Acceptance Criteria / Definition of Done
1. The workflow executes with only the minimal permissions and succeeds without a PAT by downgrading permission-related gaps to warnings rather than hard failures.【F:.github/workflows/health-40-repo-selfcheck.yml†L13-L198】
2. Error-level signals (for example, missing default-branch protection) still fail the job, but label lookup or privilege gaps emit warnings, contribute to the JSON artifact, and appear in the “How to fix” checklist.【F:.github/workflows/health-40-repo-selfcheck.yml†L157-L198】【F:docs/ci/WORKFLOW_SYSTEM.md†L12-L18】
3. The workflow publishes a machine-consumable JSON artifact capturing check outcomes and, when dispatched against a PR, posts or updates a checklist comment that clearly maps each warning/error to remediation steps.【F:.github/workflows/health-40-repo-selfcheck.yml†L157-L198】【F:docs/repo_health_self_check.md†L43-L51】
4. Documentation in `docs/ci/WORKFLOW_SYSTEM.md` and `docs/repo_health_self_check.md` explains the updated pass criteria, artifact contents, and checklist behaviour so operators understand what “green” represents and how to respond to warnings.【F:docs/ci/WORKFLOW_SYSTEM.md†L12-L18】【F:docs/repo_health_self_check.md†L43-L51】

## Initial Task Checklist
- [x] Audit workflow permissions and token handling to confirm the job can operate with only `contents: read`, `issues: write`, and `actions: write`, gracefully skipping privileged checks when no PAT is provided.【F:.github/workflows/health-40-repo-selfcheck.yml†L13-L155】
- [x] Review each health signal and ensure non-critical findings emit warnings, aggregate into the JSON artifact, and populate the remediation checklist without failing the run.【F:.github/workflows/health-40-repo-selfcheck.yml†L157-L198】
- [x] Validate that the workflow uploads the JSON artifact and posts/updates the PR checklist comment when `pull_request_number` is supplied, covering both warning-only and error scenarios.【F:.github/workflows/health-40-repo-selfcheck.yml†L7-L55】【F:.github/workflows/health-40-repo-selfcheck.yml†L157-L198】
- [x] Update the operator documentation to reflect the refined success criteria, artifact schema, and checklist expectations, then secure review from the CI maintainers for accuracy.【F:docs/ci/WORKFLOW_SYSTEM.md†L12-L18】【F:docs/repo_health_self_check.md†L43-L51】
