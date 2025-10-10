# Agent Assignment Verification – Planning Notes

## 1. Scope & Key Constraints

- **Workflow form factor**: Ship as a reusable workflow so existing call sites can `workflow_call` it and a maintainer can `workflow_dispatch` it manually with minimal scaffolding.
- **Execution context**: Job must run with `issues: read` permissions only; avoid actions that implicitly demand broader scopes (e.g., GraphQL mutations, issue write operations).
- **Input focus**: Single required `issue_number` integer input. Downstream steps resolve repository context automatically via `github.repository` to keep invocation friction low.
- **Eligibility check**: Guard clause exits early unless the referenced issue currently carries the `agent:codex` label; no-op success keeps audit noise low when mis-triggered.
- **Assignee validation**: Treat `copilot` and `chatgpt-codex-connector` as the only valid automated assignees for now. Flag absence of both as a hard failure and surface human-readable guidance.
- **Output channel**: Summaries and failure detail should land both in job logs and as a markdown table via `core.summary` so reusable consumers get consistent presentation.
- **Idempotence**: Multiple dispatches against the same issue should produce identical outputs unless the issue metadata changes; avoid stateful artifacts that persist between runs.

## 2. Acceptance Criteria / Definition of Done

- Reusable workflow file exists (e.g., `.github/workflows/verify-agent-assignment.yml`) exposing a `workflow_call` interface with a required `issue_number` input.
- Manual `workflow_dispatch` entry point remains available either directly in the same file or via a thin wrapper job.
- Workflow fetches issue metadata using the GitHub REST API with only read permissions and logs the resolved title, URL, labels, and assignees.
- Run terminates early with a success summary when the target issue lacks the `agent:codex` label.
- Run fails with exit code and summary table message when neither `copilot` nor `chatgpt-codex-connector` is assigned, including explicit remediation guidance.
- On success (valid assignee present), workflow publishes a single-row markdown table summarising issue number, title, label presence, and matched assignee.
- Summary table always renders, even in failure scenarios, and links back to the inspected issue for fast triage.
- Logging clearly distinguishes API errors (e.g., 404, permissions) from logical validation failures, aiding debugging without revealing tokens.

## 3. Initial Task Checklist

1. **Draft workflow skeleton**
   - Copy minimal permissions boilerplate (`permissions: issues: read`) and define shared defaults for `workflow_call` + `workflow_dispatch`.
   - Accept `issue_number` as an integer input and normalise to string for API calls.
2. **Implement issue fetch step**
   - Use `actions/github-script` (latest) or curl to call `GET /repos/{owner}/{repo}/issues/{issue_number}`.
   - Parse response for labels and assignees; handle 404 / permission errors with `core.setFailed`.
3. **Add validation logic**
   - Check label presence (`agent:codex` case-insensitive) and bail out successfully if absent.
   - Verify required assignee list; fail with descriptive error when both are missing.
4. **Generate summary output**
   - Produce markdown table with columns: Issue, Title, Has Label?, Valid Assignee? / Name.
   - Ensure summaries render regardless of pass/fail by wrapping in `finally` or sequential steps.
5. **Document invocation examples**
   - Update `docs/agent-automation.md` (or relevant runbook) with instructions for manual dispatch and automation wiring.
6. **Add regression coverage**
   - Extend existing workflow smoke test harness (if available) or add unit test script verifying summary renderer with mocked payloads.
7. **Update changelog / release notes**
   - Record workflow availability and intended usage for maintainers.

These steps establish a structured path for implementing the assignment verification workflow while respecting repository governance and automation constraints.
