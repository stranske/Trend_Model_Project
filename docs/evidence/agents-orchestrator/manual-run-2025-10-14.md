# Agents Orchestrator Manual Dispatch — 2025-10-14

## Target
- Workflow: `.github/workflows/agents-70-orchestrator.yml`
- Trigger: `workflow_dispatch`
- Payload: `{ "enable_verify_issue": "true", "verify_issue_number": "1234" }`

## Status
- :hourglass_flowing_sand: Pending external execution.
- The container environment used for development does not have credentials to trigger GitHub Actions runs. A repository maintainer must perform the manual dispatch.

## Execution Checklist for Maintainers
1. Navigate to **Actions → Agents 70 Orchestrator → Run workflow**.
2. Paste the payload above into the `params_json` input (the other inputs remain at their defaults).
3. Run the workflow on the default branch.
4. Confirm all jobs finish successfully:
   - `Resolve Parameters`
   - `Dispatch Agents Toolkit`
   - `Verify Assignment` (should execute because `enable_verify_issue` is `true` and the issue number is provided).
5. Capture the workflow run URL and paste it into [Issue #2560](https://github.com/stranske/Trend_Model_Project/issues/2560).
6. Record the run URL and a brief outcome summary below once available.

## Run Log
| Run Date | Triggered By | URL | Outcome | Notes |
|----------|--------------|-----|---------|-------|
| _pending_ | _pending_ | _pending_ | _pending_ | Awaiting maintainer dispatch. |
