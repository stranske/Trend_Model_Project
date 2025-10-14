# Manual Orchestrator Run — Issue #2566 Verification Sweep

> **Status:** Pending execution — requires a GitHub Actions run using repository credentials.

## Dispatch Parameters

Use the following payload when triggering **Agents 70 Orchestrator** manually:

```json
{
  "enable_verify_issue": "true",
  "verify_issue_number": "<replace-with-controlled-issue-number>",
  "verify_issue_valid_assignees": "copilot,chatgpt-codex-connector,stranske-automation-bot"
}
```

### GitHub CLI

```bash
BRANCH="phase-2-dev"
ISSUE="<replace-with-controlled-issue-number>"

cat <<'JSON' > orchestrator-verify.json
{
  "enable_verify_issue": true,
  "verify_issue_number": "${ISSUE}",
  "verify_issue_valid_assignees": "copilot,chatgpt-codex-connector,stranske-automation-bot"
}
JSON

gh workflow run agents-70-orchestrator.yml \
  --ref "${BRANCH}" \
  --raw-field params_json="$(cat orchestrator-verify.json)"
```

### REST API

```bash
curl -X POST \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/stranske/Trend_Model_Project/actions/workflows/agents-70-orchestrator.yml/dispatches \
  -d @<(jq -nc --arg ref "phase-2-dev" --arg params "$(cat orchestrator-verify.json)" '{ref: $ref, inputs: {params_json: $params}}')
```

## Run Log

Once dispatched, record the resulting Actions run URL below so future audits can confirm the `verify-assignment-summary` step logged the matched assignee. Replace `<RUN_URL>` with the actual link and add the assignee details captured in the summary.

- **Run:** `<RUN_URL>`
- **Matched assignee:** `<matched-login>`
- **Status:** `pass`

## Notes

- The orchestrator now forwards `verify_issue_valid_assignees` to both the reusable toolkit and verification workflow so `stranske-automation-bot` satisfies the check when label rules require automation involvement.
- The reusable verification workflow publishes the matched assignee via outputs and step summary; ensure the run summary includes the appended section before closing out Issue #2566.
