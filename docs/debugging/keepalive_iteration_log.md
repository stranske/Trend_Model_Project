# Keepalive Iteration Log

## What failed before
- **Token theory rejected** – `Codex Keepalive Sweep` authenticates with `SERVICE_BOT_PAT`, verifies the identity as `stranske-automation-bot`, and successfully posts comments (workflow logs confirm).
- **Gate gating theory rejected** – Gate completes on the relevant head SHA prior to keepalive attempts; status checks are green.
- **Agent intake theory rejected** – Human `@codex` comments trigger the Agents 63/71/72 pipeline; the same chain remains idle only after keepalive edits.

## Evidence-backed root cause (pre-Oct 2025)
- Keepalive edited the existing status comment (`commented • edited`) instead of creating a new instruction comment.
- Agents 63 listens to `issue_comment.created`; no new comment event ⇒ no second-round dispatch.
- Actions history showed no runs for “issue comment created by stranske-automation-bot,” matching edit-only behaviour.

## Attempt — Round comment publishing (Oct 2025)
- ✅ Keepalive now creates discrete `Keepalive Round N` comments that @mention the agent; connector responds to each round (#3191 timeline).
- ✅ Secrets pass-through confirmed: `actions_bot_pat` supplied to dispatcher + worker in Agents 70 run #1860.
- ❌ Still no agent commits after keepalive. Commit list on #3191 shows only bootstrap + manual test commit; connector posts a summary but branch stays unchanged.
- 🔍 Root cause: `belt-worker` job skips whenever a PR already exists (`pr_exists == 'true'`). Keepalive rounds run against the active PR, so the worker never re-engages. Step summary logs “Result: skipped: PR exists.”

## Updated resolution strategy
1. Keep posting **new** keepalive instruction comments with `@codex` and `<!-- keepalive-round:N -->` markers (working as intended).
2. Allow `belt-worker` to run when keepalive is enabled even if the PR already exists, so follow-up rounds can push commits to the same branch.
3. Adjust orchestrator summaries so the “skipped: PR exists” guard only fires when keepalive is disabled.
4. (Optional) Add a precheck to halt keepalive once acceptance criteria are satisfied to avoid redundant work.
5. Re-run keepalive flows on an active PR to verify the worker now delivers incremental commits.

## Attempt — Repository dispatch keepalive (Nov 2025)
- ✅ PR-meta correctly detects the round markers and extracts the linked issue/branch.
- ❌ Repository dispatch to `agents-orchestrator-ping` fails with **403 Resource not accessible by integration** when executed with the default `GITHUB_TOKEN` (run ID 18996478914, job “Dispatch orchestrator keepalive”).
- 🔍 Failure occurs before the orchestrator starts, so no belt worker is invoked; keepalive remains idle.
- 📌 Conclusion: escalation option 1 is blocked without elevating credentials. The keepalive path must trigger the belt workflows directly.

## Implementation notes (worker guard relaxation)
- Modify `.github/workflows/agents-70-orchestrator.yml` so the belt worker's `if` clause permits execution when `enable_keepalive` is `true`, even if a PR already exists.
- Retain the guard summary for the non-keepalive path, but switch the message to “keepalive override active” when the worker is allowed to continue.
- Bubble the same logic into the dispatch summary so round-two runs show the worker result instead of a forced skip.
- Keep the PAT pass-through unchanged (`actions_bot_pat` for dispatcher/worker, `service_bot_pat` for keepalive) to avoid regressing authentication.

## Escalation options (recorded)
1. **Repository dispatch → Orchestrator** – _Blocked_. PR-meta lacks token scope to call `repos.createDispatchEvent`, resulting in 403s and no orchestrator run. Escalation path disabled unless a PAT is wired in.
2. **Direct belt workflows** – ✅ Implemented November 2025. PR-meta now invokes `Agents 72 Codex Belt Worker` directly with the detected issue/branch so the worker re-engages without involving the chat connector.
3. **Round parser hardening** – Treat `<!-- keepalive-round:N -->` as the stable sentinel, verify the author is one of our automation accounts, and optionally ensure the Gate check suite reports “concluded” before dispatching. This keeps false positives out of the escalation path.

## Keepalive dispatch options
- `enable_keepalive` – master toggle; set to `'true'` to allow follow-up rounds to bypass the existing-PR guard.
- `keepalive_idle_minutes` – idle threshold before a new round posts the instruction comment (default 10).
- `keepalive_repeat_minutes` – cooldown between rounds to prevent comment spam (default 30).
- `keepalive_labels` – optional comma-separated label filter so the sweep only targets matching issues or PRs.
- `keepalive_command` – custom instruction text; defaults to the orchestrator's canned `@codex plan-and-execute` prompt.
- `keepalive_pause_label` – label that pauses keepalive on specific threads when present (`keepalive:paused`).

> Pass the values above via the orchestrator's `params_json` payload, e.g. `{ "enable_keepalive": true, "keepalive_idle_minutes": 15 }`. Nested overrides belong inside the embedded `options_json` field when dispatched from composite workflows.

## Verification checklist (post-update)
1. Manually dispatch **Agents 70 Orchestrator** with `enable_keepalive: true` (or include the flag in `params_json`).
2. Confirm the Actions summary shows “keepalive override active – worker may resume existing branch.”
3. Inspect the **Codex Belt Worker** job; it should run even when an existing PR is detected.
4. Validate that a fresh `Keepalive Round N` comment appears on the target issue/PR with the correct marker `<!-- keepalive-round:N -->`.
5. Check the worker logs for commit pushes or task execution to ensure the branch received updates after the keepalive round.
6. If the run fails, capture the Actions URL, PR number, and worker logs, and add a new bullet in “What failed before” with observed symptoms.
