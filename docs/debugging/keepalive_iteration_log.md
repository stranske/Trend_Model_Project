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

## Implementation notes (worker guard relaxation)
- Modify `.github/workflows/agents-70-orchestrator.yml` so the belt worker's `if` clause permits execution when `enable_keepalive` is `true`, even if a PR already exists.
- Retain the guard summary for the non-keepalive path, but switch the message to “keepalive override active” when the worker is allowed to continue.
- Bubble the same logic into the dispatch summary so round-two runs show the worker result instead of a forced skip.
- Keep the PAT pass-through unchanged (`actions_bot_pat` for dispatcher/worker, `service_bot_pat` for keepalive) to avoid regressing authentication.
