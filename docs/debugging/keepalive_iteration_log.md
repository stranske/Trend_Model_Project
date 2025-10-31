# Keepalive Iteration Log

## What failed before
- **Token theory rejected** – `Codex Keepalive Sweep` authenticates with `SERVICE_BOT_PAT`, verifies the identity as `stranske-automation-bot`, and successfully posts comments (workflow logs confirm).
- **Gate gating theory rejected** – Gate completes on the relevant head SHA prior to keepalive attempts; status checks are green.
- **Agent intake theory rejected** – Human `@codex` comments trigger the Agents 63/71/72 pipeline; the same chain remains idle only after keepalive edits.

## Evidence-backed root cause
- Keepalive edits the existing status comment (`commented • edited`) instead of creating a new instruction comment.
- Agents 63 listens to `issue_comment.created`; no new comment event ⇒ no second-round dispatch.
- Actions history shows no runs for “issue comment created by stranske-automation-bot,” aligning with edit-only behaviour.

## Resolution strategy
1. Post a **new** keepalive instruction comment for every round, with `@codex` on the first line and a hidden marker `<!-- keepalive-round:N -->`.
2. Compute the next round number by scanning prior bot comments; keep status updates in the long-lived summary comment.
3. Ensure keepalive assigns agent connectors before posting so they remain eligible recipients.
4. Extend tests (JS harness + pytest) to assert `issue_comment.created` semantics and marker handling.
5. Validate via orchestrator dry run that keepalive now triggers the agent for subsequent rounds until acceptance criteria complete.
