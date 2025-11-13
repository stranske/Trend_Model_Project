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
- ✅ Secrets pass-through confirmed: `ACTIONS_BOT_PAT` supplied to dispatcher + worker in Agents 70 run #1860.
- ❌ Still no agent commits after keepalive. Commit list on #3191 shows only bootstrap + manual test commit; connector posts a summary but branch stays unchanged.
- 🔍 Root cause: `belt-worker` job skips whenever a PR already exists (`pr_exists == 'true'`). Keepalive rounds run against the active PR, so the worker never re-engages. Step summary logs “Result: skipped: PR exists.”

## Updated resolution strategy
1. Keep posting **new** keepalive instruction comments with `@codex` and `<!-- keepalive-round: N -->` markers (working as intended).
2. Allow `belt-worker` to run when keepalive is enabled even if the PR already exists, so follow-up rounds can push commits to the same branch.
3. Adjust orchestrator summaries so the “skipped: PR exists” guard only fires when keepalive is disabled.
4. (Optional) Add a precheck to halt keepalive once acceptance criteria are satisfied to avoid redundant work.
5. Re-run keepalive flows on an active PR to verify the worker now delivers incremental commits.

## Attempt — Repository dispatch keepalive (Nov 2025)
- ✅ PR-meta correctly detects the round markers and extracts the linked issue/branch.
- ❌ Repository dispatch to `agents-orchestrator-ping` fails with **403 Resource not accessible by integration** when executed with the default `GITHUB_TOKEN` (run ID 18996478914, job “Dispatch orchestrator keepalive”).
- 🔍 Failure occurs before the orchestrator starts, so no belt worker is invoked; keepalive remains idle.
- 📌 Conclusion: escalation option 1 is blocked without elevating credentials. The keepalive path must trigger the belt workflows directly.

## Attempt — Direct belt worker dispatch (Nov 2025)
- ✅ `agents-pr-meta` recognises keepalive round comments and assembles the worker payload (`issue`, `branch`, `base`).
- ❌ Workflow runs triggered by keepalive comments end with **startup_failure** because the `keepalive_worker` job requires `secrets.ACTIONS_BOT_PAT`, and GitHub withholds repository secrets from `issue_comment` dispatches authored by automation accounts. Recent examples: runs 18997968818 and 18997967860 (both cancelled before any jobs executed).
- 🔍 Since the reusable worker never starts, no commits or task execution occur—confirming that option 2 remains blocked without a PAT that can be shared with the comment-triggered workflow.
- 📌 Next step: either move the keepalive path back through Agents 70 (with PAT credentials) or provision an alternative credential scope that the PR-meta workflow can access when reacting to automation-authored comments.

## Attempt — Orchestrator relay with PAT (Nov 2025)
- ✅ Updated `agents-pr-meta` to dispatch **Agents 70 Orchestrator** directly whenever a keepalive round comment is detected. The job now uses `secrets.ACTIONS_BOT_PAT` to call `actions.createWorkflowDispatch`, forwarding `dispatcher_force_issue`, branch/base metadata, and an explicit `keepalive_enabled` flag.
- ⏳ Pending verification: need to observe a follow-up run to confirm the orchestrator honours the forced issue, invokes the belt worker, and resumes task execution on the existing PR branch.
- 📌 If GitHub still blocks the dispatch (e.g. PAT missing or insufficient scope), capture the run ID and revisit credential strategy.

## Attempt — Keepalive sentinel handshake (Nov 2025)
- ✅ `agents-pr-meta` now compiles a "Dispatch keepalive orchestrator" job and evaluates the same branch/issue metadata used in manual runs.
- ❌ Runs triggered by automation-authored keepalive comments exit with **Status: Skipped**; all jobs short-circuit because the dispatch guard resolves to `false`.
- 🔍 Detector script requires two hidden markers—`<!-- codex-keepalive-marker -->` and `<!-- keepalive-round: N -->`—plus an allow-listed author. Current keepalive comments from `stranske-automation-bot` only contain plain text (`"Keepalive Round N"` + `@codex`), so the sentinel check never passes.
- 📌 Net effect: orchestrator dispatch is skipped, no repository_dispatch/workflow_dispatch is issued, and the keepalive loop stalls despite the sweep posting comments successfully.

## Implementation notes (worker guard relaxation)
- Modify `.github/workflows/agents-70-orchestrator.yml` so the belt worker's `if` clause permits execution when `enable_keepalive` is `true`, even if a PR already exists.
- Retain the guard summary for the non-keepalive path, but switch the message to “keepalive override active” when the worker is allowed to continue.
- Bubble the same logic into the dispatch summary so round-two runs show the worker result instead of a forced skip.
- Keep the PAT pass-through unchanged (`ACTIONS_BOT_PAT` for dispatcher/worker, `service_bot_pat` for keepalive) to avoid regressing authentication.

## Patch — Hidden markers and concurrency (Nov 2025)
- Keepalive comments now always start with `<!-- keepalive-round: N -->` followed by `<!-- codex-keepalive-marker -->`, matching the sentinel contract used by PR-meta.
- Each round explicitly reminds the agent to keep the checklist current and post a summary when the round concludes; the legacy command listener is disabled so keepalive is the single automation trigger.
- Orchestrator concurrency keys on the PR (falling back to the ref) and no longer cancels in-flight runs, so consecutive keepalive rounds cannot interrupt one another.
- The keepalive sweep declares write permissions up front, avoiding token-scope regressions when posting round comments.

## Regression — Belt dispatcher outputs missing (Nov 2025)

## Regression — Ledger validation blocks keepalive (Nov 2025)
- 🧪 Run [19021825748](https://github.com/stranske/Trend_Model_Project/actions/runs/19021825748) was dispatched from keepalive round 8 (`stranske-automation-bot` comment) and resolved `enable_keepalive: true`, so the worker was allowed to continue on the existing PR branch.
- ❌ `Codex Belt Worker / Prepare Codex automation PR` failed during `Validate ledger schema (final)` with `tasks[0].commit 91e08ebd6d60e67d0a5d7fc9af4c13cb1691cb82 must include non-ledger changes`.
- 🔍 Commit `91e08ebd6d60e67d0a5d7fc9af4c13cb1691cb82` (authored by `stranske-automation-bot`) only touched `.agents/issue-3209-ledger.yml`, so the validator rejects it; the worker aborts before pushing any follow-up changes.
- 📉 Net effect: keepalive comments continue posting, but the branch never receives updates and the summary still reports `skipped: PR exists`, masking the ledger failure.
- 🛠️ Next steps: adjust the ledger workflow so keepalive runs either reference a commit with real code changes or relax the validator for pure ledger bootstrap commits; also update the orchestrator summary to surface the actual worker failure when keepalive overrides are active.

## Noise — Connector autop replies (Nov 2025)
- 🔁 Every keepalive round triggered an immediate `chatgpt-codex-connector` reply of “To use Codex here, create a Codex account…,” resulting in eight duplicate noise comments on PR #3210.
- ⚖️ These replies violate the “prune unhelpful bot chatter” goal and bury the human keepalive instructions (`@codex` checklist plus capitalised emphasis) under boilerplate.
- 📌 Suppress the connector’s marketing stub for keepalive-authored comments while retaining the genuine status summaries triggered by real commits.
- 🧹 Ensure only the human keepalive prompt, the automation round comment, and the agent’s work summaries remain visible so the human instruction continues to anchor the workflow.

## Mitigation — Ledger + Connector adjustments (Dec 2025)
- ✅ Ledger validator now allows `chore(ledger): …` commits that only touch the active ledger file (plus ledger sidecars) so bootstrap tasks stop failing the non-ledger guard.
- ✅ Keepalive-triggered belt worker runs skip reposting the `@codex start` activation comment, preventing the connector from spamming marketing replies every round.
- ✅ Orchestrator summary now surfaces downstream worker failures directly, keeping ledger-validation errors visible instead of falling back to the “skipped: PR exists” guard message.
- ✅ Gate-completion dispatch now marks keepalive sweeps as gate-triggered unconditionally, so every Gate run resets the idle timer and bypasses the cooldown checks.
- 🔄 Follow-up: trigger a fresh keepalive round to confirm the worker progresses past ledger validation and that the connector noise no longer appears.
## Escalation options (recorded)
1. **Repository dispatch → Orchestrator** – _Blocked_. PR-meta lacks token scope to call `repos.createDispatchEvent`, resulting in 403s and no orchestrator run. Escalation path disabled unless a PAT is wired in.
2. **Direct belt workflows** – ✅ Implemented November 2025. PR-meta now invokes `Agents 72 Codex Belt Worker` directly with the detected issue/branch so the worker re-engages without involving the chat connector.
3. **Round parser hardening** – Treat `<!-- keepalive-round: N -->` as the stable sentinel, verify the author is one of our automation accounts, and optionally ensure the Gate check suite reports “concluded” before dispatching. This keeps false positives out of the escalation path.
4. **Option A — Inject hidden sentinels** – Update the keepalive comment template in Agents 70 so each posted round includes both `<!-- codex-keepalive-marker -->` and `<!-- keepalive-round: N -->`, satisfying the detector without touching PR-meta.
5. **Option B — Relax detector heuristics** – Modify `.github/workflows/agents-pr-meta.yml` so the keepalive path accepts either the hidden markers or the current plain-text pattern (`"Keepalive Round"` plus `@codex`) while retaining the author allow list.

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
4. Validate that a fresh `Keepalive Round N` comment appears on the target issue/PR with the correct marker `<!-- keepalive-round: N -->`.
5. Check the worker logs for commit pushes or task execution to ensure the branch received updates after the keepalive round.
6. If the run fails, capture the Actions URL, PR number, and worker logs, and add a new bullet in “What failed before” with observed symptoms.
