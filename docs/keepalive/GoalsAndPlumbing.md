# Keepalive — Goals & Plumbing (Canonical Reference)

This runbook is the single source of truth for the Codex keepalive workflow. Automation agents **must review this document before touching any keepalive-related logic or workflows.** It captures the success contract, guard rails, and dispatcher expectations that keep the iterative checklist loop predictable and safe.

## Scope

- Guidelines and goals for the keepalive workflow (posting instructions, dispatch, synchronization, and guard rails).

Non-goals:

- Instructions for unrelated automation surfaces or non-keepalive workflows.

## Purpose

Drive an iterative, hands-off loop where an agent continues working on a pull request in small, verifiable increments until all acceptance criteria are met—while guaranteeing safety rails and predictable behaviour. Automation must surface the remaining work, nudge only when conditions allow, and stop cleanly once the checklist is complete.

## Activation Contract (First Instruction on a PR)

Keepalive must not post or dispatch unless all of the following are true:

1. **Label opt-in:** The pull request has the `agents:keepalive` label.
2. **Human kickoff:** A human posted an `issue_comment` that @mentions an agent listed in the PR's `agent:*` labels (dynamic agent name detection—no hard-coded logins).
3. **Gate ready:** The Gate workflow for the current head SHA completed with a success conclusion (or another explicitly allow-listed conclusion).

## Repeat Contract (Round _N_ → _N_ + 1)

Before keepalive posts the next instruction or dispatches another worker run:

- Re-validate the three activation preconditions.
- Confirm the concurrent run cap is not exceeded.
- Ensure the branch-sync gate confirms that prior work landed on the PR branch.

If any precondition fails, **keepalive must stay silent** (no comments). It may record the skip reason in step summaries for operator visibility.

## Run Cap Rules

- Default maximum concurrent runs: **2** per PR.
- Override via label: `agents:max-parallel:<K>` (integer 1–5). Do not exceed the configured cap when dispatching Orchestrator/Worker jobs.

## Pause & Stop Controls

- Removing `agents:keepalive` halts future keepalive rounds until the label is re-applied and prerequisites pass again.
- Optional hard pause label `agents:pause` blocks **all** keepalive activity, including fallbacks.

## Instruction Comment Contract

When posting an instruction:

- Create a brand-new comment (never edit an existing one).
- Author: prefer `stranske` via `ACTIONS_BOT_PAT`; fallback to `stranske-automation-bot` via `SERVICE_BOT_PAT`.
- The comment body **must start** with the hidden markers:
  ```
  <!-- keepalive-round: {N} -->
  <!-- codex-keepalive-marker -->
  <!-- keepalive-trace: {TRACE} -->
  ```
- Follow the markers with the visible instruction addressed to the agent (e.g., `@codex …`).
- Include the current Scope/Tasks/Acceptance block and mark items complete **only when acceptance criteria have been satisfied**.
- After posting, add an 👀 reaction. PR-meta must acknowledge with 🚀 within the expected TTL.

## Detection & Dispatch Responsibilities

PR-meta listens for qualifying keepalive comments (author + markers) and, once acknowledged, dispatches both:

- `workflow_dispatch` → Agents-70 Orchestrator (`options_json` includes `{ round, trace, pr }`).
- `repository_dispatch` → `codex-pr-comment-command` connector (`{ issue, base, head, comment_id, comment_url, agent }`).

PR-meta writes a summary table for every event (`ok | reason | author | pr | round | trace`).

## Branch-Sync Gate

Before issuing the next instruction for a round, the system must confirm that the PR head SHA changed (new work landed). If not:

1. Scan the agent's last reply for “Update Branch” / “Create PR” URLs (connector hand-offs). Call the URL automatically.
2. Poll for a new commit (short TTL). If still unchanged, retry with the alternate path (e.g., Create PR after Update Branch).
3. When both recovery attempts fail, pause keepalive, add the `agents:sync-required` label, and (only when `agents:debug` is present) post a one-line escalation comment containing the `{trace}` token. Do not post a fresh instruction.

## Orchestrator Invariants

- **No self-cancellation:** Configure concurrency as `{pr}-{trace}` with `cancel-in-progress: false`.
- **Explicit bails:** On early exits (missing preconditions, run cap reached, gate not green, sync unresolved) write a one-line summary reason. Optionally add a terse PR comment when `agents:debug` is present using the format `**Keepalive {round}** \{{trace}\} skipped: <reason-code>`.
- **Assignee hygiene:** Ignore bot/app accounts. If no human assignees remain, skip gracefully instead of failing the round.
- Removing and re-adding `agents:keepalive` restarts keepalive once all contracts above are satisfied.
- Success condition: stop when all acceptance criteria are checked complete. Optionally remove `agents:keepalive` and apply `agents:done`.

## Operator Checklist

- ✅ Confirm labels (`agents:keepalive`, optional `agents:max-parallel`, no `agents:pause`).
- ✅ Ensure human @mention kickoff occurred after the latest commit.
- ✅ Verify Gate succeeded for the current head SHA.
- ✅ Check branch-sync summary for recent head movement before issuing next round.
- ✅ Keep a trace log of reactions (👀/🚀) and dispatch payloads for audit.

Keeping this document current is mandatory whenever keepalive workflows, orchestrator logic, or connector contracts change. Update cross-references (e.g., `docs/agent-automation.md`, `docs/keepalive/SyncChecklist.md`) simultaneously so every agent entry point points back to this canonical reference.
