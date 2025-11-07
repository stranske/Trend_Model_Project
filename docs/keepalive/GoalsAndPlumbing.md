# Keepalive — Goals & Plumbing (Canonical Reference)

> **Audience:** Human maintainers and automation agents responsible for the Codex keepalive workflow. Review this document **before** touching any keepalive logic or dispatch plumbing.

## Quick Navigation
- [Purpose & Scope](#purpose--scope)
- [Lifecycle Overview](#lifecycle-overview)
- [1. Activation Guardrails (Round 0 → 1)](#1-activation-guardrails-round-0--1)
- [2. Repeat Contract (Round N → N+1)](#2-repeat-contract-round-n--n1)
- [3. Run Cap Enforcement](#3-run-cap-enforcement)
- [4. Pause & Stop Controls](#4-pause--stop-controls)
- [5. No-Noise Policy](#5-no-noise-policy)
- [6. Instruction Comment Contract](#6-instruction-comment-contract)
- [7. Detection & Dispatch Flow](#7-detection--dispatch-flow)
- [8. Branch-Sync Gate](#8-branch-sync-gate)
- [9. Orchestrator Invariants](#9-orchestrator-invariants)
- [10. Restart & Success Conditions](#10-restart--success-conditions)
- [Appendix: Operator Checklist](#appendix-operator-checklist)

---

## Purpose & Scope

- **Purpose:** Maintain a safe, iterative loop where keepalive nudges an agent through small, verifiable increments on a PR until every acceptance criterion is complete—while guaranteeing predictable behaviour and safety rails.
- **Scope:** Activation requirements, dispatch plumbing, throttling, branch-sync guarantees, and shutdown rules for the GitHub PR keepalive workflow.
- **Non-goals:** Guidance for automation unrelated to keepalive.

---

## Lifecycle Overview

1. **Human kickoff:** A maintainer @mentions the active agent, which primes the orchestrator to watch the PR.
2. **Guarded check:** Orchestrator guardrails confirm the label, Gate success, human activation, and run-cap capacity before any instruction posts.
3. **Timed repeats:** Subsequent scheduled sweeps rerun the guardrails (including Gate completion) and only dispatch when the contract still holds.
4. **Definition of done:** As soon as the acceptance criteria are all checked complete, keepalive posts no further rounds and removes the `agents:keepalive` label.
5. **Suspend on label change:** If the label disappears or the guardrails fail mid-run, the workflow records the skip reason and stays silent until a human re-applies `agents:keepalive` and restores the prerequisites.

---

## 1. Activation Guardrails (Round 0 → 1)
Keepalive **must not** post or dispatch its first instruction unless *all* conditions hold:

1. **PR opt-in:** The PR carries the `agents:keepalive` label.
2. **Human kickoff:** A human `issue_comment.created` @mentions an agent whose handle is discovered dynamically from the PR's `agent:*` labels. No hard-coded agent names.
3. **Gate green:** The Gate workflow for the current head SHA completed successfully (or matches an approved allow-list of positive conclusions).

---

## 2. Repeat Contract (Round N → N+1)
Before the next instruction comment or worker run:

- Re-validate the three activation guardrails.
- Confirm the concurrent run cap is still available (see Section 3).
- Ensure the branch-sync gate reports that the previous round's work actually landed on the PR branch (see Section 8).

If any requirement fails, keepalive stays silent—no PR comments. Operators may record the skip reason in run summaries only.

---

## 3. Run Cap Enforcement

- **Default limit:** Maximum of **2** concurrent orchestrator/worker runs per PR.
- **Label override:** Respect `agents:max-parallel:<K>` when present (integer 1–5).
- **Enforcement:** Dispatch only when the count of in-progress orchestrator/worker runs is `< K`. If at cap, exit quietly after updating the run summary.

---

## 4. Pause & Stop Controls

- Removing `agents:keepalive` halts new instructions and dispatches until the label is re-applied and all guardrails pass again. The orchestrator records the skip reason but emits no PR comments while the label is missing.
- Respect the stronger `agents:pause` label, which blocks *all* keepalive activity, including fallback automation.

---

## 5. No-Noise Policy

When preconditions are missing (labels absent, no human kickoff, Gate not green, run cap reached), keepalive must not add new PR comments. At most, log a concise operator summary explaining the skipped action.

---

## 6. Instruction Comment Contract

When posting is allowed:

1. **Brand-new comment:** Never edit an existing status comment.
2. **Author identity:** Post as `stranske` using `ACTIONS_BOT_PAT`; fallback to the automation bot via `SERVICE_BOT_PAT` when necessary.
3. **Required header markers:**
   ```markdown
   <!-- keepalive-round: {N} -->
   <!-- codex-keepalive-marker -->
   <!-- keepalive-trace: {TRACE} -->
   @<agent> Use the scope, acceptance criteria, and task list so the keepalive workflow continues nudging until everything is complete. Work through the tasks, checking them off only after each acceptance criterion is satisfied. During each comment implementation, check off tasks and acceptance criteria that have been satisfied and re-post the current version of the initial scope, task list, and acceptance criteria whenever any new items are completed.

   <Scope/Tasks/Acceptance block>
   ```
4. **Reaction contract:** After posting, add 👀. PR-meta must acknowledge with 🚀 within the expected TTL.

---

## 7. Detection & Dispatch Flow

- **Event listener:** PR-meta consumes `issue_comment.created` events from `stranske` or the automation bot.
- **Validation:** Hidden markers are mandatory; PR-meta deduplicates events via the 🚀 acknowledgement.
- **Dispatch actions:**
  - Trigger `workflow_dispatch → Agents-70 Orchestrator` with `options_json = { round, trace, pr }`.
  - Trigger `repository_dispatch (codex-pr-comment-command)` with `{ issue, base, head, comment_id, comment_url, agent }`.
- **Run logging:** PR-meta records each event as `ok | reason | author | pr | round | trace` in its summary table.

---

## 8. Branch-Sync Gate

Before the next round begins:

1. Verify that the PR head SHA changed after the agent reported "done".
2. If unchanged, scan the agent's latest reply for "Update Branch" or "Create PR" URLs, invoke the detected action automatically, and poll for a new commit (short TTL).
3. Still unchanged? Attempt the alternate path (e.g., fall back to "Create PR" when "Update Branch" is unavailable) and poll again.
4. When both attempts fail, pause keepalive, apply `agents:sync-required`, and—with `agents:debug` present—post a single-line escalation containing the `{trace}` token. Do **not** post a new instruction.

---

## 9. Orchestrator Invariants

- **No self-cancellation:** Configure concurrency as `{pr}-{trace}` with `cancel-in-progress: false`.
- **Explicit bails:** For early exits (missing preconditions, run cap reached, Gate not green, sync unresolved), write a one-line reason to the run summary. When `agents:debug` is present, optionally add:
  ```text
  **Keepalive {round}** `{trace}` skipped: <reason-code>
  ```
- **Assignee hygiene:** Ignore bot/app accounts; if no human assignees remain, skip gracefully instead of failing the round.

---

## 10. Restart & Success Conditions

- Removing and re-applying `agents:keepalive` restarts the workflow once the activation guardrails pass again.
- Keepalive stands down when **all acceptance criteria are checked complete**. At that point the orchestrator removes `agents:keepalive`, may add `agents:done`, and stops issuing further rounds.

---

## Appendix: Operator Checklist

| Phase | Key Checks |
|-------|------------|
| Activation | `agents:keepalive` label · human @mention from valid agent label · Gate success |
| Repeat | Activation guardrails still true · run cap respected · branch-sync satisfied |
| Posting | Fresh comment · required hidden markers · correct author identity |
| Dispatch | Hidden markers validated · 👀/🚀 reactions complete · orchestrator and connector dispatch triggered |
| Exit | All acceptance criteria satisfied · keepalive removed or marked `agents:done` |

Keep this document in sync with `docs/agent-automation.md` and `docs/keepalive/SyncChecklist.md` whenever the workflow evolves.
