# Keepalive — Goals & Plumbing (Canonical Reference)

> **Agents & Operators:** Consult this document whenever you modify or validate any keepalive behavior. It is the canonical contract for how the workflow must operate.

## Overview

- **Purpose:** Maintain a safe, iterative loop where agents continue small, verifiable updates on a PR until every acceptance criterion is met—while ensuring predictable behavior and safety rails.
- **Scope:** Keepalive activation, dispatch, throttling, comment format, synchronization, and shutdown rules for GitHub PR workflows.
- **Non-Goals:** Instructions or policies unrelated to keepalive.

---

## 1. Activation Guardrails (Round 0 → 1)

Keepalive must not post or dispatch for the first time unless **all** conditions are true:

1. **PR Label Present** – The PR carries the label `agents:keepalive`.
2. **Human Initiation Logged** – At least one human `issue_comment.created` on the PR @mentions an agent whose name is drawn dynamically from the PR's `agent:*` labels. No hard-coded agent handles.
3. **Gate Workflow Green** – The Gate workflow for the current head SHA completed with `conclusion = success` (or another allow-listed positive result).

---

## 2. Repeat Contract (Round N → N+1)

Before keepalive posts the next instruction comment or dispatches another run, verify:

- The three activation guardrails remain satisfied.
- The **run cap** has not been exceeded (see Section 3).
- The **branch-sync gate** reports that the previous agent work landed on the PR branch (see Section 6).

---

## 3. Run Cap Enforcement

- **Configurable via Label:** If the PR has `agents:max-parallel:K`, use that integer (1–5).
- **Default Limit:** `K = 2` when no label is present.
- **Enforcement Rule:** Only dispatch when `in-progress orchestrator/worker runs` < `K`. Otherwise, exit quietly (optionally writing only a run summary).

---

## 4. Pause & Stop Controls

- Removing `agents:keepalive` halts new runs and prevents further keepalive comments.
- Optionally honor a stronger `agents:pause` label that blocks **all** keepalive activity, including fallbacks.

---

## 5. No-Noise Policy

If any precondition fails (labels missing, human @mention absent, Gate not green, run cap reached), keepalive must not post new PR comments. It may emit an operator run summary, but the PR thread remains untouched.

---

## 6. Branch-Sync Gate

Before advancing to the next round:

1. Detect whether the PR head SHA moved after the agent reported "done."
2. If unchanged, parse the agent's latest reply for an "Update Branch" or "Create PR" URL (connector patterns), trigger it automatically, and poll for a new commit (short TTL).
3. If still unchanged, attempt the alternate path (e.g., invoke "Create PR" when "Update Branch" is blocked) and poll again.
4. If there is still no new head SHA, pause keepalive and apply `agents:sync-required`. When a debug label (e.g., `agents:debug`) exists, a short single-line PR comment containing the `{trace}` may be posted.

---

## 7. Instruction Comment Contract

When posting is allowed:

1. **Create a Fresh Comment** – Never edit a status comment.
2. **Author Identity** – Post as `stranske` using `ACTIONS_BOT_PAT`. Fallback: `SERVICE_BOT_PAT` (automation bot account).
3. **Required Markers & Structure:**
   ```markdown
   <!-- keepalive-round: {N} -->
   <!-- codex-keepalive-marker -->
   <!-- keepalive-trace: {TRACE} -->
   @<agent> Continue with the remaining tasks. Re-post Scope/Tasks/Acceptance and check off only when acceptance criteria are satisfied.

   <Scope/Tasks/Acceptance block>
   ```
4. **Reactions:** After posting, the system adds 👀; PR-meta must add 🚀 (ack) quickly.

---

## 8. Detection & Dispatch Flow

- **Event Listener:** PR-meta listens to `issue_comment.created` authored by `stranske` or the automation bot.
- **Validation:** Hidden markers must be present; PR-meta deduplicates via the 🚀 reaction.
- **Dispatch Actions:**
  - `workflow_dispatch → Agents-70 Orchestrator` with `options_json = {round, trace, pr}`.
  - `repository_dispatch (codex-pr-comment-command)` for the connector with `{issue, base, head, comment_id, comment_url, agent}`.
- **Run Logging:** PR-meta appends a compact summary row (`ok | reason | author | pr | round | trace`) per event.

---

## 9. Orchestrator Invariants

- **No Self-Cancellation:** `cancel-in-progress = false`; concurrency scoped by `{pr}-{trace}`.
- **Explicit Bails:** For any early exit (missing precondition, cap hit, Gate not green, branch-sync unresolved), log a one-line reason in the run summary and—when a debug label exists—optionally post a terse PR comment:
  
  ```text
  **Keepalive {round}** `{trace}` skipped: <reason-code>
  ```
- **Assignees:** Ignore bot/app accounts. If no human assignees remain, skip gracefully without failing the round.

---

## 10. Restart Behavior

Removing and later re-adding `agents:keepalive` restarts the automation once all activation guardrails are satisfied again.

---

## 11. Success Condition

Keepalive stands down when every acceptance criterion is checked complete. Optionally remove `agents:keepalive` and apply `agents:done`.

---

## Appendix: Quick Reference Checklist

| Phase | Key Checks |
|-------|------------|
| Activation | `agents:keepalive` label · human @mention · Gate success |
| Repeat | Activation checks still true · run cap respected · branch-sync satisfied |
| Posting | Fresh comment · required hidden markers · correct author token |
| Dispatch | Hidden markers verified · 🚀 ack · orchestrator and connector dispatch triggered |
| Exit | All criteria complete → remove keepalive · optionally add `agents:done` |

---

_Last synchronized via workflow run referenced in PR metadata._
