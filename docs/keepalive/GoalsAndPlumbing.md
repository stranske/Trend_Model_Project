# Keepalive — Goals & Plumbing (Canonical Reference)

> ⚠️ **Mandatory reading.** Automation agents and human operators must review this guide before touching any keepalive workflow.
> It is the single source of truth for the checklist nudge loop and must stay in sync with every code change or workflow update.

---

## Purpose & Scope

| | |
|---|---|
| **Purpose** | Maintain a safe, iterative loop where an agent continues shipping small, verifiable updates on a PR until every acceptance criterion is met while guaranteeing predictable behavior and safety rails. |
| **Scope** | Labels, activation, dispatch, throttling, branch synchronization, comment formatting, and shutdown rules for keepalive automation. |
| **Non-goals** | Guidance for workflows unrelated to keepalive. |

---

## 1. Activation Guardrails (Round 0 → 1)
Keepalive must not post or dispatch for the first time unless **all** conditions below are true:

1. **Opt-in label** – The pull request carries the `agents:keepalive` label.
2. **Human kickoff** – A human `issue_comment.created` @mentions an agent whose login is derived dynamically from the PR's `agent:*` labels. Never hard-code handles.
3. **Gate success** – The Gate workflow for the current head SHA completed with `conclusion = success` (or another explicitly allow-listed positive result).

---

## 2. Repeat Contract (Round N → N + 1)
Before keepalive posts the next instruction or dispatches another run, confirm that:

- The three activation guardrails above are still satisfied.
- The concurrent run cap has not been exceeded (see Section 3).
- The branch-sync gate confirms the prior round's work landed on the PR branch (see Section 8).

If any item fails, **do not post a new instruction**. A run summary may be written for operators, but the PR thread must stay quiet.

---

## 3. Run Cap & Throttling

- **Default limit:** `K = 2` concurrent orchestrator/worker runs per PR.
- **Label override:** Respect `agents:max-parallel:<K>` when present (integer 1–5).
- **Enforcement rule:** Dispatch only when the number of in-progress runs is `< K`. Otherwise exit quietly or log the skip reason in the run summary.

---

## 4. Pause & Stop Labels

- Removing `agents:keepalive` halts future rounds until the label is re-applied **and** the activation guardrails pass again.
- `agents:pause` (when used) is a hard stop—block every form of keepalive activity, including fallbacks.

---

## 5. No-Noise Discipline
Missing prerequisites, red Gate results, or a saturated run cap must never produce a new PR comment. Keepalive may emit a step summary for operator awareness, but the PR stays untouched.

---

## 6. Instruction Comment Contract
When posting an instruction comment:

1. **Create a new comment** – Never edit existing status updates.
2. **Author identity** – Post as `stranske` via `ACTIONS_BOT_PAT`. Fallback: the automation bot via `SERVICE_BOT_PAT`.
3. **Required scaffolding** – The comment body **must start** with the hidden markers and trace block:
   ```markdown
   <!-- keepalive-round: {N} -->
   <!-- codex-keepalive-marker -->
   <!-- keepalive-trace: {TRACE} -->
   @<agent> Continue with the remaining tasks. Re-post Scope/Tasks/Acceptance and check off only when acceptance criteria are satisfied.

   <Scope/Tasks/Acceptance block>
   ```
4. **Reactions** – After posting, add 👀. PR-meta must acknowledge with 🚀 within the defined TTL.
5. **Checklist integrity** – Mark items complete only when the acceptance criteria are truly satisfied.

---

## 7. Detection & Dispatch Flow

- PR-meta listens for `issue_comment.created` events authored by `stranske` or the automation bot that contain the hidden markers.
- After deduplicating via the 🚀 reaction, PR-meta dispatches:
  - `workflow_dispatch` → `Agents-70 Orchestrator` with `options_json = { round, trace, pr }`.
  - `repository_dispatch` → `codex-pr-comment-command` with `{ issue, base, head, comment_id, comment_url, agent }`.
- Every event appends a summary row (`ok | reason | author | pr | round | trace`) to the operator log.

---

## 8. Branch-Sync Gate
Before proceeding to the next round:

1. **Detect movement** – Confirm the PR head SHA changed after the agent reported "done".
2. **Auto-remediation** – If unchanged, parse the agent's last reply for connector URLs ("Update Branch" / "Create PR"), invoke them automatically, and poll for a new commit (short TTL).
3. **Second-chance path** – When the first attempt fails, try the alternate connector route and poll again.
4. **Escalation** – If the head still does not advance, pause keepalive, apply `agents:sync-required`, and—only when `agents:debug` is present—post a single-line escalation that includes `{trace}`. Do **not** emit another instruction.

---

## 9. Orchestrator Invariants & Logging

- **No self-cancellation** – Configure concurrency as `{pr}-{trace}` with `cancel-in-progress: false`.
- **Explicit bailouts** – On any early exit (missing preconditions, cap reached, Gate not green, sync unresolved) record a one-line reason in the run summary. When `agents:debug` is set, optionally post `**Keepalive {round}** `{trace}` skipped: <reason-code>`.
- **Assignee hygiene** – Ignore bot or app accounts. If no human assignees remain, skip gracefully without failing the round.

---

## 10. Restart & Success Conditions

- Removing and then re-adding `agents:keepalive` restarts the automation as soon as all activation guardrails are satisfied again.
- Keepalive stands down once every acceptance criterion is checked complete. Optionally remove `agents:keepalive` and add `agents:done` to document closure.

---

## 11. Quick Reference Checklist

| Phase | Required Checks |
|-------|-----------------|
| **Activation** | `agents:keepalive` label · human @mention · Gate success |
| **Repeat** | Activation guardrails remain true · run cap respected · branch-sync satisfied |
| **Posting** | Fresh comment · hidden markers present · correct author token · 👀/🚀 reactions applied |
| **Dispatch** | Orchestrator + connector dispatch triggered with trace payloads |
| **Shutdown** | Acceptance criteria complete · remove keepalive label · optionally apply `agents:done` |

---

_Whenever the keepalive contract changes, update this document **and** its cross-references (e.g., `docs/agent-automation.md`, `docs/keepalive/Agents.md`, `docs/keepalive/SyncChecklist.md`) in the same commit so every entry point continues pointing at the canonical reference._
