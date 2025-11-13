<!-- bootstrap for codex on issue #3364 -->

## Scope
- Detect when the PR head SHA has not advanced after an agent reports completion during keepalive rounds.
- Post a visible `/update-branch trace:{TRACE}` command as **stranske** (ACTIONS_BOT_PAT preferred, SERVICE_BOT_PAT fallback) and react with 👀 whenever an unsynced branch is detected.
- Dispatch the `agents-keepalive-branch-sync.yml` fallback workflow with full metadata when the comment-first path does not land within the short TTL, and poll until the PR head advances or the long TTL expires.
- Halt further keepalive instructions, apply `agents:sync-required`, and log concise `$GITHUB_STEP_SUMMARY` status when the branch still has not moved after the fallback path.
- Keep all guardrails (label checks, gate status, run-cap) unchanged while avoiding additional PR noise beyond the command comment (and optional debug escalation).

## Task List
- [ ] Persist & compare PR head SHA values between keepalive rounds.
  - [ ] Record `{PR, round, trace, head_sha}` at instruction time (e.g., hidden HTML comment or JSON artifact).
  - [ ] Read the stored value post-work and compute `unsynced = (previous_head_sha == current_head_sha)`.
- [ ] Implement the comment-first update command when `unsynced` is true.
  - [ ] Select ACTIONS_BOT_PAT, falling back to SERVICE_BOT_PAT if required, to author the comment as stranske.
  - [ ] Post a new PR comment with body exactly `/update-branch trace:{TRACE}` and add an 👀 reaction.
  - [ ] Append a `$GITHUB_STEP_SUMMARY` entry logging the comment id, URL, author, trace, and round.
  - [ ] Poll the PR head every 5 seconds up to `TTL_short` (≈60–120s) and mark success (`mode=comment-update-branch`) if the head SHA changes.
- [ ] Build the fallback dispatch path when the PR head remains unchanged after `TTL_short`.
  - [ ] Ensure the orchestrator job has permissions `actions: write`, `contents: read`, and `pull-requests: write`.
  - [ ] Trigger `agents-keepalive-branch-sync.yml` via `createWorkflowDispatch`, passing `pr_number`, `trace`, `base_ref`, `head_ref`, `head_sha`, `agent`, `round`, `comment_id`, `comment_url`, and an idempotency key (trace is sufficient).
  - [ ] Authenticate the dispatch with ACTIONS_BOT_PAT.
  - [ ] Log the dispatch (status code, workflow file, trace, run URL when available) to `$GITHUB_STEP_SUMMARY`.
- [ ] Monitor for branch advancement or timeout after the fallback dispatch.
  - [ ] Continue polling the PR head up to `TTL_long` (≈2–5 minutes) and on success log `mode=action-sync-pr` plus the merged commit SHA.
  - [ ] If the head is still unchanged at `TTL_long`, apply the `agents:sync-required` label, suppress the next keepalive instruction, and record `reason=sync-timeout trace:{TRACE}` in the summary (optionally emit a one-line debug comment only when `agents:debug` is present).
- [ ] Guarantee idempotency and no-noise safeguards.
  - [ ] Treat `{PR, round, trace}` as the idempotency key so duplicate runs do not re-post comments or re-dispatch the workflow.
  - [ ] Ensure all negative guardrail failures (labels missing, gate incomplete, run cap reached) produce summary-only logs with no PR comments.

## Acceptance Criteria
- [ ] On a test PR carrying `agents:keepalive` + `agent:codex`, when an agent finishes but the head SHA is unchanged, a new comment authored by stranske appears with body `/update-branch trace:{TRACE}` and an 👀 reaction.
- [ ] If the head moves within `TTL_short`, the orchestrator summary records `mode=comment-update-branch`, and the next keepalive instruction proceeds.
- [ ] If the head does **not** move within `TTL_short`, the Actions UI shows a run of **Keepalive Branch Sync**, and the summary logs `dispatched=keepalive-branch-sync`, the HTTP status, and the trace identifier.
- [ ] After the fallback completes successfully, the PR head SHA changes within `TTL_long`, the summary records `mode=action-sync-pr` with the merged SHA, and keepalive continues.
- [ ] If neither path advances the branch within `TTL_long`, the automation applies `agents:sync-required`, posts no new instruction, and the summary notes `reason=sync-timeout trace:{TRACE}`.
- [ ] Re-running the same `{PR, round, trace}` key never emits duplicate comments or dispatches.
- [ ] Guardrail failures (missing labels, Gate incomplete, run cap exceeded) remain PR-noise free and surface only in `$GITHUB_STEP_SUMMARY`.

## Implementation Notes
- Use ACTIONS_BOT_PAT whenever possible for both commenting and workflow dispatch; fall back to SERVICE_BOT_PAT solely for the comment command when needed.
- Restrict execution to origin-repo PRs and mask tokens in logs; avoid introducing new secrets.
- Store previous head SHAs via hidden PR comments (e.g., `<!-- keepalive-last-sha:{SHA} trace:{TRACE} -->`) or durable artifacts and read them safely before comparisons.
- Keep logging concise—limit PR chatter to the `/update-branch` command (and optional debug escalation) while writing detailed traces to `$GITHUB_STEP_SUMMARY`.

## Progress Tracker
- **Current status:** No tasks or acceptance criteria have been satisfied yet; all scope items remain open for implementation.
- **In progress:**
  - Finalizing the persistence helper that wraps GitHub comment read/write calls. The helper will:
    1. Search existing PR comments for `<!-- keepalive-last-sha:{SHA} trace:{TRACE} -->` matching `{PR, round}` and return `{head_sha, trace}` when found.
    2. Create or update the hidden comment when a new `{PR, round, trace}` tuple is observed, ensuring idempotency by comparing stored vs. observed values before writing.
    3. Emit structured summary logs (`persist_state=hit` | `persist_state=write`) without surfacing user-visible comments beyond the hidden marker.
  - Designing the comment-first `/update-branch trace:{TRACE}` executor sequence. The design now captures:
    1. Credential selection flow (prefer ACTIONS_BOT_PAT with SERVICE_BOT_PAT fallback when comment permissions fail).
    2. Comment emission with deterministic body formatting and immediate 👀 reaction on the created comment.
    3. `$GITHUB_STEP_SUMMARY` entry schema: ``comment_update: {"round":<int>,"trace":"<id>","comment_id":<int>,"url":"<html_url>","actor":"stranske"}``.
    4. Short-TTL polling loop parameters (24 iterations × 5s) and success logging (`mode=comment-update-branch`, `new_head_sha`).
- **Next step:** Build the executor helper functions (`persist_head_sha`, `post_update_branch_comment`, `await_head_advance`) so the implementation can begin toggling task checkboxes once end-to-end comment mode succeeds in dry-run testing.
- **Notes toward upcoming acceptance criteria:**
  - Unsynced detection will hinge on comparing the stored `head_sha` with the live head fetched via the GitHub API. A mismatch will move the orchestrator straight to the next keepalive round without comment noise; an exact match will trigger the comment-first flow.
  - The polling loop will cap at 24 iterations (≈2 minutes at 5-second intervals) to satisfy the `TTL_short` expectation before escalating to the fallback workflow.
  - Guardrail enforcement will remain summary-only by exiting early before comment/post flows when required labels or gate states are missing.
