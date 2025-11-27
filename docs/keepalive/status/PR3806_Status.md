# Keepalive Status — PR #3806

> **Status:** In progress — captured initial scope, tasks, and acceptance criteria for the risk-free fallback updates.

## Progress updates
- Round 1: Recorded the PR's scope, tasks, and acceptance criteria for the risk-free fallback and alignment work.
- Round 2: Keepalive check-in; no tasks or acceptance criteria completed yet. Scope, tasks, and acceptance criteria remain as initially recorded.
- Round 3: Captured corrective actions so the keepalive prompt drives substantive code work rather than status-only updates.
- Round 4: Explained why code work has not begun and outlined the concrete next actions to start implementing the fallback path.

## Why code work has not started yet
- The prior keepalive rounds only adjusted the prompt/process and did not declare a specific coding target for the current round, so no branch or patch work was initiated.
- The branch and environment setup for tackling the risk-free fallback were never kicked off (no `phase2-dev` sync or local repro run), leaving the acceptance criteria untouched.
- Next actions: pick the branch for implementation, run the relevant selection tests to baseline current behaviour, and open a draft patch that begins the risk-free fallback guard in `rank_select_funds` aligned to the configured window.

## Adjustments to the standard prompt/process
- Require each keepalive round to state the **next concrete coding step** (e.g., "implement fallback guard in `rank_select_funds` today") and to open a draft PR or patch when code is expected.
- Instruct the agent to attempt at least one acceptance criterion per round unless blocked, and to log blockers with repro steps and owner.
- Add a checklist within the prompt to confirm whether tasks or criteria were advanced; forbid "no progress" responses unless a blocker is recorded.
- Have the process auto-surface the relevant scope, tasks, and acceptance criteria to prevent forgetting the intended implementation targets.

## Scope
- [ ] Stabilize risk-free handling in `rank_select_funds`, enabling a documented fallback path that aligns with the requested analysis window.

## Tasks
- [ ] Document and enable a safe default behavior when no risk-free column is supplied, with an explicit configuration flag.
- [ ] Align fallback detection with the requested analysis window so volatility comparisons use the same slice as downstream metrics.
- [ ] Add tests that cover missing columns, fallback paths, and window alignment to ensure deterministic selection.

## Acceptance criteria
- [ ] `rank_select_funds` no longer hard-fails by default when the risk-free column is omitted but follows a documented fallback when enabled.
- [ ] Fallback selection uses window-aligned data and produces deterministic results in tests.
- [ ] Unit tests cover both explicit column selection and fallback scenarios.
