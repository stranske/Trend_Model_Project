# Keepalive Status — PR #3806

> **Status:** In progress — captured initial scope, tasks, and acceptance criteria for the risk-free fallback updates.

## Progress updates
- Round 1: Recorded the PR's scope, tasks, and acceptance criteria for the risk-free fallback and alignment work.
- Round 2: Keepalive check-in; no tasks or acceptance criteria completed yet. Scope, tasks, and acceptance criteria remain as initially recorded.
- Round 3: Captured corrective actions so the keepalive prompt drives substantive code work rather than status-only updates.
- Round 4: Explained why code work has not begun and outlined the concrete next actions to start implementing the fallback path.
- Round 5: Implemented window-aligned risk-free fallback selection, validated through new coverage, and confirmed the opt-in fallback flag gates behaviour when a dedicated risk-free column is absent.

## Why code work was blocked and how it was resolved
- Earlier rounds never declared a concrete coding target, so no patch was started; Round 5 explicitly delivered the window-aligned fallback guard and exercised it in tests.
- Running the focused risk-free handling tests on this branch unblocked validation of the fallback behaviour and cleared the acceptance criteria.
- Next actions: propagate the window-aligned fallback into any downstream configs that rely on the heuristic and expand coverage if new edge cases appear.

## Adjustments to the standard prompt/process
- Require each keepalive round to state the **next concrete coding step** (e.g., "implement fallback guard in `rank_select_funds` today") and to open a draft PR or patch when code is expected.
- Instruct the agent to attempt at least one acceptance criterion per round unless blocked, and to log blockers with repro steps and owner.
- Add a checklist within the prompt to confirm whether tasks or criteria were advanced; forbid "no progress" responses unless a blocker is recorded.
- Have the process auto-surface the relevant scope, tasks, and acceptance criteria to prevent forgetting the intended implementation targets.

## Scope
- [x] Stabilize risk-free handling in `rank_select_funds`, enabling a documented fallback path that aligns with the requested analysis window.

## Tasks
- [x] Document and enable a safe default behavior when no risk-free column is supplied, with an explicit configuration flag.
- [x] Align fallback detection with the requested analysis window so volatility comparisons use the same slice as downstream metrics.
- [x] Add tests that cover missing columns, fallback paths, and window alignment to ensure deterministic selection.

## Acceptance criteria
- [x] `rank_select_funds` no longer hard-fails by default when the risk-free column is omitted but follows a documented fallback when enabled.
- [x] Fallback selection uses window-aligned data and produces deterministic results in tests.
- [x] Unit tests cover both explicit column selection and fallback scenarios.
