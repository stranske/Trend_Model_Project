<!-- bootstrap for codex on issue #2884 -->

## Scope
- [x] Agents control plane: Orchestrator only; workers callable; stronger Guard patterns

## Task Checklist
- [x] Remove all `on: schedule` from belt/dispatcher/worker/conveyor workflows; expose them via `workflow_call` with typed inputs.
- [x] Update Orchestrator to call the belt jobs with explicit parameters and short circuit when no labeled work exists.
- [x] Expand Agents Guard from explicit filenames to glob `.github/workflows/agents-*.yml`, requiring `agents:allow-change` to proceed.
- [x] Add a single “keepalive” mechanism inside Orchestrator rather than per-worker.

## Acceptance Criteria
- [x] Only Orchestrator runs on a schedule; all other agent workflows show “callable only.”
- [x] Agents Guard blocks edits to any agents workflow unless the allow-label is present.
- [x] Run volume drops meaningfully, with no loss of function (workers still execute when called).
