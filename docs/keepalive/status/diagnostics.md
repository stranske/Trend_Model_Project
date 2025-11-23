# Diagnostics keepalive checklist

This checklist is scoped to the diagnostics keepalive run. Use a separate file under `docs/keepalive/status/` for each concurrent keepalive effort to avoid PR conflicts.

## Scope
- [ ] Enumerate early-exit conditions across preprocessing, calendar alignment, missing-data filtering, and selection stages.
- [ ] Replace bare `None` returns with status objects or warnings that include reason codes and key context (e.g., empty timestamps, all funds filtered by missing-data policy).
- [ ] Ensure CLI/API surfaces these diagnostics to users (e.g., logged warnings or structured responses).

## Tasks
- [ ] Catalogue current `None` return paths and add reason codes/messages.
- [ ] Thread diagnostics through the pipeline and surface them in user-facing logs/outputs.
- [ ] Add tests verifying each early-exit condition reports the expected diagnostic.

## Acceptance criteria
- [ ] Every early-exit path returns a diagnostic payload instead of silent `None`.
- [ ] CLI/API outputs clearly state why a run ended early, with tests covering representative cases.

## Progress
- No checklist items have been met yet. Re-run the scope/tasks/acceptance lists as work lands and check items off only once the
  corresponding acceptance criteria are satisfied.
