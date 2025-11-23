# Diagnostics keepalive checklist

This checklist is scoped to the diagnostics keepalive run. Use a separate file under `docs/keepalive/status/` for each concurrent keepalive effort to avoid PR conflicts.

## Scope
- [x] Enumerate early-exit conditions across preprocessing, calendar alignment, missing-data filtering, and selection stages.
- [x] Replace bare `None` returns with status objects or warnings that include reason codes and key context (e.g., empty timestamps, all funds filtered by missing-data policy).
- [x] Ensure CLI/API surfaces these diagnostics to users (e.g., logged warnings or structured responses).

## Tasks
- [x] Catalogue current `None` return paths and add reason codes/messages.
- [x] Thread diagnostics through the pipeline and surface them in user-facing logs/outputs.
- [x] Add tests verifying each early-exit condition reports the expected diagnostic.

## Acceptance criteria
- [x] Every early-exit path returns a diagnostic payload instead of silent `None`.
- [x] CLI/API outputs clearly state why a run ended early, with tests covering representative cases.

## Progress
- Diagnostics now replace all previously catalogued `None` returns with structured `DiagnosticResult` payloads carrying reason codes and context, with CLI/reporting surfacing the messages. Tests cover each early-exit path to confirm diagnostics propagate to user-facing output.
