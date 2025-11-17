<!-- bootstrap for codex on issue #3645 -->

## Scope

- Harden config handling so typos in keys or wrong value types are caught before the runtime pipeline explodes.
- Keep the schema definition in a single shared module that both the CLI and application code import so behaviour never drifts.
- Stick to stdlib/dataclasses plus handwritten validation so we do not balloon dependencies or break CI expectations.

## Task list

- [ ] Define a lightweight config schema for the core knobs (data paths, universe file, frequency, costs, etc.).
- [ ] Validate configs on startup, including defaults and helpful error text for missing/invalid values.
- [ ] Add tests that cover missing required fields and wrong types so regressions fail quickly.

## Acceptance criteria

- [ ] Invalid configs fail fast with a single, clear error message.
- [ ] Valid configs round-trip through load + validate without mutation or data loss.
