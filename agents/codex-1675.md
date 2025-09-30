<!-- bootstrap for codex on issue #1675 -->

## Task List
- [ ] Relocate `sitecustomize.py` to `src/trend_model/_sitecustomize.py` (or remove if functionality is obsolete).
- [ ] Gate any remaining behavior behind `TREND_MODEL_SITE_CUSTOMIZE="1"` so no code executes without the opt-in flag.
- [ ] Update imports or packaging hooks to reference the relocated module when the flag is enabled.
- [ ] Add a unit test that imports an arbitrary module from the project and asserts no side effects occur when the env var is unset.
- [ ] Ensure documentation and packaging metadata no longer mention the root-level `sitecustomize.py`.
- [ ] Run the full test suite to confirm zero references to the old root-level module remain.

## Acceptance Criteria
- No implicit imports or side effects occur when importing the package without setting the opt-in environment variable.
- Test output shows zero references to a top-level `sitecustomize.py` file.
