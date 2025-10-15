<!-- bootstrap for codex on issue #2609

Source Issue: https://github.com/stranske/Trend_Model_Project/issues/2609
Topic GUID: 74de39f9-545c-551e-98b6-7d5b842282db

Focus: consolidate CI/automation workflows, align docs, and enforce branch protection after migrating Gate + Maint automation.
Key checkpoints:
- Gate: add docs-only fast path and retire PR-14.
- Autofix consolidation: centralize Maint-46 and deprecate PR-02 path.
- Compatibility cleanup: retire Maint-47 and ensure the legacy consumer shims
  (former Agents-61/62) stay removed.
- Collapse self-tests into one parameterized runner; remove wrappers.
- Lock agents workflow to orchestrator + bridge; publish agent issue template.
- Update branch protection to require Gate on default branch.
- Publish Workflow System Overview updates and sync documentation.
- Align mypy execution with pinned Python version.

Out of scope: modifying test logic, expanding coverage, or adding new languages/toolchains.
Done means: all child issues closed and Health-44 confirms branch protection settings.
-->
