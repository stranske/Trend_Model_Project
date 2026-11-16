## Keepalive Next Task

Your objective is to satisfy the **Acceptance Criteria** by completing each **Task** within the defined **Scope**.

**This round you MUST:**
1. Implement actual code or test changes that advance at least one incomplete task toward acceptance.
2. Commit meaningful source code (.py, .yml, .js, etc.)—not just status/docs updates.
3. Mark a task checkbox complete ONLY after verifying the implementation works.
4. Focus on the FIRST unchecked task unless blocked, then move to the next.

**Guidelines:**
- Do not create branches or pull requests; commit only to the checked-out PR branch.
- Keep edits scoped to the current task rather than reshaping the entire PR.
- Use repository instructions, conventions, and tests to validate work.
- Prefer small, reviewable commits; leave clear notes when follow-up is required.
- Do NOT work on unrelated improvements until all PR tasks are complete.

## Pre-Commit Formatting Gate (Black)

Before you commit or push any Python (`.py`) changes, you MUST:
1. Run Black to format the relevant files (line length 100).
2. Verify formatting passes CI by running:
   `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .`
3. If the check fails, do NOT commit/push; format again until it passes.

**COVERAGE TASKS - SPECIAL RULES:**
If a task mentions "coverage" or a percentage target (e.g., "≥95%", "to 95%"), you MUST:
1. After adding tests, run TARGETED coverage verification to avoid timeouts:
   - For a specific script like `scripts/foo.py`, run:
     `pytest tests/scripts/test_foo.py --cov=scripts/foo --cov-report=term-missing -m "not slow"`
   - If no matching test file exists, run:
     `pytest tests/ --cov=scripts/foo --cov-report=term-missing -m "not slow" -x`
2. Find the specific script in the coverage output table
3. Verify the `Cover` column shows the target percentage or higher
4. Only mark the task complete if the actual coverage meets the target
5. If coverage is below target, add more tests until it meets the target

IMPORTANT: Always use `-m "not slow"` to skip slow integration tests that may timeout.
IMPORTANT: Use targeted `--cov=scripts/specific_module` instead of `--cov=scripts` for faster feedback.

A coverage task is NOT complete just because you added tests. It is complete ONLY when the coverage command output confirms the target is met.

**The Tasks and Acceptance Criteria are provided in the appendix below.** Work through them in order.

## Run context
---
## PR Tasks and Acceptance Criteria

**Progress:** 7/7 tasks complete, 0 remaining

### Scope
Resolve one authoritative evaluated user-weighted return stream for both cost-enabled and cost-absent runs, and build NAV from that stream with the existing first-value-1.0 normalization. Preserve base trading costs and the existing canonical first-strategy-per-path bundle convention.

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] src/trend_analysis/monte_carlo/runner.py:694-797 — pass net_returns if not None else gross_returns to _extract_nav_series tests/monte_carlo/test_runner.py / test_results.py — two-strategy regression and zero-bps invariance; check metadata nav_paths and parquet export

### Additional PR tasks

- [x] `python3 -m pytest tests/monte_carlo/test_runner.py::test_nav_uses_user_weighted_returns_without_scenario_costs -q`
- [x] `python3 -m pytest tests/monte_carlo/test_runner.py -q`

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] Named test `tests/monte_carlo/test_runner.py::test_nav_uses_user_weighted_returns_without_scenario_costs` (two strategies 80/20 vs 20/80, no costs -> distinct NAVs reconciled to user-weighted OOS returns; enabling zero-bps cost leaves NAV dates/values identical) passes: `python3 -m pytest tests/monte_carlo/test_runner.py::test_nav_uses_user_weighted_returns_without_scenario_costs -q`
- [x] Deliberate break → revert, shown in the PR: Pass returns=None to _extract_nav_series when costs absent (current behaviour) -> identical NAVs, test fails; record that the named test fails, then revert and record that it passes.
- [x] The whole test file still passes: `python3 -m pytest tests/monte_carlo/test_runner.py -q`
- [x] The invariance check runs with zero base trading costs; a separate case with nonzero base costs shows they are still applied (runner.py:685-689 removes engine base costs when scenario costs are enabled).

### Recently Attempted Tasks
Avoid repeating these unless a task needs explicit follow-up:

- src/trend_analysis/monte_carlo/runner.py:694-797 — pass net_returns if not None else gross_returns to _extract_nav_series tests/monte_carlo/test_runner.py / test_results.py — two-strategy regression and zero-bps invariance; check metadata nav_paths and parquet export
- no-focus

---
