# Issue #6094 closer acceptance

Check date: 2026-11-03. Base: current `main`,
`12e324f97bebf4b226b7039bec4948f894510e7e`, including merged PR #6116.

Outcome 1: the existing real runner-count assertion rejects a strategy-only
cache disconnect with `12 == 6` in the single-period case; normal single- and
multi-period evaluation passes the runner gates and complete cache suite.
The input is `_real_runner`: deterministic stationary bootstrap, seed 123,
one path, one versus three identical strategies, and real metric kernels.

Command printing the control and actual counts:

```sh
.venv/bin/python -m pytest tests/monte_carlo/test_cache.py -k strategy_only_disconnect -s -q
```

Stop rule: three repair attempts or 45 minutes. On reaching either limit,
stop at the first broken step and report it. No production change is in scope.

Runtime: no project virtual environment existed. The owner-selected fallback
`/opt/anaconda3/bin/python3` (3.12.2) failed before collection with
`AttributeError: module 'numpy' has no attribute 'long'` while importing SciPy
(exit 4). It also falls below this repository's Python >=3.13 requirement.
Created the conventional `.venv` selected by `scripts/setup_env.sh`, using
installed Homebrew Python 3.13.13 and `uv pip sync requirements.lock` (exit 0).
Shared packages and launchers were not changed.

Gate/drain pair: hosted CI and review remain required for merge; while those
gates are closed, this branch can provide local control/restoration evidence
and address findings in the two authorized files. This record is local
acceptance evidence, not merge or post-merge verification evidence.


## Result and scope

- [x] Provide a reproducible strategy-only disconnect control demonstrating the required 12-versus-6 failure.
- [x] Record passing restored runner gates and the complete cache regression suite.

The prior attempt's selective cache bypass was correct and was retained. Its
loose `actual_multi > actual_single` check and duplicate runner-count assertion
were replaced with invocation of the **unchanged original runner-count test**.
The control now requires its actual final assertion to raise `assert 12 == 6`
(single period) or `assert 111 == 39` (multi-period). Other assertion failures
cannot satisfy that expectation. Both `_generate_path_context` calls still enter
the real cache context; the test checks the exact two generation/four evaluation
scope sequence. The scoped patch is undone and that same original test must pass.
The negative-control test exits 0 because the precise rejection is expected;
removing the disconnect makes it fail with `DID NOT RAISE`.

Observed one-strategy / three-strategy kernel counts from the inherited control,
run before changing the test (exit 0, 2 passed in 5.82s):

| Mode | Strategy-only disconnected | Restored |
| --- | --- | --- |
| Single period | 6 / 12 | 6 / 6 |
| Multi-period | 39 / 111 | 39 / 39 |

These are measurements, not an assumed doubling for all configurations. The
multi-period fixture uses monthly evaluation, a six-month in-sample window and
one-month out-of-sample window across 2023; its required rejection is 111 versus
39, not 12 versus 6. The permanent control checks both observed pairs.

Only `tests/monte_carlo/test_cache.py` and this record differ from main.
Production files, the original runner gate and merged PR #6116 remain unchanged.
The complete suite below is this issue's cache suite, not the entire repository.
Existing pandas deprecation warnings and the optional missing data-file warning
are retained in the output; no warning suppression or dataset stub was added.
Black (`--check --line-length 100`), Ruff and `git diff --check` passed (exit 0).

## Retained command output

Commands ran at the repository root with the selected `.venv/bin/python`
(Python 3.13.13), using `python -m pytest` to bind pytest to that interpreter.
Each command completed in under two minutes. The control captures the original
failed assertion text and then verifies restoration; the following two commands
run without any disconnection patch installed.

### Strategy-only negative control

```sh
.venv/bin/python -m pytest tests/monte_carlo/test_cache.py -k strategy_only_disconnect -s -q
```

Exit status: 0.

```text
multi_period=False: original runner gate rejected strategy-only disconnect
assert 12 == 6
 +  where 12 = <Mock id='4753053344'>.call_count
.multi_period=True: original runner gate rejected strategy-only disconnect
assert 111 == 39
 +  where 111 = <Mock id='4753062080'>.call_count
.
=============================== warnings summary ===============================
tests/monte_carlo/test_cache.py: 56 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/stages/portfolio.py:880: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    "User": out_user.astype(float, copy=False),

tests/monte_carlo/test_cache.py: 56 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/stages/portfolio.py:881: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    "Equal-Weight": out_ew.astype(float, copy=False),

tests/monte_carlo/test_cache.py: 56 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/analysis/results.py:86: UserWarning: data file missing: /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/data/Trend Universe Data.csv
    warnings.warn(message, UserWarning)

tests/monte_carlo/test_cache.py: 48 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/multi_period/engine.py:766: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    prepared = df.astype(float, copy=True)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 passed, 19 deselected, 216 warnings in 3.94s
```

### Restored runner gates

```sh
.venv/bin/python -m pytest tests/monte_carlo/test_cache.py -k runner_kernel_calls
```

Exit status: 0.

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
plugins: langsmith-0.10.9, cov-7.1.0, xdist-3.8.0, rerunfailures-16.3, datadir-1.8.0, typeguard-4.5.1, hypothesis-6.155.7, regressions-2.11.0, anyio-4.13.0
collected 21 items / 19 deselected / 2 selected

tests/monte_carlo/test_cache.py ..                                       [100%]

=============================== warnings summary ===============================
tests/monte_carlo/test_cache.py: 28 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/stages/portfolio.py:880: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    "User": out_user.astype(float, copy=False),

tests/monte_carlo/test_cache.py: 28 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/stages/portfolio.py:881: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    "Equal-Weight": out_ew.astype(float, copy=False),

tests/monte_carlo/test_cache.py: 28 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/analysis/results.py:86: UserWarning: data file missing: /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/data/Trend Universe Data.csv
    warnings.warn(message, UserWarning)

tests/monte_carlo/test_cache.py: 24 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/multi_period/engine.py:766: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    prepared = df.astype(float, copy=True)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================ 2 passed, 19 deselected, 108 warnings in 2.25s ================
```

### Complete cache regression suite

```sh
.venv/bin/python -m pytest tests/monte_carlo/test_cache.py -q
```

Exit status: 0.

```text
.....................                                                    [100%]
=============================== warnings summary ===============================
tests/monte_carlo/test_cache.py: 135 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/stages/portfolio.py:880: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    "User": out_user.astype(float, copy=False),

tests/monte_carlo/test_cache.py: 135 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/stages/portfolio.py:881: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    "Equal-Weight": out_ew.astype(float, copy=False),

tests/monte_carlo/test_cache.py: 135 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/analysis/results.py:86: UserWarning: data file missing: /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/data/Trend Universe Data.csv
    warnings.warn(message, UserWarning)

tests/monte_carlo/test_cache.py: 108 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Trend_Model_Project__6094__closer-repair/src/trend_analysis/multi_period/engine.py:766: Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.
    prepared = df.astype(float, copy=True)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
21 passed, 513 warnings in 10.31s
```
