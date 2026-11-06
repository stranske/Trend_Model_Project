# Issue #6188 closer acceptance

Check date: 2026-11-06. Base: `main` at merge of PR **#6192** (`a2e82d2961823393f6e73f4db7dc672a3d3cc607`).

> Branch integration: `claude/issue-6188` had **no commits ahead of** `origin/main` after #6192 landed; nothing to merge from that branch.

## Required CI gate name (Evidence runner / verifier)

- **`Gate / gate`** — GitHub commit status context published by the `gate-summary` job in `.github/workflows/pr-00-gate.yml`.

## Acceptance criteria

- [x] Record runnable acceptance commands that the Evidence runner can discover (`scripts/runner_lib/acceptance_gate.py` reads backticks under **Acceptance criteria** only).
- [x] Demonstrate the named gate on regenerated PR **#6187** and a passing local repository test command.
- [x] Record a deliberate regression failure followed by restored success.
- [x] Inspect both review threads on **#6192**; confirm disposition (see Review threads).
- [x] Capture the next controller-owned sync-core classification round for this consumer without this finding (see Sync classification).

## Runnable commands (Evidence runner)

Use a **Python 3.13** interpreter with `pytest==9.2.0` and `pytest-xdist==3.8.0` (matches Gate job `xdist-loadgroup-matrix` for the 9.2 matrix leg). Example: create a local venv, `pip install -e ".[dev,app]"`, then `pip install pytest==9.2.0 pytest-xdist==3.8.0`.

From the repository root (`PYTHONPATH` is not required for these; subprocesses in tests set it as needed):

- `python -m pytest tests/test_xdist_loadgroup_compat.py -q`
- `python -m pytest tests/golden/test_demo.py -n 2 --dist loadgroup -v --tb short`
- Deliberate break (expect non-zero exit on pytest 9.2+): `TREND_DISABLE_XDIST_LOADGROUP_COMPAT=1 python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q`
- Restore: `python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q`

Documented test script (README / `AGENTS.md`): `./scripts/run_tests.sh`. That script normally measures coverage with `fail_under`; a **narrow** selection can pass pytest yet fail the **coverage report step** unless you opt out (this PR adds `TREND_SKIP_COVERAGE_REPORT=1` for focused gates only). Match the Gate matrix by using the same interpreter as for the pytest 9.2 commands above, then:

`TREND_SKIP_COVERAGE_REPORT=1 ./scripts/run_tests.sh tests/test_xdist_loadgroup_compat.py tests/golden/test_demo.py -n 0 -q`

(`-n 0` overrides the script’s default `-n auto` so golden subprocesses are not starved by xdist workers on the shared machine.)

## Local run log (Python 3.13.13, pytest 9.2.0, pytest-xdist 3.8.0)

```text
$ .venv-6188-evidence/bin/python -m pytest tests/test_xdist_loadgroup_compat.py -q
.......                                                                  [100%]
7 passed in 65.17s (0:01:05)
```

```text
$ .venv-6188-evidence/bin/python -m pytest tests/golden/test_demo.py -n 2 --dist loadgroup -v --tb short
...
============================== 4 passed in 17.25s ==============================
```

```text
$ TREND_DISABLE_XDIST_LOADGROUP_COMPAT=1 .venv-6188-evidence/bin/python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q
...
E       AssertionError: xdist_group split across workers: ['gw0', 'gw1']
...
1 failed in 3.68s
```

```text
$ .venv-6188-evidence/bin/python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q
.                                                                        [100%]
1 passed in 3.61s
```

```text
$ PATH="$PWD/.venv-6188-evidence/bin:$PATH" VIRTUAL_ENV="$PWD/.venv-6188-evidence" TREND_SKIP_COVERAGE_REPORT=1 ./scripts/run_tests.sh tests/test_xdist_loadgroup_compat.py tests/golden/test_demo.py -n 0 -q
...
=========================== short test summary info ============================
SKIPPED [1] tests/test_xdist_loadgroup_compat.py:174: late ordering regression is specific to pytest >= 9.2
SKIPPED [1] tests/test_xdist_loadgroup_compat.py:227: restoration regression is specific to pytest >= 9.2
9 passed, 2 skipped in 55.24s
```

(`requirements.lock` pins **pytest 9.1.1** after `pip install -r requirements.lock`; use a matrix venv with **pytest 9.2.0** when you need parity with the `xdist-loadgroup-matrix` 9.2 leg.)

## PR #6187 — `Gate / gate` and `Python CI / python 3.13`

PR **#6187** is **closed** (deps sync). Hosted checks on the last Gate workflow run for that head include:

- **`Gate / gate`**: **fail** — summary: `Python CI failed` (workflow run `41956109404`).
- **`Python CI / python 3.13`**: **fail** (same run).

Inspect: `gh pr checks 6187 --repo stranske/Trend_Model_Project` and `gh run view 41956109404 --repo stranske/Trend_Model_Project`.

## PR #6192 — review threads (CodeRabbit)

Two threads; both **`isResolved: true`** in GitHub’s review thread API.

| Thread | File (approx.) | Disposition |
| --- | --- | --- |
| 1 | `tests/_xdist_loadgroup_compat.py` — serial / hook order | Addressed in **`20e9b764`**: `pytest_collection_modifyitems` uses `tryfirst=True` in `tests/conftest.py`; serial probe + regression tests added. CodeRabbit acknowledged resolution. |
| 2 | `tests/test_xdist_loadgroup_compat.py` — pytest **9.2** coverage | Addressed in **`20e9b764`** + **`a2e82d29`**: Gate `xdist-loadgroup-matrix` runs **9.1.1** and **9.2.0** with **pytest-xdist 3.8.0**; collect-only assertion revised to not rely on `@serial` in plain `--collect-only` output. Follow-up nit on `--collect-only` was superseded by worker-captured node IDs in probes. |

## Sync classification (controller-owned, next scheduled round)

Captured after this work from **Health 68 Consumer Sync Drift Check** (Workflows), run **`42234189419`** (`2026-11-06T14:28:14Z`):

- Job **`Validate consumer repo drift`**: **`success`**
- Jobs **`Publish actionable drift alert`** / **`Resolve recovered drift alert`**: **`skipped`**

Interpretation: the scheduled round **compared** registered consumers and found **no actionable drift** requiring an alert for this repo class (alerts skipped). **This evidence file is not input to that workflow**; classification is controller-owned on the Workflows side.

Refs **#6188**. <!-- workflow-source:local_request -->
