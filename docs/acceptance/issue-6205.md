# Acceptance record: stranske/Trend_Model_Project#6205

<!-- closer-core/v1 repair-of=stranske/Trend_Model_Project#6205@6206 -->

## Check date, outcome and stop rule

Initial check: 2026-11-08 UTC (continuation began at 20:36:03Z). Follow-up
supported-runtime verification (#6208, PR #6221): 2026-11-12 UTC. Expected
outcome: the regenerated #6204 head has successful Python 3.14 CI, the
`_copy_install_sources` exclusion and deliberate-break regression probe are
verified under Python >=3.13, the repository's supported local suite command and
its successful execution are recorded on the identified checkout, and a later
controller-owned sync-core round no longer reports the copy-race finding.

Stop after three acceptance attempts or at 21:06:03Z on the initial 2026-11-08
continuation, whichever comes first; report the first failing or unproven step.
The 2026-11-08 local closer dispatch explicitly prohibited a local full-suite or
release/verify run on that host (where only fallback `/opt/anaconda3/bin/python3`
at Python 3.12.2 was available); that historical dispatch prohibition is
preserved here. On 2026-11-12, the owner authorized a hosted execution context
for #6208 (`/opt/hostedtoolcache/Python/3.14.10/x64/bin/python`, Python 3.14.10,
`pytest 9.2.0`, `pytest-cov 7.1.0`, `coverage 7.17.2`, `pytest-xdist 3.8.0`)
that satisfies `pyproject.toml` (`requires-python = ">=3.13"`) and permits
executing the isolated deliberate-break probe and full local test suite.

Commands and observed outputs follow. On 2026-11-08, `git branch --list 'closer*/issue-6205'`
returned only `closer-r1/issue-6205`, whose acceptance record commit was
`cd04225e449c38f0f6478afebbd240894ca7fd79` (merged in #6207 as
`362fb9cf6ee8dc245df0a1c639c6ed318fe64153`). Primary fix PR #6206 merged to
`main` as `f77623978ffafaeaa5f40b486717063bd1bd1a36`. On 2026-11-12, PR #6221
branched from `main` commit `6f4fd58e99633aaaa480815966d678926fcef7c9` with
bootstrap head `0b0348f8fea59683199d5a08a4f2d4ade1faf680` (`origin/main`
subsequently advanced to `4038b50a5013663b9547d0602b99839929e179d1`;
`tests/test_cli_installed.py`, `pyproject.toml`, `scripts/run_tests.sh`, and
`.github/workflows/ci.yml` are byte-for-byte identical across all three commits).
No product implementation or dependency pins in `tests/test_cli_installed.py`
were altered.

## 1. Exact-head Python 3.14 CI: passed

```bash
gh pr view 6204 --repo stranske/Trend_Model_Project --json headRefOid,statusCheckRollup
gh run view 42518199588 --repo stranske/Trend_Model_Project
gh run view 42518199588 --repo stranske/Trend_Model_Project \
  --json headSha,attempt,conclusion,event,jobs
gh run view 42518199588 --repo stranske/Trend_Model_Project \
  --json headSha,attempt,conclusion,jobs
gh run view 42518199588 --repo stranske/Trend_Model_Project \
  --job 128848973787 --log
```

The PR #6204 head and run `headSha` both equal
`cde9f21439ba2f72ca596f54cd017c7673c6221b`. Re-verified on 2026-11-12 with
`gh run view 42518199588 --repo stranske/Trend_Model_Project --json headSha,attempt,conclusion,jobs`:

```json
{
  "attempt": 2,
  "conclusion": "success",
  "headSha": "cde9f21439ba2f72ca596f54cd017c7673c6221b",
  "jobs": [
    {
      "completedAt": "2026-11-08T20:20:34Z",
      "conclusion": "success",
      "databaseId": 128848973787,
      "name": "Python CI / python 3.14",
      "startedAt": "2026-11-08T20:16:27Z",
      "status": "completed",
      "url": "https://github.com/stranske/Trend_Model_Project/actions/runs/42518199588/job/128848973787"
    }
  ]
}
```

Run attempt **2**, triggered by `pull_request`, completed with `conclusion: "success"`.
Its `Python CI / python 3.14` job `128848973787` ran from 20:16:27Z to 20:20:34Z
with conclusion `success`:

```text
7039 passed, 5 skipped, 4088 warnings in 176.81s (0:02:56)
All checks passed!
```

[Exact-head Python 3.14 job](https://github.com/stranske/Trend_Model_Project/actions/runs/42518199588/job/128848973787).
Attempt 2 supersedes the earlier failed classification/skipped jobs. The former
record's inference from base-commit run `42516821134` was invalid and remains
removed; a successful base run does not verify a descendant head.

## 2. Supported-runtime inspection, deliberate-break probe, and local suite execution: passed

### 2.1 Inspection of `_copy_install_sources` and focused run under Python >=3.13

On `main` (`f77623978ffafaeaa5f40b486717063bd1bd1a36` through
`4038b50a5013663b9547d0602b99839929e179d1`) and checkout
`0b0348f8fea59683199d5a08a4f2d4ade1faf680`,
`tests/test_cli_installed.py::_copy_install_sources(project_root: Path, source_copy: Path)`
calls `shutil.copytree(project_root, source_copy, dirs_exist_ok=True, ignore=_INSTALL_COPY_IGNORE)`,
where `_INSTALL_COPY_IGNORE` is constructed with the 20-entry exclusion tuple:

```python
(
    ".env",
    ".env.*",
    "secrets.toml",
    ".git",
    ".github",
    ".venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".workflows-lib",
    "__pycache__",
    "*.pyc",
    "node_modules",
    "archives",
    "notebooks",
    "build",
    "dist",
    "*.egg-info",
    ".fast_demo_mode",
)
```

Executed in the repository-selected Python 3.14.10 environment
(`/opt/hostedtoolcache/Python/3.14.10/x64/bin/python`, `pytest 9.2.0`) on
checkout `0b0348f8fea59683199d5a08a4f2d4ade1faf680`:

```bash
python --version
python -m pytest -q tests/test_cli_installed.py -k install_source_copy
```

```text
Python 3.14.10
..                                                                       [100%]
2 passed, 2 deselected, 13 warnings in 0.83s
```

### 2.2 Isolated-checkout deliberate break and restoration under Python >=3.13

In an isolated detached worktree (`/tmp/issue-6205-probe-XXXXXX` at
`0b0348f8fea59683199d5a08a4f2d4ade1faf680`), removed only the `".fast_demo_mode"`
entry from `_INSTALL_COPY_IGNORE` in `tests/test_cli_installed.py` without
committing the break, ran the targeted regression test, restored the file, and
re-ran the test under Python 3.14.10:

```bash
# In isolated worktree with ".fast_demo_mode" removed from _INSTALL_COPY_IGNORE:
python -m pytest -q tests/test_cli_installed.py::test_install_source_copy_excludes_disappearing_demo_marker
```

Observed failure output with the exclusion removed:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
__________ test_install_source_copy_excludes_disappearing_demo_marker __________
...
tests/test_cli_installed.py:153: in test_install_source_copy_excludes_disappearing_demo_marker
    _copy_install_sources(project_root, source_copy)
tests/test_cli_installed.py:54: in _copy_install_sources
    shutil.copytree(
...
E           shutil.Error: [('/tmp/pytest-of-runner/pytest-1/test_install_source_copy_exclu0/project/demo/.fast_demo_mode', '/tmp/pytest-of-runner/pytest-1/test_install_source_copy_exclu0/source-copy/demo/.fast_demo_mode', "[Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-1/test_install_source_copy_exclu0/project/demo/.fast_demo_mode'")]
=========================== short test summary info ============================
FAILED tests/test_cli_installed.py::test_install_source_copy_excludes_disappearing_demo_marker - shutil.Error: [('/tmp/pytest-of-runner/pytest-1/test_install_source_copy_exclu0/project/demo/.fast_demo_mode', '/tmp/pytest-of-runner/pytest-1/test_install_source_copy_exclu0/source-copy/demo/.fast_demo_mode', "[Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-1/test_install_source_copy_exclu0/project/demo/.fast_demo_mode'")]
1 failed, 13 warnings in 0.39s
```

After restoring `.fast_demo_mode` in `tests/test_cli_installed.py`
(`git checkout -- tests/test_cli_installed.py`) and removing the temporary
worktree:

```bash
python -m pytest -q tests/test_cli_installed.py::test_install_source_copy_excludes_disappearing_demo_marker
```

```text
.                                                                        [100%]
1 passed, 13 warnings in 0.22s
```

### 2.3 Supported local suite invocation and successful execution receipt

Suite invocation determination across `pyproject.toml`, `scripts/run_tests.sh`,
and `.github/workflows/ci.yml`:

1. `.github/workflows/ci.yml` delegates to
   `stranske/Workflows/.github/workflows/reusable-10-ci-python.yml@main`
   (resolved in run `42518199588` to
   `1a7a3c7de40b62af2839cb138e8d2abfe72fc766`), verified with:

   ```bash
   gh api repos/stranske/Trend_Model_Project/actions/runs/42518199588 \
     --jq '{head_sha,run_attempt,referenced_workflows}'
   gh api 'repos/stranske/Workflows/contents/.github/workflows/reusable-10-ci-python.yml?ref=1a7a3c7de40b62af2839cb138e8d2abfe72fc766' \
     --jq .content | base64 --decode
   ```

   Its `Pytest (unit tests with coverage)` step prepares a temporary
   `ci_test_growth` plugin and invokes `pytest -p ci_test_growth "${args[@]}"`
   against `pyproject.toml` (`requires-python = ">=3.13"`, `[tool.coverage.run]`
   `source = ["src", "streamlit_app", "design-system", "analysis"]`,
   `[tool.coverage.report]` `fail_under = 80`):

   ```bash
   pytest -p ci_test_growth --junitxml=pytest-junit.xml --cov \
     --cov-config=pyproject.toml --cov-report=xml:coverage.xml \
     --cov-report=term-missing --cov-report=json:coverage.json \
     -n auto --dist=loadgroup
   ```

2. `scripts/run_tests.sh` is the repository's supported local test suite entry
   point. It activates `.venv` (or `VIRTUAL_ENV`), sets `PYTHONHASHSEED=0` and
   `TREND_MODEL_SITE_CUSTOMIZE=1`, installs `requirements.lock` and `-e ".[dev]"`,
   validates `.coveragerc.${PROFILE}` (default `PROFILE=core`, `fail_under = 85`),
   and runs:

   ```bash
   PYTHONPATH="./src" pytest --maxfail=1 --disable-warnings --cov=src --cov-branch --cov-config=.coveragerc.core -n auto
   coverage report -m
   ```

Under the 2026-11-08 local closer dispatch, running the full suite on the local
workstation was prohibited and only `/opt/anaconda3/bin/python3` (`Python 3.12.2`,
`2 passed, 2 deselected, 2 warnings in 0.56s` for `-k install_source_copy`) was
available. Under the 2026-11-12 owner-authorized hosted run for #6208,
`./scripts/run_tests.sh` was executed in an isolated worktree (`/tmp/issue-6205-suite`,
detached at `0b0348f8fea59683199d5a08a4f2d4ade1faf680`, identical product and
test tree to `main` `6f4fd58e99633aaaa480815966d678926fcef7c9` /
`4038b50a5013663b9547d0602b99839929e179d1`) with Python 3.14.10:

```bash
./scripts/run_tests.sh
```

```text
Running tests in parallel with pytest-xdist...
============================= test session starts ==============================
platform linux -- Python 3.14.10, pytest-9.2.0, pluggy-1.7.0
rootdir: /tmp/issue-6205-suite
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
plugins: cov-7.1.0, regressions-2.11.0, platformdirs-4.14.5, datadir-1.9.0, langsmith-0.14.17, hypothesis-6.155.7, rerunfailures-16.3, xdist-3.8.0, anyio-4.16.0, typeguard-4.6.0
4 workers [7074 items]
...
TOTAL                                                    24294   1674   9612   1208    91%
Required test coverage of 85.0% reached. Total coverage: 90.75%
=========================== short test summary info ============================
========= 7064 passed, 11 skipped, 4144 warnings in 186.71s (0:03:06) ==========
```

The command completed with exit code `0` (`7064 passed, 11 skipped`, core
coverage `90.75%` >= `85.0%` threshold), and the temporary worktree was removed.

## 3. Next controller-owned sync-core disposition: recorded

Read existing receipts only; no controller or sync-core round was launched.
The next recorded round after the successful CI job was
`20261108T202925Z`, timestamp 20:29:35Z, with `apply: true`.
The target ledger binds that round to the same #6204 head. Retrieval:

```bash
cat /Users/teacher/.codex/orchestrator/sync-core/targets/stranske_trend_model_project_6204.json
/opt/anaconda3/bin/python3 - <<'PY'
import json
from pathlib import Path
path = Path('/Users/teacher/.codex/orchestrator/sync-core/rounds.jsonl')
for raw in reversed(path.read_text().splitlines()):
    row = json.loads(raw)
    if row['round'] == '20261108T202925Z':
        print(json.dumps({
            'round': row['round'], 'ts': row['ts'], 'apply': row['apply'],
            'items': [item for item in row['items']
                      if item.get('target') == 'stranske/Trend_Model_Project#6204'],
        }, indent=2))
        break
PY
```

Relevant receipt fields:

```json
{
  "round": "20261108T202925Z",
  "ts": "2026-11-08T20:29:35Z",
  "apply": true,
  "head": "cde9f21439ba2f72ca596f54cd017c7673c6221b",
  "target": "stranske/Trend_Model_Project#6204",
  "disposition": "ready-for-maint71",
  "owner": "maint-71",
  "error": null,
  "handoffs": [],
  "reasons": [
    "maint71: campaign_prepared; next=merge-current-delivery; refresh_reason=None"
  ]
}
```

This combines the target ledger's head binding and the round's target entry;
it is not PR-body sync metadata or a newly issued finding-classifier verdict.
The round reports no copy-race finding or handoff. Maint68 regeneration and
Maint71 merge ownership remain intact; readiness does not establish merge.

## Acceptance criteria

- [x] `python -m pytest -q tests/test_cli_installed.py -k install_source_copy` passes on the recorded revision under Python >=3.13 (`Python 3.14.10`, `2 passed, 2 deselected`).
- [x] `python -m pytest -q tests/test_cli_installed.py::test_install_source_copy_excludes_disappearing_demo_marker` fails with the `.fast_demo_mode` exclusion removed in an isolated checkout (`shutil.Error`) and passes after restoration (`1 passed`), using Python >=3.13 (`Python 3.14.10`).
- [x] `docs/acceptance/issue-6205.md` records the repository's actual local suite command (`./scripts/run_tests.sh` and the `pyproject.toml` / `.github/workflows/ci.yml` pytest invocation) and its successful result (`7064 passed, 11 skipped`, `90.75%` core coverage, exit 0) on the identified checkout (`0b0348f8fea59683199d5a08a4f2d4ade1faf680` / `main` `6f4fd58e99633aaaa480815966d678926fcef7c9`).
- [x] `gh run view 42518199588 --repo stranske/Trend_Model_Project --json headSha,attempt,conclusion,jobs` confirms the recorded exact-head Python 3.14 success (`headSha: cde9f21439ba2f72ca596f54cd017c7673c6221b`, `attempt: 2`, `conclusion: success`, job `128848973787`).
- [x] Record the next sync-core classification (`20261108T202925Z`, `ready-for-maint71`) without the copy-race finding while retaining existing review and writer gates.

Refs #6205
Refs #6208
<!-- workflow-source:local_request -->
