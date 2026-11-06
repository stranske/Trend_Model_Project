# Issue #6188 closer acceptance (follow-up #6194)

Check date: 2026-11-06. Base: `main` at `87ace67e2` (includes merged PR **#6192**, `a2e82d296`).
Observed results are marked **Observed**; anything not verified is under **Unknown / not met**.

## Review threads on PR #6193

Read through the GraphQL `reviewThreads` API (2 threads, both `isResolved: false`).

| Thread | State | Finding | Disposition |
| --- | --- | --- | --- |
| `PRRT_kwDOO0LrSc60jyDb` (Codex, P2) | outdated | Evidence runner extracted `[]` from the acceptance doc. | Valid for the earlier revision. Fixed here: the passing pytest commands sit inside the acceptance checkboxes below, and the intentionally failing break command is kept outside the checkboxes. |
| `PRRT_kwDOO0LrSc60j71o` (CodeRabbit, Minor) | active | Run description called the `run_tests.sh` run pytest-9.2 parity, but the script installs `requirements.lock` (pytest 9.1.1). | Valid. Direct commands below use pytest 9.2.0. The script run uses 9.1.1 and is labelled that way. |

PR #6193's `TREND_SKIP_COVERAGE_REPORT` change to `scripts/run_tests.sh` was not carried over. It would weaken the documented command's coverage requirement, and the task forbids that.

## Hosted check requested: `Python CI / python 3.13`

The requested check is **`Python CI / python 3.13`** (a job of the `Python CI` reusable workflow called from `.github/workflows/pr-00-gate.yml`, job `python-ci`). `Gate / gate` is the aggregate and is not a substitute.

- **Observed:** PR **#6187** (`deps/sync-dev-versions-058fc8abcab9`, "deps: sync dev tool versions") is CLOSED (2026-11-05T05:34:59Z). Its Gate run `41956109404` failed with `Python CI / python 3.13` failing.
- **Observed:** the successor is PR **#6189** (`deps/sync-dev-versions-91791b44e525`, same title, opened 2026-11-05). Head `3083f697e6e9bb675b67fe5dc5b71192935873fb`, Gate run `41964883333`: `Python CI / python 3.13` **FAILURE**, `Python CI / python 3.14` **FAILURE**, `Gate / gate` **FAILURE**.
- That run predates #6192 merging the golden-demo interpreter fix, so it cannot be the passing receipt.

## Acceptance criteria

Direct commands run under **Python 3.13.18** with `pytest==9.2.0` and `pytest-xdist==3.8.0` (install `pip install -e ".[dev,app]"`, then `pip install pytest==9.2.0 pytest-xdist==3.8.0`; the dev extra pins 9.1.1).

- [x] `python -m pytest tests/test_xdist_loadgroup_compat.py -q` exits 0 (7 passed).
- [x] `python -m pytest tests/golden/test_demo.py -n 2 --dist loadgroup -v --tb short` exits 0 (4 passed).
- [x] With the shim restored, `python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q` passes (1 passed).
- [ ] `./scripts/run_tests.sh` (documented command, unmodified) exits 0. **Not met** (see below).
- [ ] `gh run view <id>` shows `Python CI / python 3.13` passing on the regenerated sync head with successor mapping. **Not met** (no regenerated head yet).
- [ ] A sync-core receipt identifies #6187 or #6189 at the tested head and classifies it without the original finding. **Not met.**

Deliberate break (intentionally failing, outside the gates):
`TREND_DISABLE_XDIST_LOADGROUP_COMPAT=1 python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q`

## Local run log (Python 3.13.18, pytest-xdist 3.8.0)

```text
$ python -m pytest --version            -> pytest 9.2.0
$ python -m pytest tests/test_xdist_loadgroup_compat.py -q
7 passed in 32.71s            (exit 0)

$ python -m pytest tests/golden/test_demo.py -n 2 --dist loadgroup -v --tb short
4 passed in 15.08s            (exit 0)

$ TREND_DISABLE_XDIST_LOADGROUP_COMPAT=1 python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q
E   AssertionError: xdist_group split across workers: ['gw0', 'gw1']
1 failed in 1.45s

$ python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q
1 passed in 1.45s
```

## Documented test command (pytest 9.1.1 from `requirements.lock`)

`./scripts/run_tests.sh` with no filters, `VENV_DIR=/tmp/venv-script`, Python 3.13.18:

```text
1 failed, 4637 passed, 6 skipped in 238.43s
FAILED tests/test_cli_installed.py::test_removed_compat_entrypoints_are_not_installed
  ModuleNotFoundError: No module named 'pandas'   (trend CLI resolved to ./src via PYTHONPATH)
FAIL Required test coverage of 85.0% not reached. Total coverage: 76%   (run stopped by --maxfail=1)
```

- The script exports `PYTHONPATH=./src`, and one installed-CLI test did not clear it. `tests/test_cli_installed.py` now pops `PYTHONPATH` there, as its sibling tests already do. With that change the test passes.
- In this runner the nested `system_site_packages` venv cannot see pandas, so two other installed-CLI tests still fail when the script runs inside its own venv. I did not confirm whether this is a runner artifact or a repo defect.
- The 76% coverage figure came from a run that stopped at the first failure, so it is not a measurement of the full suite.

## Unknown / not met

- No passing hosted `Python CI / python 3.13` run on a regenerated sync head. Regeneration of #6189 (or a newer successor) is controller-owned, and I did not push to or alter `deps/*` branches.
- No sync-core receipt after this head. The earlier Health 68 run `42234189419` from PR #6193's notes is not target-specific (it does not name #6187/#6189), so it is not counted as classification.
- The full `./scripts/run_tests.sh` exit code is non-zero until the nested-venv issue and a full uninterrupted run are resolved.

Refs #6188, #6194.
