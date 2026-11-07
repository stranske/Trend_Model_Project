# Issue #6188 closer acceptance

Review date: 2026-11-07. The merged compatibility fix is PR #6192,
`a2e82d2961823393f6e73f4db7dc672a3d3cc607`. This follow-up addresses review
findings on PR #6193; it does not establish hosted acceptance on a regenerated
sync head.

## Required Evidence check

The requested result is **Dev-tool pin update breaks Python CI / python 3.13
(sync PR 6187)**. Its concrete execution path is
`.github/workflows/pr-00-gate.yml` and `tests/test_xdist_loadgroup_compat.py`.
`Gate / gate` alone cannot satisfy this requirement. No exact-head Evidence
receipt for the requested check has been verified in this follow-up.

## Acceptance criteria

The direct pytest commands in these checkboxes require a selected **Python
3.13** environment with **pytest 9.2.0** and **pytest-xdist 3.8.0**. Install the
project extras first (`python -m pip install -e ".[dev,app]"`), then install
`python -m pip install pytest==9.2.0 pytest-xdist==3.8.0` in that environment.

- [ ] The selected Python 3.13 interpreter passes the compatibility suite:
  `python -m pytest tests/test_xdist_loadgroup_compat.py -q`.
- [ ] The same interpreter passes the golden demo using loadgroup scheduling:
  `python -m pytest tests/golden/test_demo.py -n 2 --dist loadgroup -v --tb short`.
- [ ] After observing the deliberate failure described below, remove the disabling
  variable and pass the same regression test:
  `python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q`.
- [ ] The documented repository command, `./scripts/run_tests.sh`, exits 0 with
  its normal dependency installation and coverage requirements. This is a
  separate run: the script installs `requirements.lock` and `.[dev]`, which both
  pin **pytest 9.1.1**. Record its actual versions separately from the direct
  pytest 9.2.0 runs above.
- [ ] An exact-head Evidence receipt shows the requested check passing on the
  regenerated sync head. Record the Evidence run ID, job name, head SHA, and
  `gh run view <evidence-run-id> --repo stranske/Trend_Model_Project` output.
- [ ] A subsequent controller-owned sync-core receipt identifies #6187 or its
  explicitly mapped successor at the tested head and classifies it without the
  original finding. Record any successor PR and its relationship to #6187.
- [ ] The final delivery has no active non-outdated review threads.

## Deliberate failure probe (outside executable acceptance gates)

In the selected Python 3.13 / pytest 9.2.0 / pytest-xdist 3.8.0 environment, run:

```bash
TREND_DISABLE_XDIST_LOADGROUP_COMPAT=1 python -m pytest tests/test_xdist_loadgroup_compat.py::test_loadgroup_keeps_xdist_group_on_one_worker -q
```

The expected outcome is a nonzero exit with `xdist_group split across workers`.
The passing restoration command belongs to the acceptance checkbox above.
Do not export the disabling variable into the passing gates.

## PR #6193 review dispositions

- [x] Inspect the two unresolved PR #6193 review threads through the GitHub
  review-thread API, address their substantive findings, and record each
  disposition. The extraction checks below pass; remote thread closure remains
  a separate acceptance requirement.

Observed through the GitHub review-thread API on 2026-11-07. The API returned
two unresolved threads, one of which is already outdated.

| Thread / review comment | API state at inspection | Finding and disposition |
| --- | --- | --- |
| `PRRT_kwDOO0LrSc60jyDb` / [4493915276](https://github.com/stranske/Trend_Model_Project/pull/6193#discussion_r4493915276) | Unresolved, outdated | The original command section was outside the extractor's acceptance scope; the inspected PR head already moved commands into a checkbox. This follow-up retains discoverable passing commands in separate checkboxes and puts the intentional failure under a separate heading. Regression tests exercise the actual Evidence extractor against this file. |
| `PRRT_kwDOO0LrSc60j71o` / [4493979584](https://github.com/stranske/Trend_Model_Project/pull/6193#discussion_r4493979584) | Unresolved, current | Valid: the script installs pytest 9.1.1 through both the lockfile and the dev extra. Its run is now described separately; the direct pytest gates require 9.2.0. No script result is claimed as pytest 9.2 parity. |

These are implementation dispositions, not claims that the remote threads have
been resolved. Thread closure still requires verification on the final delivery.

Verification in this follow-up used the available Python 3.14.10 / pytest 9.1.1
environment, not the acceptance matrix:

```text
$ python -m pytest tests/scripts/test_issue_6188_acceptance.py tests/test_script_error_handling.py -q -m 'not slow' -k 'evidence_extracts or deliberate_failure or run_tests_installs_declared_dev_dependencies_with_active_python'
3 passed, 6 deselected in 0.87s
$ python scripts/runner_lib/acceptance_gate.py --extract < docs/evidence/issue-6188-closer-acceptance.md
```

The extractor returned exactly three passing pytest gates, attached to criteria
0, 1 and 2. The deliberate failure was excluded. This verifies command discovery,
not execution of the acceptance gates themselves.
Ruff and Black passed for the added Python test file. The required full-tree
Black check failed because the sandbox prevents multiprocessing sockets; a
single-worker retry was interrupted without a result. Full-tree formatting is
therefore unverified, and no commit was made.

## Historical results and remaining unknowns

The reviewed document at PR #6193 commit
`bbfcd7fab76dab3038786d25fbe99f44aca54b2d` reported the following results.
They are historical reports, not fresh command receipts for this branch:

- Direct runs reportedly used Python 3.13.13, pytest 9.2.0 and pytest-xdist
  3.8.0: the compatibility suite passed seven tests, the demo passed four,
  the disabled-shim probe failed, and the restored probe passed.
- The script invocation selected only the compatibility and demo files, used
  `-n 0`, and supplied `TREND_SKIP_COVERAGE_REPORT=1`. It reported nine passes
  and two skips specific to pytest below 9.2. This is not evidence that the
  documented full repository command passed its normal acceptance requirements.
  The current script has no implementation of that skip variable.
- [Gate run 41956109404](https://github.com/stranske/Trend_Model_Project/actions/runs/41956109404)
  reportedly failed both `Gate / gate` and `Python CI / python 3.13` on #6187.
  A failed check cannot complete hosted acceptance. The tested head and an
  exact requested-check Evidence receipt remain unverified here.
- [Health 68 run 42234189419](https://github.com/stranske/Workflows/actions/runs/42234189419)
  reportedly completed its aggregate drift comparison and skipped alert jobs.
  That does not identify #6187 or a successor at the tested head, and cannot
  establish target-specific sync-core classification.

Controller-owned regeneration, a verified successor mapping if necessary,
exact-head Evidence, and subsequent target-specific sync-core classification
remain outstanding. No workflow files are changed by this review follow-up.

Delivery blocker: this runner mounts `.git` read-only. Even staging with
`git add --dry-run` fails to create `.git/index.lock` with `Read-only file system`.
The review fixes and regression tests are present in the workspace, but cannot
be committed in this environment. A writable checkout is required to deliver
them before continuing hosted acceptance.
Attempts to add the required `needs-human` label and blocker comment to #6194
were also rejected: the GitHub connector requires approval, and this run's
approval policy is `never`. The blocker is therefore recorded here locally.
