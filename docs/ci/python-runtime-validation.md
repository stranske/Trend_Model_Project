# Python runtime migration validation

Validation for #6125 / PR #6158 in the 2026-10-26 automation checkout:

- [ ] Run the runtime guard and both WASM test modules with the repository-selected
  `.venv/bin/python` (Python 3.12.2). This checkout has no `.venv`; installed
  interpreters include Python 3.12.15 and 3.14.8, but not 3.12.2.
- [x] Verify deliberate 3.12 matrix regression and exact-byte restoration. The
  regression test copies every workflow YAML into a temporary directory, changes
  only the caller matrix pin, and checks that the named guard rejects
  `ci.yml.jobs.python.with.python-versions` (and the corresponding Gate input).
  For each caller, a child pytest process must exit with `TESTS_FAILED`, report
  exactly one failed `test_workflow_python_pins_follow_standard`, and include
  the expected runtime-input path. Restoration runs in `finally`, followed by
  an exact-byte comparison of all copied workflows and another child pytest
  process that must exit with `OK` and one passing guard test. Repository
  workflows are never edited by this test.
- [ ] Complete Ruff, Black, actionlint, and whitespace validation using the
  selected interpreter and existing isolated formatter dependencies. Ruff and
  `git diff --check` pass. The edited test passes Black with target `py312` and
  line length 100. The required repository-wide Black command, run with
  `BLACK_NUM_WORKERS=1` and a 180-second timeout, exits with status 124. Its
  interrupted report lists 1,064 unchanged files; the nonzero exit does not
  establish a passing gate. Actionlint is absent. Existing Black 26.10.1 and
  Ruff 0.17.2 were used with Python 3.14.8. No shared formatter dependencies or
  environments were changed; validation with the selected interpreter remains
  blocked.

The following command passes **103 tests** on the available Python **3.14.8**:

```sh
python -m pytest tests/test_ci_python_standard.py tests/wasm/test_build_wasm_site.py tests/wasm/test_demo_no_external_cdn.py -m 'not slow' -q
```

This includes both parametrized cases of
`test_json_matrix_deliberate_break_and_restore`. The latest focused addition
also verifies that the package still supports Python 3.12, advertises that
classifier, and retains the mypy Python 3.12 target as CI moves to newer runtimes.
The full command passes in 7.10 seconds. This result does not establish the
requested Python 3.12.2 validation. The count increased from the earlier 94
because subsequent commits added guard tests.

The recent commits were reconciled against all three tasks: only the deliberate
regression/restoration task is verified. Updating the PR body's checkboxes was
attempted through the GitHub connector, but the tool requires approval and this
run has approval policy `never`; the PR body was not changed. The blocker comment
and `needs-human` label were also attempted and rejected for the same reason.

Committing remains blocked: the sandbox mounts `.git` read-only, and this run
confirms `test -w .git` fails. The required repository-wide Black gate has also
not passed, so the Python changes must not be committed or pushed yet.
