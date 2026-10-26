# Python runtime migration validation

Validation for #6125 / PR #6158 in the 2026-10-26 automation checkout:

- [ ] Run the runtime guard and both WASM test modules with the repository-selected
  `.venv/bin/python` (Python 3.12.2). This checkout has no `.venv`; installed
  interpreters include Python 3.12.15 and 3.14.8, but not 3.12.2.
- [x] Verify deliberate 3.12 matrix regression and exact-byte restoration. The
  regression test copies every workflow YAML into a temporary directory, changes
  only the caller matrix pin, and checks that the named guard rejects
  `ci.yml.jobs.python.with.python-versions` (and the corresponding Gate input).
  Restoration runs in `finally`, followed by an exact-byte comparison of all
  copied workflows and another successful guard run. Repository workflows are
  never edited by this test.
- [ ] Complete Ruff, Black, actionlint, and whitespace validation using the
  selected interpreter and existing isolated formatter dependencies. Ruff and
  `git diff --check` pass. The edited test passes Black with target `py312` and
  line length 100. The required repository-wide Black command fails in its
  default multiprocessing mode because socket creation is denied; single-worker
  retries did not finish and were stopped. Actionlint is absent, and an isolated
  installation attempt was blocked by sandbox network restrictions.

The following command passes **94 tests** on the available Python **3.14.8**:

```sh
python -m pytest tests/test_ci_python_standard.py tests/wasm/test_build_wasm_site.py tests/wasm/test_demo_no_external_cdn.py -m 'not slow' -q
```

The two parametrized cases of
`test_json_matrix_deliberate_break_and_restore` also pass independently.
This result does not establish the requested Python 3.12.2 validation.

Committing is blocked: `git add tests/test_ci_python_standard.py` fails because
the sandbox mounts `.git` read-only and cannot create `.git/index.lock`.
