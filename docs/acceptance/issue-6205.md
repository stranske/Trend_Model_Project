# Acceptance record: stranske/Trend_Model_Project#6205

<!-- closer-core/v1 repair-of=stranske/Trend_Model_Project#6205@6206 -->

## Check date, outcome and stop rule

Checked 2026-11-08 UTC; continuation began at 20:36:03Z. Expected outcome:
the regenerated #6204 head has successful Python 3.14 CI, the actual reusable
workflow suite command and local evidence are recorded accurately, and a later
controller-owned sync-core round no longer reports the copy-race finding.

Stop after three acceptance attempts or at 21:06:03Z, whichever comes first;
report the first failing or unproven step. One acceptance attempt was needed.
The dispatch explicitly prohibits a local full-suite or release/verify run.
That overrides the earlier request to run the reusable workflow's full suite
locally: the local full-suite result remains **unproven**.

Commands and observed outputs follow. The existing repair branch was retained:
`git branch --list 'closer*/issue-6205'` returned only `closer-r1/issue-6205`.
After fetching, its only commit beyond current `origin/main` was the existing
acceptance record, `cd04225e449c38f0f6478afebbd240894ca7fd79`.
Main was `f77623978ffafaeaa5f40b486717063bd1bd1a36` (#6206).
No implementation or dependency pins changed in this continuation.

## 1. Exact-head Python 3.14 CI: passed

```bash
gh pr view 6204 --repo stranske/Trend_Model_Project --json headRefOid,statusCheckRollup
gh run view 42518199588 --repo stranske/Trend_Model_Project
gh run view 42518199588 --repo stranske/Trend_Model_Project \
  --json headSha,attempt,conclusion,event,jobs
gh run view 42518199588 --repo stranske/Trend_Model_Project \
  --job 128848973787 --log
```

The PR head and run `headSha` both equal
`cde9f21439ba2f72ca596f54cd017c7673c6221b`. Run attempt **2**, triggered by
`pull_request`, completed successfully. Its `Python CI / python 3.14` job
`128848973787` ran from 20:16:27Z to 20:20:34Z with conclusion `success`.

```text
7039 passed, 5 skipped, 4088 warnings in 176.81s (0:02:56)
All checks passed!
```

[Exact-head Python 3.14 job](https://github.com/stranske/Trend_Model_Project/actions/runs/42518199588/job/128848973787).
Attempt 2 supersedes the earlier failed classification/skipped jobs. The former
record's inference from base-commit run `42516821134` was invalid and is removed.
A successful base run does not verify a descendant head.

## 2. Actual suite command and local evidence: partially proven

`.github/workflows/ci.yml` delegates to
`stranske/Workflows/.github/workflows/reusable-10-ci-python.yml@main`.
The exact-head Gate run resolved that reusable workflow to
`1a7a3c7de40b62af2839cb138e8d2abfe72fc766`, verified with:

```bash
gh api repos/stranske/Trend_Model_Project/actions/runs/42518199588 \
  --jq '{head_sha,run_attempt,referenced_workflows}'
gh api 'repos/stranske/Workflows/contents/.github/workflows/reusable-10-ci-python.yml?ref=1a7a3c7de40b62af2839cb138e8d2abfe72fc766' \
  --jq .content | base64 --decode
```

Its `Pytest (unit tests with coverage)` step generates a temporary
`ci_test_growth` plugin and invokes `pytest -p ci_test_growth "${args[@]}"`.
For the Python 3.14 coverage job, `ci.yml` defaults expand to:

```bash
pytest -p ci_test_growth --junitxml=pytest-junit.xml --cov \
  --cov-config=pyproject.toml --cov-report=xml:coverage.xml \
  --cov-report=term-missing --cov-report=json:coverage.json \
  -n auto --dist=loadgroup
```

The plugin and its temporary directory on `PYTHONPATH` are prepared by the
workflow; this command assumes that setup. `scripts/run_tests.sh` is a different
local helper, not the reusable workflow's suite invocation. The successful
PR Gate job above uses the same reusable step with the additional prefix
`-m 'not mc_viz_integration'`; its result is remote CI evidence, not a local
full-suite result or evidence for the additional MC-viz selection in `ci.yml`.

No repository `.venv/bin/python` was present, so the owner-specified fallback
was used for the authorized focused command:

```bash
/opt/anaconda3/bin/python3 --version
/opt/anaconda3/bin/python3 -m pytest -q tests/test_cli_installed.py -k install_source_copy
```

```text
Python 3.12.2
..                                                                       [100%]
2 passed, 2 deselected, 2 warnings in 0.56s
```

Warnings report local `numexpr` 2.8.7 and `bottleneck` 1.3.7 below pandas'
requested versions. This focused helper-test result is not a supported-runtime
application qualification: `pyproject.toml` requires Python >=3.13.
No full suite was run locally. The earlier record incorrectly marked this
criterion passed using only the focused result.

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

## Remaining acceptance

- [x] Record a successful Python CI / python 3.14 run bound to the regenerated #6204 head.
- [ ] Record the repository's actual suite command and successful local result.
  The actual command and successful focused result are recorded; a local
  full-suite result is unproven under the explicit local-run prohibition.
- [x] Record the next sync-core classification without this finding.

The first unproven step is the requested local full-suite result. The primary
fix in #6206 is preserved, and this record does not assert issue completion.

Refs #6205
<!-- workflow-source:local_request -->
