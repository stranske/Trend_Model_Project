<!-- bootstrap for codex on issue #2962 -->

## 2025-10-24: Gate Workflow Validation Fixes

**Problem**: PR #2985 was unable to complete checks because Gate workflow was failing immediately with "workflow file issue" error before any jobs could run.

**Root Causes Identified**:
1. **Invalid workflow_call outputs from matrix job**: The reusable CI workflow was trying to export outputs from a matrix job (`jobs.tests.outputs.*`), which is not supported by GitHub Actions since matrix jobs produce multiple instances.
   
2. **Exceeded workflow_dispatch input limit**: GitHub enforces a maximum of 10 inputs for `workflow_dispatch` events, but the workflow provided 13 inputs, causing validation failure.

**Fixes Applied**:
- Removed invalid `workflow_call` outputs section from `reusable-10-ci-python.yml`
- Removed redundant workflow_dispatch inputs to meet 10-input limit:
  - `python-version` (redundant with `python-versions`)
  - `run-mypy` (deprecated alias for `typecheck`)
  - `enable-coverage-delta` (specialized use case)
- All inputs remain available in `workflow_call` for programmatic consumers

**Outcome**: Gate workflow now runs successfully on PR #2985. The CI pipeline can properly evaluate the reusable workflow canonicalization changes.

**Commits**:
- `15ca9c6e`: Remove invalid workflow_call outputs from matrix job
- `c4176498`: Reduce workflow_dispatch inputs to GitHub's 10-input limit

