# Workflow System Overview

**Purpose.** Document what runs where, why each workflow exists, and how the
pieces interlock so contributors can land changes without tripping the
guardrails. Automation shows up in four canonical buckets that mirror what a
contributor experiences on a pull request or on the maintenance calendar:

1. **PR checks** – gatekeeping for every pull request (Gate, PR 02 Autofix).
2. **Maintenance & repo health** – scheduled and follow-up automation that keeps
   the repository clean (Maint 46 Post CI, Maint 45, recurring health checks).
3. **Issue / agents automation** – orchestrated agent work and issue
   synchronisation (Agents 70 orchestrator plus Agents 63/64 companions).
4. **Error checking, linting, and testing topology** – reusable workflows that
   fan out lint, type, test, and container verification across the matrix.

Each bucket below calls out the canonical workflows, the YAML entry point, and
the policy guardrails that keep the surface safe. Keep this mental map handy:

```
PR checks ──► Reusable CI matrix
    │              │
    │              └──► Error checking, linting, testing topology
    ▼
Maintenance & repo health ──► Issue / agents automation
```

Gate opens the door, reusable CI fans out the heavy lifting, maintenance keeps
the surface polished, and the agents stack orchestrates follow-up work.

### Quick orientation for new contributors

When you first land on the project:

1. **Skim the bucket summaries below** to understand which workflows will react
   to your pull request or scheduled automation.
2. **Use the workflow summary table** as the canonical source for triggers,
   required status, and log links when you need to confirm behaviour or share a
   run with reviewers.
3. **Review [How to change a workflow safely](#how-to-change-a-workflow-safely)**
   before editing any YAML. It enumerates the guardrails, labels, and approval
   steps you must follow.
4. **Cross-reference the [Workflow Catalog](WORKFLOWS.md)** for deeper YAML
   specifics (inputs, permissions, job layout) once you know which surface you
   are touching.

## Buckets and canonical workflows

### PR checks (Gate + Autofix)
- **Gate** – `.github/workflows/pr-00-gate.yml`
  - Required on every pull request. Detects docs-only diffs (Markdown anywhere,
    the entire `docs/` tree, and `assets/`) and skips the heavier Python and
    Docker matrices when nothing executable changed. Gate owns the short skip
    comment (`<!-- gate-docs-only -->`) and publishes the final combined status.
  - Requests `pull-requests: write` and `statuses: write` scopes so the comment
    and status appear with the correct phrasing.
- **PR 02 Autofix** – `.github/workflows/pr-02-autofix.yml`
  - Opt-in via the `autofix` label only. Runs the same formatters and light
    hygiene steps that Gate would otherwise leave to contributors.
  - When enabled, it must cancel duplicates to avoid fighting with Maint 46.

### Maintenance & repo health
- **Maint 46 Post CI** – `.github/workflows/maint-46-post-ci.yml` consolidates
  CI results, uploads artifacts, and applies small, low-risk fixes (for example,
  syncing generated docs or updating the failure tracker).
- **Maint 45 Cosmetic Repair** – `.github/workflows/maint-45-cosmetic-repair.yml`
  is a manual workflow. It runs pytest and the guardrail fixers, then opens a
  labelled PR if changes are needed.
- **Health checks** – recurring workflows that keep the repo honest:
  - `health-40-repo-selfcheck.yml` (daily pulse),
  - `health-41-repo-health.yml` (weekly sweep),
  - `health-42-actionlint.yml` (actionlint enforcement),
  - `health-43-ci-signature-guard.yml` (signature verification),
  - `health-44-gate-branch-protection.yml` (required check enforcement), and
  - `health-45-agents-guard.yml` (immutable agents surface guardrail).

### Issue / agents automation
- **Agents 70 Orchestrator** – `.github/workflows/agents-70-orchestrator.yml`
  remains the single dispatch surface for every consumer workflow. Agents 61/62
  shims stay retired.
- **Agents 63 Codex Issue Bridge** – `.github/workflows/agents-63-codex-issue-bridge.yml`
  turns labelled issues into branches and bootstrap PRs.
- **Agents 63 ChatGPT Issue Sync** – `.github/workflows/agents-63-chatgpt-issue-sync.yml`
  keeps curated topic files (for example `Issues.txt`) aligned with tracked
  issues.
- **Agents 64 Assignment Verifier** – `.github/workflows/agents-64-verify-agent-assignment.yml`
  audits that orchestrated work is assigned correctly and feeds the orchestrator.
- **Guardrail** – The orchestrator and both `agents-63-*` workflows are locked
  down by CODEOWNERS, branch protection, the Agents Critical Guard check, and a
  repository ruleset. See [Agents Workflow Protection Policy](./AGENTS_POLICY.md)
  for the change allowlist and override procedure.

### Error checking, linting, and testing topology
- **Reusable Python CI** – `reusable-10-ci-python.yml` fans out ruff, mypy, and
  pytest across the interpreter matrix. It reads `python_version = "3.11"` from
  `pyproject.toml` and pins the mypy leg accordingly.
- **Reusable Docker CI** – `reusable-12-ci-docker.yml` builds the container
  image and exercises the smoke tests Gate otherwise short-circuits for
  docs-only changes.
- **Reusable Agents** – `reusable-16-agents.yml` powers orchestrated dispatch.
- **Reusable Autofix** – `reusable-18-autofix.yml` centralizes fixers for PR 02
  Autofix and Maint 46.
- **Self-test Runner** – `selftest-runner.yml` is the consolidated manual entry
  point. Inputs:
  - `mode`: `summary`, `comment`, or `dual-runtime` (controls reporting surface
    and Python matrix).
  - `post_to`: `pr-number` or `none` (comment target when `mode == comment`).
  - `enable_history`: `true` or `false` (download the verification artifact for
    local inspection).
  - Optional niceties include `pull_request_number`,
    `summary_title`/`comment_title`, `reason`, and `python_versions` (JSON array
    to override the default matrix).

## Workflow summary table

| Bucket | Workflow | Trigger | Purpose | Required? | Artifacts / logs |
| --- | --- | --- | --- | --- | --- |
| PR checks | Gate (`pr-00-gate.yml`) | `pull_request`, `pull_request_target` | Detect docs-only diffs, orchestrate CI fan-out, and publish the combined status. | ✅ Always | [Gate workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml) |
| PR checks | PR 02 Autofix (`pr-02-autofix.yml`) | `pull_request` (label gated) | Run optional fixers when the `autofix` label is present. | ⚪ Optional | [Autofix runs & artifacts](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-02-autofix.yml) |
| Maintenance & repo health | Maint 46 Post CI (`maint-46-post-ci.yml`) | `workflow_run` (Gate success) | Consolidate CI output, apply small hygiene fixes, and update failure-tracker state. | ⚪ Optional (auto) | [Maint 46 run log](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-46-post-ci.yml) |
| Maintenance & repo health | Maint 45 Cosmetic Repair (`maint-45-cosmetic-repair.yml`) | `workflow_dispatch` | Run pytest + fixers manually and open a labelled PR when changes are required. | ⚪ Manual | [Maint 45 manual entry](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-45-cosmetic-repair.yml) |
| Maintenance & repo health | Health 40 Repo Selfcheck (`health-40-repo-selfcheck.yml`) | `schedule` (daily) | Capture repository pulse metrics. | ⚪ Scheduled | [Health 40 summary](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-40-repo-selfcheck.yml) |
| Maintenance & repo health | Health 41 Repo Health (`health-41-repo-health.yml`) | `schedule` (weekly) | Perform weekly dependency and repo hygiene sweep. | ⚪ Scheduled | [Health 41 dashboard](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-41-repo-health.yml) |
| Maintenance & repo health | Health 42 Actionlint (`health-42-actionlint.yml`) | `schedule` (daily) | Enforce actionlint across workflows. | ⚪ Scheduled | [Health 42 logs](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-42-actionlint.yml) |
| Maintenance & repo health | Health 43 CI Signature Guard (`health-43-ci-signature-guard.yml`) | `schedule` (daily) | Verify reusable workflow signature pins. | ⚪ Scheduled | [Health 43 verification](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-43-ci-signature-guard.yml) |
| Maintenance & repo health | Health 44 Gate Branch Protection (`health-44-gate-branch-protection.yml`) | `schedule`, `workflow_dispatch` | Ensure Gate and Agents Guard stay required on the default branch. | ⚪ Scheduled (fails if misconfigured) | [Health 44 enforcement logs](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-44-gate-branch-protection.yml) |
| Maintenance & repo health | Health 45 Agents Guard (`health-45-agents-guard.yml`) | `pull_request`, `workflow_dispatch`, `schedule` | Block unauthorized changes to protected agents workflows. | ✅ Required when `agents-*.yml` changes | [Agents Guard run history](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-45-agents-guard.yml) |
| Issue / agents automation | Agents 70 Orchestrator (`agents-70-orchestrator.yml`) | `workflow_call`, `workflow_dispatch` | Fan out consumer automation and dispatch work. | ✅ Critical surface | [Orchestrator runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-70-orchestrator.yml) |
| Issue / agents automation | Agents 63 Codex Issue Bridge (`agents-63-codex-issue-bridge.yml`) | `issues`, `workflow_dispatch` | Bootstrap branches and PRs from labelled issues. | ✅ Critical surface | [Agents 63 bridge logs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-63-codex-issue-bridge.yml) |
| Issue / agents automation | Agents 63 ChatGPT Issue Sync (`agents-63-chatgpt-issue-sync.yml`) | `workflow_dispatch` | Keep curated topic files in sync with issues. | ✅ Critical surface | [Agents 63 sync runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-63-chatgpt-issue-sync.yml) |
| Issue / agents automation | Agents 64 Verify Agent Assignment (`agents-64-verify-agent-assignment.yml`) | `schedule`, `workflow_dispatch` | Audit orchestrated assignments and alert on drift. | ⚪ Scheduled | [Agents 64 audit history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-64-verify-agent-assignment.yml) |
| Issue / agents automation | Agents Critical Guard (`agents-critical-guard.yml`) | `pull_request` | Block deletion or renaming of protected agents workflows before maintainer review. | ✅ Required when PR touches protected files | [Agents Critical Guard runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-critical-guard.yml) |
| Error checking, linting, and testing topology | Reusable Python CI (`reusable-10-ci-python.yml`) | `workflow_call` | Provide shared lint/type/test matrix for Gate and manual callers. | ✅ When invoked | [Reusable Python CI runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-10-ci-python.yml) |
| Error checking, linting, and testing topology | Reusable Docker CI (`reusable-12-ci-docker.yml`) | `workflow_call` | Build and smoke-test container images. | ✅ When invoked | [Reusable Docker runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-12-ci-docker.yml) |
| Error checking, linting, and testing topology | Reusable Agents (`reusable-16-agents.yml`) | `workflow_call` | Power orchestrated dispatch. | ✅ When invoked | [Reusable Agents history](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-16-agents.yml) |
| Error checking, linting, and testing topology | Reusable Autofix (`reusable-18-autofix.yml`) | `workflow_call` | Centralise formatter + fixer execution. | ✅ When invoked | [Reusable Autofix runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-18-autofix.yml) |
| Error checking, linting, and testing topology | Self-test Runner (`selftest-runner.yml`) | `workflow_dispatch` | Run manual verification matrix across interpreters. | ⚪ Manual | [Self-test runner history](https://github.com/stranske/Trend_Model_Project/actions/workflows/selftest-runner.yml) |

## Policy

- **Required checks.** Gate is mandatory on every PR. Health 45 Agents Guard
  becomes required whenever a change touches the `agents-*.yml` surface. Both
  checks must appear in branch protection.
- **Docs-only detection.** Lives exclusively inside Gate—there is no separate
  docs-only workflow.
- **Autofix.** Maint 46 centralizes automated follow-up fixes. Forks upload
  patch artifacts instead of pushing. Pre-CI autofix (`pr-02-autofix.yml`) must
  stay label-gated and cancel duplicates while Gate runs.
- **Branch protection.** The default branch must require Gate and Health 45. The
  Health 44 workflow resolves the current default branch via the REST API and
  either enforces or verifies the rule (requires a `BRANCH_PROTECTION_TOKEN`
  secret with admin scope for enforcement).
- **Code Owner reviews.** Enable **Require review from Code Owners** so changes
  to `agents-63-chatgpt-issue-sync.yml`, `agents-63-codex-issue-bridge.yml`, and
  `agents-70-orchestrator.yml` stay maintainer gated on top of the immutable
  guardrails.
- **Types.** When mypy is pinned, run it in the pinned interpreter only to avoid
  stdlib stub drift. `reusable-10-ci-python.yml` reads the desired version from
  `pyproject.toml` and guards the mypy step with
  `matrix.python-version == steps.mypy-pin.outputs.python-version`. Ruff and
  pytest still execute across the full matrix.
- **Automation labels.** Keep the labels used by automation available:
  `workflows`, `ci`, `devops`, `docs`, `refactor`, `enhancement`, `autofix`,
  `priority: high|medium|low`, `risk:low`, `status: ready|in-progress`,
  `agents`, and `agent:codex`.

## Final topology (keep vs retire)

- **Keep.** `pr-00-gate.yml`, `pr-02-autofix.yml`, `maint-45-cosmetic-repair.yml`,
  `maint-46-post-ci.yml`, health 40/41/42/43/44/45, agents 70/63, `agents-critical-guard.yml`,
  reusable 10/12/16/18, and `selftest-runner.yml`.
- **Retire.** `pr-14-docs-only.yml`, `maint-47-check-failure-tracker.yml`, the
  removed Agents 61/62 consumer workflows, and the legacy `selftest-*` wrappers
  superseded by `selftest-runner.yml`.

## How to change a workflow safely

1. Start with the [Agents Workflow Protection Policy](./AGENTS_POLICY.md) to
   confirm the change fits the allowlist and review the guardrails.
2. File or link the incident/maintenance issue describing the need for the
   change. Capture the risk assessment and expected blast radius in that issue.
3. Secure the required `agents:allow-change` label (maintainers only) before
   pushing edits to protected workflows. Gate or the orchestrator will block the
   PR without it.
4. Keep Code Owner review enabled so the protected files land only with explicit
   maintainer approval. At least one owning maintainer must approve before
   merging.
5. After merge, remove the label, confirm Maint 46 processed the follow-up
   hygiene, and verify both Agents Critical Guard and Health 45 report green.

## Verification checklist

- Gate badge in `README.md` and branch protection both show as required for the default branch.
- Health 45 Agents Guard appears as a required check whenever protected workflows change and reports ✅ in the latest run.
- Maintainers can point to the most recent [Workflow System Overview](../ci/WORKFLOW_SYSTEM.md) update in pull-request history, demonstrating that contributors can discover the guardrails without escalation.
- Gate runs and passes on docs-only PRs and appears as a required check.
- Health 45 blocks unauthorized agents workflow edits and reports as the required check whenever `agents-*.yml` files change.
- Health 44 confirms branch protection requires Gate and Agents Guard on the default branch.
- Maint 46 posts a single consolidated summary; autofix artifacts or commits are attached where allowed.

## Branch protection playbook

1. **Confirm the default branch.**
   - Health 44 resolves the branch name automatically through `repos.get`. No
     manual input is required for scheduled runs.
   - For ad-hoc verification, run `gh api repos/<owner>/<repo> --jq .default_branch`
     or read the repository settings (currently `phase-2-dev`).
2. **Verify enforcement credentials.**
   - Create a fine-grained personal access token with
     `Administration: Read and write` on the repository.
   - Store it as the `BRANCH_PROTECTION_TOKEN` secret. With the token present,
     Health 44 applies the branch protection before verifying. Without it the
     workflow performs a read-only check, uploads an observer-mode summary, and
     still fails if Gate is not required.
3. **Run the enforcement script locally when needed.**
   - `python tools/enforce_gate_branch_protection.py --repo <owner>/<repo> --branch <default-branch> --check`
     reports the current status.
   - Add `--require-strict` to fail if the workflow token cannot confirm
     “Require branches to be up to date” (needs admin scope).
   - Add `--apply` to enforce the rule locally (requires admin token in
     `GITHUB_TOKEN`/`GH_TOKEN`). Use `--snapshot path.json` to capture
     before/after state for change control.
4. **Audit the result.**
   - Health 44 uploads JSON snapshots (`enforcement.json`, `verification.json`)
     mirroring the script output and writes a step summary when it runs in
     observer mode.
   - In GitHub settings, confirm that **Gate** and **Health 45 Agents Guard**
     appear under required status checks and that “Require branches to be up to
     date before merging” is enabled.
5. **Trigger Health 44 on demand.**
   - Kick a manual run with `gh workflow run "Health 44 Gate Branch Protection" --ref <default-branch>`
     whenever you change branch-protection settings.
   - Scheduled executions run daily at 06:00 UTC; a manual dispatch confirms the
     fix immediately after you apply it.
6. **Verify with a test PR.**
   - Open a throwaway PR against the default branch and confirm that the Checks
     tab shows **Gate / gate** under “Required checks.” When you modify
     `agents-*.yml`, also confirm **Health 45 Agents Guard / Enforce agents
     workflow protections** is listed as required.
   - Close the PR after verification to avoid polluting history.

### Recovery scenarios

- **Health 44 fails because a required check is missing.**
  1. Confirm you have access to an admin-scoped token (see step 2 above) and
     re-run the workflow with the token configured.
  2. If the failure persists, run `python tools/enforce_gate_branch_protection.py --check`
     locally to inspect the status and `--apply` to restore both required
     contexts.
  3. Re-dispatch Health 44 to record the remediation snapshots and attach them to
     the incident report.
- **Required check accidentally removed during testing.**
  1. Restore the branch-protection snapshot from the most recent successful
     Health 44 run (download from the workflow artifact, then feed into
     `--apply --snapshot` to replay).
  2. Notify the on-call in `#trend-ci` so they can watch the next scheduled job
     for regressions.
  3. Open a short-lived PR targeting the default branch to confirm that Gate and
     Agents Guard return as required before declaring recovery complete.
