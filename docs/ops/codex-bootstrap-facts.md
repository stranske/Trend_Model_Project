# Repo Ops Facts — Codex Bootstrap

This page captures the established, validated facts about the Codex bootstrap and workflow wiring in this repository. It is the single source of truth so we don’t keep re‑asking for already‑confirmed details.

Last updated: 2026-02-07

## Branches and Event Semantics
- Default branch: `phase-2-dev`.
- Issue label events execute workflows from the default branch.
- We do not reuse existing PR branches for fixes; each run creates a fresh branch from default.

## Trigger Labels
- Primary label: `agent:codex` (case‑sensitive).
- Alias supported: `agents:codex` (legacy).

## PR Hygiene and Activation
- PRs are created as non‑draft by default.
- The first PR comment includes `@codex start`.
- PRs and the source issue are auto‑assigned to: `chatgpt-codex-connector`, `stranske-automation-bot` (best‑effort; failures are logged as warnings).
- PRs are labeled with `agent:codex`.

## Tokens, Permissions, and Secrets
- Token priority (authoring intent): `OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`.
  - `OWNER_PR_PAT`: When provided, the PR is authored as the human owner and Codex engages on a human‑authored PR.
  - `SERVICE_BOT_PAT`: Service bot author, used when owner PAT is absent.
  - `GITHUB_TOKEN`: Fallback; ensure repository Actions permissions are set to Read/Write.
- The workflow avoids using `secrets.*` inside step‑level `if:` blocks; token decisions are propagated via `env` and inputs.

## Interpreter Bootstrap Invariants
- Repository-root `sitecustomize.py` remains a no-op shim until
  `TREND_MODEL_SITE_CUSTOMIZE=1` is exported; wrapper scripts under `scripts/`
  set this flag so CI/dev invocations still exercise the guarded bootstrap.
- Reviewers should reject contributions that restore eager imports or side
  effects at the repository root.

- ## Workflows and Actions
- Unified orchestrator: `.github/workflows/agents-41-assign-and-watch.yml`.
  - Handles label-triggered assignment, Codex bootstrap, manual watchdog runs, and the scheduled stale sweep.
  - Shares readiness probes via `reusable-90-agents.yml` so availability checks remain single-sourced.
- Wrapper (labels): `.github/workflows/agents-41-assign.yml`.
  - Forwards Issue/PR label events into the unified workflow without duplicating logic.
- Wrapper (manual watchdog): `.github/workflows/agents-42-watchdog.yml`.
  - Preserves the historical manual `workflow_dispatch` interface while delegating to the unified workflow.
- Composite action: `.github/actions/codex-bootstrap-lite/action.yml`.
  - PAT‑first, fallback to `GITHUB_TOKEN` if allowed.
  - Creates a new branch with deterministic prefix `agents/codex-issue-<num>-<runid>`.
  - Seeds a bootstrap file under `agents/` and opens a non‑draft PR.
  - Labels/assigns and posts `@codex start` in PR comments.

## Quick Index

- Agents Assign + Watch — [`agents-41-assign-and-watch.yml`](../../.github/workflows/agents-41-assign-and-watch.yml) · [jump](#wf-assign-and-watch)
- Assign Wrapper — [`agents-41-assign.yml`](../../.github/workflows/agents-41-assign.yml) · [jump](#wf-assign-wrapper)
- Watchdog Wrapper — [`agents-42-watchdog.yml`](../../.github/workflows/agents-42-watchdog.yml) · [jump](#wf-watchdog-wrapper)
- Label Agent PRs — [`label-agent-prs.yml`](../../.github/workflows/label-agent-prs.yml) · [jump](#wf-label-agent-prs)
- Merge Manager — [`merge-manager.yml`](../../.github/workflows/merge-manager.yml) · [jump](#wf-merge-manager)
- Autofix Lane — [`autofix.yml`](../../.github/workflows/autofix.yml) · [jump](#wf-autofix)
- CI — [`pr-10-ci-python.yml`](../../.github/workflows/pr-10-ci-python.yml) · [jump](#wf-ci)
- Docker — [`pr-12-docker-smoke.yml`](../../.github/workflows/pr-12-docker-smoke.yml) · [jump](#wf-docker)
- Cleanup Codex Bootstrap — [`cleanup-codex-bootstrap.yml`](../../.github/workflows/cleanup-codex-bootstrap.yml) · [jump](#wf-cleanup-codex-bootstrap)
- Quarantine TTL — [`maint-34-quarantine-ttl.yml`](../../.github/workflows/maint-34-quarantine-ttl.yml) · [jump](#wf-quarantine-ttl)
- Verify Service Bot PAT — [`verify-service-bot-pat.yml`](../../.github/workflows/verify-service-bot-pat.yml) · [jump](#wf-verify-service-bot-pat)
- Copilot Readiness — [`copilot-readiness.yml`](../../.github/workflows/copilot-readiness.yml) · [jump](#wf-copilot-readiness)
- Guard: No‑Reuse PR Branches — [`guard-no-reuse-pr-branches.yml`](../../.github/workflows/guard-no-reuse-pr-branches.yml) · [jump](#wf-guard-no-reuse)
- Verify Codex Bootstrap Matrix — [`verify-codex-bootstrap-matrix.yml`](../../.github/workflows/verify-codex-bootstrap-matrix.yml) · [jump](#wf-verify-codex-bootstrap-matrix)
- Check Failure Tracker — [`maint-33-check-failure-tracker.yml`](../../.github/workflows/maint-33-check-failure-tracker.yml) · [jump](#wf-check-failure-tracker)
- Release — [`release.yml`](../../.github/workflows/release.yml) · [jump](#wf-release)

## Workflow Catalog (purpose, triggers, jobs)

This catalog explains what each active workflow does, how it’s triggered, the jobs it runs, and any notable relationships or token usage.

<a id="wf-assign-and-watch"></a>
1) [`agents-41-assign-and-watch.yml`](../../.github/workflows/agents-41-assign-and-watch.yml) — Unified agent orchestration
   - Triggers: `workflow_dispatch`, `schedule`
   - Jobs: `resolve`, `readiness_probe`, `assign_issue`, `clear_assignment`, `watch_pr`, `watchdog_sweep`
     - Resolves incoming events (manual, forwarded label events, scheduled sweeps)
     - Calls `reusable-90-agents.yml` for availability checks before assignment or stale sweeps
     - Assigns automation accounts, boots Codex PRs, and starts cross-reference watchdog monitoring
     - Clears assignments when labels are removed and escalates stale issues when agents are unavailable
  - Links to: `.github/actions/codex-bootstrap-lite`, `codex-issue-bridge.yml` (fallback), wrappers below

<a id="wf-assign-wrapper"></a>
2) [`agents-41-assign.yml`](../../.github/workflows/agents-41-assign.yml) — Compatibility wrapper
   - Triggers: `issues: [labeled, unlabeled]`, `pull_request_target: [labeled]`, `workflow_dispatch`
   - Jobs: `forward`
     - Captures historical triggers and forwards payloads into `agents-41-assign-and-watch.yml`
  - Links to: Unified orchestrator (above)

<a id="wf-watchdog-wrapper"></a>
3) [`agents-42-watchdog.yml`](../../.github/workflows/agents-42-watchdog.yml) — Manual watchdog wrapper
   - Triggers: `workflow_dispatch`
   - Jobs: `forward`
     - Forwards manual watchdog requests into `agents-41-assign-and-watch.yml`
  - Links to: Unified orchestrator (above)

<a id="wf-label-agent-prs"></a>
3) [`label-agent-prs.yml`](../../.github/workflows/label-agent-prs.yml) — Apply agent labels to PRs (keeps downstream automation deterministic)
   - Triggers: `pull_request_target: [opened, synchronize, reopened]`
   - Jobs: `label`
     - Computes desired labels based on actor/head-ref; adds `from:codex|copilot`, `agent:codex|copilot`, `automerge`, `risk:low`
     - Uses `SERVICE_BOT_PAT` if available; otherwise `GITHUB_TOKEN`
  - Links to: `merge-manager.yml`, `autofix.yml` (they predicate on these labels)

<a id="wf-merge-manager"></a>
4) [`merge-manager.yml`](../../.github/workflows/merge-manager.yml) — Unified merge automation
   - Triggers: `pull_request_target: [opened, labeled, unlabeled, synchronize, ready_for_review, reopened]`, `workflow_run: ["CI", "Docker"] (completed)`
   - Jobs: `manage-on-pr-event`, `manage-post-status`
     - Validates allowlist patterns and change size before auto-approving eligible PRs.
     - Enforces label, quiet-period, and active-workflow gates before enabling squash auto-merge.
     - Synchronizes the `ci:green` label with actual check status so pending/failing runs pause automation with a single status comment.
     - Posts (and updates) a single status comment when automation is paused, explaining the blocking condition.

<a id="wf-guard-no-reuse"></a>
5) [`guard-no-reuse-pr-branches.yml`](../../.github/workflows/guard-no-reuse-pr-branches.yml) — Enforce no reuse of merged PR branches
   - Triggers: `pull_request_target: [opened, reopened, synchronize]`
   - Jobs: `guard`
     - Fails the run if the PR head branch previously backed a merged PR (prevents accidental reuse)

<a id="wf-autofix-consumer"></a>
6) [`autofix-consumer.yml`](../../.github/workflows/autofix-consumer.yml) — Thin wrapper around `reuse-autofix`
   - Triggers: `pull_request` (various types) on `phase-2-dev`/`main`
   - Jobs: `autofix`
     - Calls reusable workflow `reuse-autofix.yml` with repo defaults (label, commit prefix)
     - Uses local composite `.github/actions/autofix` which now chains:
       1. Safe Ruff/isort/black/docformatter passes
       2. Targeted unsafe Ruff fixes for F401/F841
       3. Mypy return-type autofix (`scripts/mypy_return_autofix.py`)
       4. Autofixable test expectation refresh (`scripts/update_autofix_expectations.py`)
       The composite exposes `outputs.changed` after all phases.
     - Same‑repo PRs: commits `autofix(ci): …` and pushes directly to the PR branch
     - Fork PRs: generates `autofix.patch` (`git format-patch -1 --stdout`), uploads as `autofix-patch-pr-<num>`, and comments on the PR with apply instructions (`git am < autofix.patch`; push to branch)
     - The job summary reports whether changes were applied and whether this was a same‑repo or fork path. For forks, it includes the artifact name

<a id="wf-autofix-on-failure"></a>
7) [`autofix-on-failure.yml`](../../.github/workflows/autofix-on-failure.yml) — Attempt autofix when CI/Docker fail
   - Triggers: `workflow_run` for `CI`, `Docker`, `Lint`, `Tests` (type: `completed`)
   - Jobs: `handle-failure`
     - Locates corresponding PR, checks out same-repo branches, runs autofix, commits, and pushes

<a id="wf-ci"></a>
8) [`pr-10-ci-python.yml`](../../.github/workflows/pr-10-ci-python.yml) — Test suite on pushes and PRs
   - Triggers: `push` to `phase-2-dev`, `pull_request`
   - Jobs: `core-tests` (matrix on Python 3.11/3.12), `gate` (aggregates)

<a id="wf-docker"></a>
9) [`pr-12-docker-smoke.yml`](../../.github/workflows/pr-12-docker-smoke.yml) — Build, test, and push container image
   - Triggers: `push` to `phase-2-dev`, `pull_request`, `workflow_dispatch`
   - Jobs: `build`
     - Builds image, runs tests in container, health-checks a simple endpoint, pushes to GHCR

<a id="wf-cleanup-codex-bootstrap"></a>
10) [`cleanup-codex-bootstrap.yml`](../../.github/workflows/cleanup-codex-bootstrap.yml) — Prune stale bootstrap branches
   - Triggers: `schedule` (daily), `workflow_dispatch`
   - Jobs: `prune`
     - Deletes old `agents/codex-issue-*` branches beyond TTL

<a id="wf-quarantine-ttl"></a>
11) [`maint-34-quarantine-ttl.yml`](../../.github/workflows/maint-34-quarantine-ttl.yml) — Enforce test quarantine expirations
   - Triggers: `pull_request`, `push` to `phase-2-dev`
   - Jobs: `ttl`
     - Validates `tests/quarantine.yml` entries have not expired

<a id="wf-verify-service-bot-pat"></a>
12) [`verify-service-bot-pat.yml`](../../.github/workflows/verify-service-bot-pat.yml) — Verify `SERVICE_BOT_PAT` presence and scopes
   - Triggers: `workflow_dispatch`
   - Jobs: `check-token`
     - Checks secret exists and minimally has `repo`/`workflow` scopes

<a id="wf-copilot-readiness"></a>
13) [`copilot-readiness.yml`](../../.github/workflows/copilot-readiness.yml) — Copilot readiness probe
   - Triggers: `workflow_dispatch`
   - Jobs: `probe`
     - GraphQL `suggestedActors` check, temp issue assign attempt to `@copilot`, close, verdict

<a id="wf-verify-codex-bootstrap-matrix"></a>
14) [`verify-codex-bootstrap-matrix.yml`](../../.github/workflows/verify-codex-bootstrap-matrix.yml) — End-to-end bootstrap scenario matrix
   - Triggers: `workflow_dispatch`, `schedule`, `push` to specific paths
   - Jobs: `plan`, `matrix` (parallel), `sequential`
     - Runs `scripts/verify_codex_bootstrap.py` across scenarios; uploads artifacts and summaries

<a id="wf-check-failure-tracker"></a>
15) [`maint-33-check-failure-tracker.yml`](../../.github/workflows/maint-33-check-failure-tracker.yml) — Open/close CI failure issues
   - Triggers: `workflow_run` for `CI` and `Docker`
   - Jobs: `failure`, `success`
     - Opens an issue on failures; closes the corresponding issue on subsequent success

<a id="wf-release"></a>
16) [`release.yml`](../../.github/workflows/release.yml) — Build/publish package and GitHub release
   - Triggers: `push` tags `v*`, `workflow_dispatch`
   - Jobs: `build`, `test-install`, `test-pypi` (optional), `release`
     - Builds wheels/sdist, checks, tests install, creates GitHub Release, uploads assets, publishes to PyPI/TestPyPI

Archived / Removed (Issue #1140 hardening, 2025‑09‑18): Legacy agent assignment/bootstrap experiments were deleted to reduce attack surface and maintenance overhead. See `Old/.github/workflows/archive/README.md` for a historical list.

## Diagnostics Workflows — purposes and use cases

Use these when investigating bootstrap, authorization, or automation behaviours:

- `agents-41-assign.yml` (`workflow_dispatch`) — Re-run bootstrap/assignment logic for a specific issue or PR.
  - Purpose: Validate that Codex bootstrap still succeeds end-to-end (branch, PR, command) and that Copilot assignment works.
  - Use when: Codex bootstrap stalls, or you need to replay assignment after changing labels/tokens.

- `agents-42-watchdog.yml` (`workflow_dispatch`) — Inspect Codex PR parity.
  - Purpose: Confirm whether a linked PR appeared (success) or record a timeout window with a comment for follow-up.
  - Use when: Diagnosing missing PR links or verifying the watchdog timeout threshold.

- `copilot-readiness.yml` (Probe Copilot assignability)
  - Purpose: Validate that `@copilot` is assignable via GraphQL and REST.
  - Use when: Copilot agent appears inactive or assignment fails silently.

- `verify-codex-bootstrap-matrix.yml` (Scenario harness)
  - Purpose: Run curated end-to-end scenarios against the bootstrap (success/failure cases), collect artifacts, and produce a summary table.
  - Use when: Regression testing the bootstrap after changes or verifying environment constraints (e.g., missing PAT, disallowed commands).

## Diagnostic Artifacts and Summaries
- Scenario runs upload `codex-verification-report.json`, `codex-verification-report.md`, and `codex-scenario-logs/` as artifacts.
- Most workflows publish a concise run summary via the GitHub summary API for quick triage.

## Known Good Behaviours (Confirmed)

## Diagnostic Guarantees
- Event summary step logs the action, label, and issue number at run start.
- On failure, the run logs indicate whether branch creation, PR creation, labeling, assignment, or commenting failed and with which token.

## Known Good Behaviours (Confirmed)
- PR #1419 merged: the assigner/watchdog pair is present on `phase-2-dev`.
- Label path is dual‑trigger safe and avoids undefined contexts when dispatched.
- Safe checkout prevents missing‑ref errors.

## Expectations for Future Work
- Keep this page up to date if labels, assignees, or token policy changes.
- If org rules change, document the new constraints here (e.g., PAT usage restrictions).

## What changed and why it works now

This section summarizes the differences between the failing Option 2 (manual/"create" mode) runs and the current working implementation.

- Sha-safe file updates: When the bootstrap file already exists (e.g., `agents/codex-<issue>.md`), updates to `/contents` now include the existing `sha` to satisfy the GitHub API and avoid 422 “Invalid request. 'sha' wasn’t supplied.”
- Resilient fallback path: The git-based fallback commits only when there are changes but will still push the new branch even if there are no changes (no-op safe), ensuring the branch exists for PR creation.
- Strict no-reuse policy with guard: Unique, deterministic branch names are used for each run, and a dedicated guard workflow blocks reusing branches that have already backed merged PRs.
- Safer event handling: The assigner resolves the issue number early and gates subsequent steps behind `should_run`, with fail-fast behavior for manual runs that omit required inputs.
- Clearer run summaries: The event summary shows both the event-derived and input-provided issue numbers, helping diagnose mismatches quickly.
- Token priority clarified: `OWNER_PR_PAT` is preferred to author the PR as the human; else `SERVICE_BOT_PAT`, else `GITHUB_TOKEN`.
- PR hygiene defaults: PRs are opened as non-draft and include an initial `@codex start` comment; the PR body replicates the source issue content.

### Autofix pipeline — CI follower consolidation

- Composite action `.github/actions/autofix` no longer commits; it emits `outputs.changed` after running `ruff`, `black`, `isort`, `docformatter`, and light type‑hygiene steps.
- Workflow `.github/workflows/autofix.yml` executes after CI completes and coordinates two lanes:

  - `small-fixes` (CI success + safe diff): commits `chore(autofix): apply small fixes` with `SERVICE_BOT_PAT` for same-repo PRs; forks receive a `small-autofix.patch` artifact.
  - `fix-failing-checks` (CI failure w/ trivial job names): commits `chore(autofix): fix failing checks` when format/lint/type jobs fail; forks receive `failing-checks-autofix.patch`; adds `needs-autofix-review` when no diff produced.
- Benefits: Same composite logic serves both proactive hygiene and reactive failure remediation with a single workflow; forks still get actionable patches without elevated permissions.

## Invite vs Create — when to use which

- Invite mode (default on issue events): Use when you want Codex to assist on PRs authored by a human. The workflow prepares a branch and compare link; you open the PR.
- Create mode (manual dispatch): Use when you want automation to open the PR. Provide `test_issue` and, ideally, `OWNER_PR_PAT` to ensure the PR is authored as you; otherwise the service bot will author the PR.

## Troubleshooting map: symptom → fix

- 422 “Invalid request. 'sha' wasn’t supplied.” when updating `agents/codex-*.md` → The composite now fetches and supplies the existing `sha` on updates.
- “nothing to commit, working tree clean” then branch missing → Fallback now pushes the branch even without a commit (no-op safe push).
- PR created from an old/merged branch → Guard workflow fails the run; create a new branch and retry.

## Tips and Operational Notes
- Issue-label path enforces `invite` mode, guaranteeing human-authored PRs so Codex engages reliably; use `workflow_dispatch` with `pr_mode=create` and `OWNER_PR_PAT` if you need the workflow to open the PR on your behalf.
- Labeling uses `pull_request_target` to ensure labels apply on forked PRs without checking out untrusted code.
- Auto-merge requires all four labels: `from:*`, `agent:*`, `automerge`, `risk:low`, and a non-draft PR.
