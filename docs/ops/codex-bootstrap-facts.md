# Repo Ops Facts — Codex Bootstrap

This page captures the established, validated facts about the Codex bootstrap and workflow wiring in this repository. It is the single source of truth so we don’t keep re‑asking for already‑confirmed details.

Last updated: 2025‑09‑21

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

## Workflows and Actions
- Bridge workflow: `.github/workflows/codex-issue-bridge.yml`.
  - Triggers on Issues: `opened`, `labeled`, `reopened`, and `workflow_dispatch` with optional `test_issue`.
  - Safe checkout of default branch before performing operations.
  - Prefers local composite `.github/actions/codex-bootstrap-lite`; falls back to an inline path if the composite fails.
- Composite action: `.github/actions/codex-bootstrap-lite/action.yml`.
  - PAT‑first, fallback to `GITHUB_TOKEN` if allowed.
  - Creates a new branch with deterministic prefix `agents/codex-issue-<num>-<runid>`.
  - Seeds a bootstrap file under `agents/` and opens a non‑draft PR.
  - Labels/assigns and posts `@codex start` in PR comments.

## Quick Index

- Codex Issue Bridge — [`codex-issue-bridge.yml`](../../.github/workflows/codex-issue-bridge.yml) · [jump](#wf-codex-issue-bridge)
- Label Agent PRs — [`label-agent-prs.yml`](../../.github/workflows/label-agent-prs.yml) · [jump](#wf-label-agent-prs)
- Auto-approve Agent PRs — [`autoapprove.yml`](../../.github/workflows/autoapprove.yml) · [jump](#wf-autoapprove)
- Enable Auto-merge — [`enable-automerge.yml`](../../.github/workflows/enable-automerge.yml) · [jump](#wf-enable-automerge)
- Autofix Consumer — [`autofix-consumer.yml`](../../.github/workflows/autofix-consumer.yml) · [jump](#wf-autofix-consumer)
- Autofix on Failure — [`autofix-on-failure.yml`](../../.github/workflows/autofix-on-failure.yml) · [jump](#wf-autofix-on-failure)
- CI — [`ci.yml`](../../.github/workflows/ci.yml) · [jump](#wf-ci)
- Docker — [`docker.yml`](../../.github/workflows/docker.yml) · [jump](#wf-docker)
- Cleanup Codex Bootstrap — [`cleanup-codex-bootstrap.yml`](../../.github/workflows/cleanup-codex-bootstrap.yml) · [jump](#wf-cleanup-codex-bootstrap)
- Quarantine TTL — [`quarantine-ttl.yml`](../../.github/workflows/quarantine-ttl.yml) · [jump](#wf-quarantine-ttl)
- Verify Service Bot PAT — [`verify-service-bot-pat.yml`](../../.github/workflows/verify-service-bot-pat.yml) · [jump](#wf-verify-service-bot-pat)
- Copilot Readiness — [`copilot-readiness.yml`](../../.github/workflows/copilot-readiness.yml) · [jump](#wf-copilot-readiness)
- Agents Automation — [`reuse-agents.yml`](../../.github/workflows/reuse-agents.yml) · [jump](#wf-reuse-agents)
- Guard: No‑Reuse PR Branches — [`guard-no-reuse-pr-branches.yml`](../../.github/workflows/guard-no-reuse-pr-branches.yml) · [jump](#wf-guard-no-reuse)
- Verify Codex Bootstrap Matrix — [`verify-codex-bootstrap-matrix.yml`](../../.github/workflows/verify-codex-bootstrap-matrix.yml) · [jump](#wf-verify-codex-bootstrap-matrix)
- Check Failure Tracker — [`check-failure-tracker.yml`](../../.github/workflows/check-failure-tracker.yml) · [jump](#wf-check-failure-tracker)
- Release — [`release.yml`](../../.github/workflows/release.yml) · [jump](#wf-release)

## Workflow Catalog (purpose, triggers, jobs)

This catalog explains what each active workflow does, how it’s triggered, the jobs it runs, and any notable relationships or token usage.

<a id="wf-codex-issue-bridge"></a>
1) [`codex-issue-bridge.yml`](../../.github/workflows/codex-issue-bridge.yml) — Codex bootstrap (issue → branch, invite/create PR)
   - Triggers: `issues: [opened, labeled, reopened]`, `workflow_dispatch`
   - Jobs: `bridge`
     - Selects mode: `invite` for issue events (human opens PR), `create` for manual dispatch
     - Safe default-branch checkout, then calls the composite action (PAT-first)
     - Fallback path: creates branch, either invites human with compare link (invite) or opens PR (create)
     - Labels/assigns, optional `@codex start` comment, links PR back to issue
   - Links to: `label-agent-prs.yml` (labels on PR), `reuse-agents.yml` (watchdog mode for issue-to-PR parity), and auto-merge stack (`autoapprove.yml`, `enable-automerge.yml`)

<a id="wf-label-agent-prs"></a>
2) [`label-agent-prs.yml`](../../.github/workflows/label-agent-prs.yml) — Apply agent labels to PRs (keeps downstream automation deterministic)
   - Triggers: `pull_request_target: [opened, synchronize, reopened]`
   - Jobs: `label`
     - Computes desired labels based on actor/head-ref; adds `from:codex|copilot`, `agent:codex|copilot`, `automerge`, `risk:low`
     - Uses `SERVICE_BOT_PAT` if available; otherwise `GITHUB_TOKEN`
   - Links to: `autoapprove.yml`, `enable-automerge.yml`, `autofix*.yml` (they predicate on these labels)

<a id="wf-autoapprove"></a>
3) [`autoapprove.yml`](../../.github/workflows/autoapprove.yml) — Auto-approve eligible agent PRs
   - Triggers: `pull_request_target: [opened, labeled, synchronize, ready_for_review]`
   - Jobs: `approve`
     - Checks labels: `from:*`, `agent:*`, `automerge`, `risk:low`
     - Validates changed file patterns and size, then approves

<a id="wf-enable-automerge"></a>
4) [`enable-automerge.yml`](../../.github/workflows/enable-automerge.yml) — Turn on GitHub auto-merge for eligible PRs
   - Triggers: `pull_request: [opened, labeled, synchronize, ready_for_review]`
   - Jobs: `enable`
     - Enables squash auto-merge when labels and state match

<a id="wf-guard-no-reuse"></a>
5) [`guard-no-reuse-pr-branches.yml`](../../.github/workflows/guard-no-reuse-pr-branches.yml) — Enforce no reuse of merged PR branches
   - Triggers: `pull_request_target: [opened, reopened, synchronize]`
   - Jobs: `guard`
     - Fails the run if the PR head branch previously backed a merged PR (prevents accidental reuse)

<a id="wf-autofix-consumer"></a>
6) [`autofix-consumer.yml`](../../.github/workflows/autofix-consumer.yml) — Trivial autofix on open PRs (fork‑friendly)
   - Triggers: `pull_request` (various types) on `phase-2-dev`/`main`
   - Jobs: `autofix`
     - Calls reusable workflow [`reuse-autofix.yml`](../../.github/workflows/reuse-autofix.yml) which runs the shared composite formatter and exposes `outputs.changed`.
     - Same‑repo PRs: commits `autofix(ci): …` (prefix configurable) and pushes directly to the PR branch.
     - Fork PRs: uploads `autofix.patch` artifact with application instructions; no direct pushes performed.
     - The job summary reports whether changes were applied and whether this was a same‑repo or fork path. For forks, it includes the artifact name.

<a id="wf-autofix-on-failure"></a>
7) [`autofix-on-failure.yml`](../../.github/workflows/autofix-on-failure.yml) — Attempt autofix when CI/Docker fail
   - Triggers: `workflow_run` for `CI`, `Docker`, `Lint`, `Tests` (type: `completed`)
   - Jobs: `handle-failure`
     - Locates corresponding PR, checks out same-repo branches, runs autofix, commits, and pushes

<a id="wf-ci"></a>
8) [`ci.yml`](../../.github/workflows/ci.yml) — Test suite on pushes and PRs
   - Triggers: `push` to `phase-2-dev`, `pull_request`
   - Jobs: `core-tests` (matrix on Python 3.11/3.12), `gate` (aggregates)

<a id="wf-docker"></a>
9) [`docker.yml`](../../.github/workflows/docker.yml) — Build, test, and push container image
   - Triggers: `push` to `phase-2-dev`, `pull_request`, `workflow_dispatch`
   - Jobs: `build`
     - Builds image, runs tests in container, health-checks a simple endpoint, pushes to GHCR

<a id="wf-cleanup-codex-bootstrap"></a>
10) [`cleanup-codex-bootstrap.yml`](../../.github/workflows/cleanup-codex-bootstrap.yml) — Prune stale bootstrap branches
   - Triggers: `schedule` (daily), `workflow_dispatch`
   - Jobs: `prune`
     - Deletes old `agents/codex-issue-*` branches beyond TTL

<a id="wf-quarantine-ttl"></a>
11) [`quarantine-ttl.yml`](../../.github/workflows/quarantine-ttl.yml) — Enforce test quarantine expirations
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

<a id="wf-reuse-agents"></a>
14) [`reuse-agents.yml`](../../.github/workflows/reuse-agents.yml) — Parameterised agents automation (bootstrap, readiness, diagnostics, watchdog)
   - Triggers: `workflow_call`
   - Jobs: `agents`
     - Controlled via boolean inputs:
       1. `enable_readiness` — multi-agent assignability probe (`readiness_agents` input selects accounts).
       2. `enable_preflight` — single-issue Codex preflight with optional command post.
       3. `enable_diagnostic` — environment/token diagnostic with optional branch attempt (`diagnostic_attempt_branch`).
       4. `enable_verify_issue` — verifies a specific issue has an agent assignee (`verify_issue_number`).
       5. Codex bootstrap — opens PRs for issues with `bootstrap_issues_label` (default `agent:codex`).
       6. `enable_watchdog` — lightweight issue/PR parity check and telemetry summary.
     - Inputs gate execution; unset modes skip entirely. Consumer workflows (schedule, dispatch, PR) pass flags as needed.

<a id="wf-verify-codex-bootstrap-matrix"></a>
15) [`verify-codex-bootstrap-matrix.yml`](../../.github/workflows/verify-codex-bootstrap-matrix.yml) — End-to-end bootstrap scenario matrix
   - Triggers: `workflow_dispatch`, `schedule`, `push` to specific paths
   - Jobs: `plan`, `matrix` (parallel), `sequential`
     - Runs `scripts/verify_codex_bootstrap.py` across scenarios; uploads artifacts and summaries

<a id="wf-check-failure-tracker"></a>
16) [`check-failure-tracker.yml`](../../.github/workflows/check-failure-tracker.yml) — Open/close CI failure issues
   - Triggers: `workflow_run` for `CI` and `Docker`
   - Jobs: `failure`, `success`
     - Opens an issue on failures; closes the corresponding issue on subsequent success

<a id="wf-release"></a>
17) [`release.yml`](../../.github/workflows/release.yml) — Build/publish package and GitHub release
   - Triggers: `push` tags `v*`, `workflow_dispatch`
   - Jobs: `build`, `test-install`, `test-pypi` (optional), `release`
     - Builds wheels/sdist, checks, tests install, creates GitHub Release, uploads assets, publishes to PyPI/TestPyPI

Archived / Removed (Issue #1140 hardening, 2025‑09‑18): Legacy agent assignment/bootstrap experiments were deleted to reduce attack surface and maintenance overhead. See `Old/.github/workflows/archive/README.md` for a historical list.

## Diagnostics Workflows — purposes and use cases

Use these when investigating bootstrap, authorization, or automation behaviours:

- `reuse-agents.yml` (`enable_preflight: true`) — Codex preflight
  - Purpose: Confirm the Codex connector account can be assigned here and optionally that its command phrase posts successfully.
  - Use when: Codex doesn’t react to issues/PRs; verify app installation/permissions.

- `copilot-readiness.yml` (Probe Copilot assignability)
  - Purpose: Validate that `@copilot` is assignable via GraphQL and REST.
  - Use when: Copilot agent appears inactive or assignment fails silently.

- `reuse-agents.yml` (`enable_readiness: true`) — Multi-agent matrix
  - Purpose: Test a list of candidate logins for assignability; produces a JSON-like report in the job logs/summary.
  - Use when: Standing up new repos or debugging org policy changes that affect multiple agents.

- `reuse-agents.yml` (`enable_diagnostic: true`) — Environment & token probe
  - Purpose: Inspect tokens present, base branch/sha, and optionally attempt branch creation with each token type.
  - Use when: Branch creation or PAT fallback behaves unexpectedly.

- `reuse-agents.yml` (`enable_watchdog: true`) — Issue/PR parity telemetry
  - Purpose: Detect issues labeled for Codex where bootstrap PR not yet created or summarize state telemetry.
  - Use when: Monitoring automation health or investigating missing bootstrap PRs.

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
- PR #1025 merged: the fixed bridge workflow is present on `phase-2-dev`.
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
- Safer event handling: The bridge resolves the issue number early and gates subsequent steps behind `has_issue`, with fail-fast behavior for manual runs that omit `test_issue`.
- Clearer run summaries: The event summary shows both the event-derived and input-provided issue numbers, helping diagnose mismatches quickly.
- Token priority clarified: `OWNER_PR_PAT` is preferred to author the PR as the human; else `SERVICE_BOT_PAT`, else `GITHUB_TOKEN`.
- PR hygiene defaults: PRs are opened as non-draft and include an initial `@codex start` comment; the PR body replicates the source issue content.

### Autofix pipeline — fork‑friendly refactor

- Composite action `.github/actions/autofix` no longer commits; it emits `outputs.changed` after running `ruff`, `black`, `isort`, `docformatter`, and light type‑hygiene steps.
- Workflow `.github/workflows/autofix-consumer.yml` branches by PR provenance while delegating to `reuse-autofix.yml`:
  - Same‑repo: commit and push to the PR head branch.
  - Fork: create `autofix.patch`, upload as artifact `autofix-patch-pr-<num>`, and post a PR comment with local apply instructions.
- Benefits: Same‑repo remains hands‑off; forks still get actionable fixes without elevated permissions.

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
