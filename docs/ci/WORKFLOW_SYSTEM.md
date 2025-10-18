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

> 📌 **Where this document fits.** The `README.md` “CI automation orientation”
> call-out and the opening section of `CONTRIBUTING.md` both point here as the
> canonical map of what runs where. Keep this guide side by side with
> [AGENTS_POLICY.md](./AGENTS_POLICY.md) whenever you are evaluating workflow
> edits—the policy spells out the guardrails, while this page traces the
> topology those guardrails protect.

> 🧾 **One-minute orientation.**
> - Glance at [Topology at a glance](#topology-at-a-glance) to map the four
>   automation buckets to their YAML entry points and understand why each
>   surface exists.
> - Use the [Bucket quick reference](#bucket-quick-reference) or
>   [Workflow summary table](#workflow-summary-table) when you need the
>   trigger/purpose/required matrix for a review or incident response.
> - Keep [How to change a workflow safely](#how-to-change-a-workflow-safely)
>   open next to [AGENTS_POLICY.md](./AGENTS_POLICY.md) before editing any
>   workflow so you never bypass the guardrails.

> 🧭 **Use this map to stay oriented.**
> - Start with the [quick orientation](#quick-orientation-for-new-contributors)
>   checklist when you are new or returning so you know which buckets will
>   react to your work.
> - Reference the [workflow summary table](#workflow-summary-table) for
>   triggers, required signals, and log links before you brief a reviewer or
>   rerun a check manually.
> - Follow [How to change a workflow safely](#how-to-change-a-workflow-safely)
>   alongside [AGENTS_POLICY.md](./AGENTS_POLICY.md) whenever you update
>   `.github/workflows/` so you never bypass the guardrails.

### Contents

- [Quick orientation for new contributors](#quick-orientation-for-new-contributors)
- [Onboarding checklist (save for future you)](#onboarding-checklist-save-for-future-you)
- [Scenario cheat sheet](#scenario-cheat-sheet)
- [Bucket quick reference](#bucket-quick-reference)
- [Bucket guardrails at a glance](#bucket-guardrails-at-a-glance)
- [Observability surfaces by bucket](#observability-surfaces-by-bucket)
- [Topology at a glance](#topology-at-a-glance)
- [Buckets and canonical workflows](#buckets-and-canonical-workflows)
- [Lifecycle example: from pull request to follow-up automation](#lifecycle-example-from-pull-request-to-follow-up-automation)
- [Workflow summary table](#workflow-summary-table)
- [Policy](#policy)
- [Final topology (keep vs retire)](#final-topology-keep-vs-retire)
- [How to change a workflow safely](#how-to-change-a-workflow-safely)
- [Verification checklist](#verification-checklist)
- [Branch protection playbook](#branch-protection-playbook)

### How the buckets interact in practice

- **Gate and PR 02 Autofix** are the first responders on every pull request.
  Gate decides whether to fan out into the reusable CI topology, while Autofix
  runs the optional clean-up sweep when the label is applied.
- **Maint 46 Post CI** wakes up after a successful Gate run and aggregates the
  results, while the remaining maintenance workflows keep the default branch
  protected on a schedule or by manual dispatch.
- **Agents automation** consumes labelled issues and protected workflow edits,
  using the orchestrator to coordinate downstream jobs and guards such as the
  Agents Critical Guard.
- **Reusable lint/test/topology workflows** execute only when called; they
  provide the shared matrix for Gate, Maint 46, and manual reruns so contributors
  see the same results regardless of entry point.

### Quick orientation for new contributors

When you first land on the project:

1. **Skim the [Topology at a glance](#topology-at-a-glance) table and the bucket
   summaries below** to understand which workflows will react to your pull
   request or scheduled automation.
2. **Use the workflow summary table** as the canonical source for triggers,
   required status, and log links when you need to confirm behaviour or share a
   run with reviewers. Pair it with the
   [observability surfaces](#observability-surfaces-by-bucket) section to grab
   the exact permalink or artifact bundle you need for status updates. If you
   need to know which rules you must follow before editing a YAML file, jump
   straight to [Bucket guardrails at a glance](#bucket-guardrails-at-a-glance)
   for the enforcement summary, then finish the deep dive in
   [How to change a workflow safely](#how-to-change-a-workflow-safely).
3. **Review [How to change a workflow safely](#how-to-change-a-workflow-safely)**
   before editing any YAML. It enumerates the guardrails, labels, and approval
   steps you must follow.
4. **Cross-reference the [Workflow Catalog](WORKFLOWS.md)** for deeper YAML
   specifics (inputs, permissions, job layout) once you know which surface you
   are touching.

### Onboarding checklist (save for future you)

Run through this sequence when you are new to the automation surface or when you
return after a break:

1. **Bookmark the [Agents Workflow Protection Policy](./AGENTS_POLICY.md)** so
   you can confirm label and review requirements before touching protected
   workflows. The checklist below assumes you have read that policy once.
2. **Open the latest [Gate run](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml)** and skim the
   Summary tab. It shows which reusable jobs fire for typical PRs and highlights
   the docs-only path so you know what to expect for lightweight changes.
3. **Review the most recent [Maint 46 Post CI run](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-46-post-ci.yml)**
   to see how post-merge hygiene is reported. Treat that summary comment as the
   canonical “state of CI” dashboard after every merge.
4. **Practice finding the agents guardrails** by visiting the
   [Agents Critical Guard history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-critical-guard.yml)
   and reading a recent run summary. It confirms how the label and review gates
   manifest in CI when protected files change.
5. **Walk through a dry-run change**: open this document, the Workflow Catalog,
   and the policy side by side. Trace how you would update a workflow safely and
   which checks would block an unsafe edit. Doing this once keeps the guardrails
   fresh when you work on the real issue queue.

### Scenario cheat sheet

The table below is the canonical source of truth, but these quick scenarios
highlight the most common entry points:

- **Opening or updating a feature PR?** Expect the [PR checks bucket](#pr-checks-gate--autofix)
  (Gate + optional Autofix) to run automatically and to fan out into the
  reusable CI topology.
- **Gate is red on your PR?** Expand the Gate summary comment to spot the
  failing lane, then open the linked workflow run. The reusable jobs expose a
  dedicated "Reusable CI" job section; download the attached artifact when
  Gate mentions one so you can compare the logs locally before re-running the
  check.
- **Investigating a nightly or weekend regression?** Start with the
  [Maintenance & repo health](#maintenance--repo-health) workflows—they collect
  the scheduled hygiene runs and post-merge follow-ups.
- **Maint 46 Post CI flagged drift?** Follow the summary comment back to the
  workflow run, review the uploaded artifact bundle, and check the linked
  follow-up issue before you retry. Maint 46 only exits green when both the
  reusable CI fan-out and the hygiene sweep succeed.
- **Working on labelled agent issues or Codex escalations?** Review the
  [Issue / agents automation](#issue--agents-automation) guardrails so you know
  which workflows dispatch work and which checks must stay green.
- **Editing YAML under `.github/workflows/`?** Read [How to change a workflow
  safely](#how-to-change-a-workflow-safely) before committing; it lists the
  approvals, labels, and verification steps Gate will enforce.
- **Need to run lint, type checking, or container tests by hand?** Use the
  [Error checking, linting, and testing topology](#error-checking-linting-and-testing-topology)
  section to find the reusable entry points and confirm which callers already
  exercise the matrix.

### Lifecycle example: from pull request to follow-up automation

This happy-path walk-through shows how the four buckets hand work to one another
and where to watch the result:

1. **Developer opens or updates a pull request.** Gate (`pr-00-gate.yml`) runs
   immediately, detects whether the diff is docs-only, and—when code changed—
   calls the reusable lint/test topology. You can watch progress in the
   [Gate workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml)
   and follow the linked reusable job logs from the Checks tab.
2. **Autofix (optional).** If reviewers add the `autofix` label, the PR 02
   Autofix workflow runs fixers via the reusable autofix entry point. Its logs
   show up under the same pull request for easy comparison with Gate.
3. **Merge lands on the default branch.** Maint 46 Post CI triggers from the
   Gate success signal, aggregates artifacts, and applies any low-risk cleanup.
   Scheduled maintenance jobs (Maint 45 and Health 40–45) continue to run on
   their cadence even when no one is watching, keeping the repo healthy.
4. **Issue and agents automation picks up queued work.** Labelled issues flow
   through the Agents 63 bridges into the Agents 70 orchestrator, which may in
   turn call the reusable agents topology or kick additional verification jobs
   such as the Agents Critical Guard.
5. **Manual investigations reuse the topology.** When contributors need to
   rerun linting, typing, or container checks locally, they can dispatch the
   `selftest-runner.yml` workflow or call the reusable CI entries directly,
   guaranteeing they exercise the same matrix Gate and Maint 46 rely on.

Revisit this sequence whenever you need to explain the automation lifecycle to
new contributors or track down where a particular check originated.

### Bucket quick reference

Use this cheat sheet when you need the quickest possible answer about “what
fires where” without diving into the full tables:

- **PR checks (Gate + PR 02 Autofix)**
  - **Primary workflows.** `pr-00-gate.yml`, `pr-02-autofix.yml` under
    `.github/workflows/`.
  - **Triggers.** `pull_request`, with Gate also running in
    `pull_request_target` for fork visibility. Autofix is label-gated.
  - **Purpose.** Guard every PR, detect docs-only diffs, and offer an optional
    autofix sweep before reviewers spend time on hygiene nits.
  - **Where to inspect logs.** Gate: [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml).
    Autofix: [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-02-autofix.yml).
- **Maintenance & repo health**
  - **Primary workflows.** `maint-46-post-ci.yml`, `maint-45-cosmetic-repair.yml`,
    and the health guardrails (`health-40` through `health-45`).
  - **Triggers.** Combination of `workflow_run` (Maint 46 watching Gate),
    recurring schedules, and manual dispatch for Maint 45.
  - **Purpose.** Keep the default branch stable after merges, surface drift, and
    enforce branch-protection expectations without waiting for the next PR.
  - **Where to inspect logs.** Maint 46: [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-46-post-ci.yml).
    Maint 45: [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-45-cosmetic-repair.yml).
    Health guardrails: the [Health 40–45 dashboards](https://github.com/stranske/Trend_Model_Project/actions?query=workflow%3AHealth+40+repo+OR+workflow%3AHealth+41+repo+OR+workflow%3AHealth+42+Actionlint+OR+workflow%3AHealth+43+CI+Signature+Guard+OR+workflow%3AHealth+44+Gate+Branch+Protection+OR+workflow%3AHealth+45+Agents+Guard).
- **Issue / agents automation**
  - **Primary workflows.** `agents-70-orchestrator.yml`, the paired
    `agents-63-*.yml` issue bridges, `agents-64-verify-agent-assignment.yml`, and
    `agents-critical-guard.yml`.
  - **Triggers.** A mix of orchestrator `workflow_call` hand-offs, labelled
    issues, schedules, and guarded pull requests when protected YAML changes.
  - **Purpose.** Convert tracked issues into automation tasks while preserving
    the immutable agents surface behind Code Owners, labels, and guard checks.
  - **Where to inspect logs.** Orchestrator:
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-70-orchestrator.yml).
    Agents 63 bridge:
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-63-codex-issue-bridge.yml).
    Agents Critical Guard:
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-critical-guard.yml).
- **Error checking, linting, and testing topology**
  - **Primary workflows.** `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`,
    `reusable-16-agents.yml`, `reusable-18-autofix.yml`, and `selftest-runner.yml`.
  - **Triggers.** Invoked via `workflow_call` by Gate, Maint 46, and manual
    reruns. `selftest-runner.yml` handles the nightly rehearsal (cron at 06:30 UTC)
    and manual publication modes via `workflow_dispatch`.
  - **Purpose.** Provide a consistent lint/type/test/container matrix so every
    caller sees identical results.
  - **Where to inspect logs.** Reusable Python CI:
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-10-ci-python.yml).
    Docker CI:
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-12-ci-docker.yml).
    Self-test runner:
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/selftest-runner.yml).

### Bucket guardrails at a glance

Use this table when you need a snapshot of the non-negotiable rules that govern
each automation surface. Every line links back to the policy or workflow that
enforces the guardrail so you know where to confirm compliance:

| Bucket | Guardrails you must respect | Where it is enforced |
| --- | --- | --- |
| PR checks | Gate is required on every PR; docs-only detection happens inside Gate; Autofix is label-gated and cancels duplicates so it never races Maint 46. | Gate workflow protection + [branch protection](#branch-protection-playbook) keep the check mandatory. |
| Maintenance & repo health | Maint 46 only runs after Gate succeeds; Health 40–45 must stay enabled so the default branch keeps its heartbeat; Maint 45 is manual and should only be dispatched by maintainers. | Maint 46 summary comment, Health dashboard history, and Maint 45 run permissions. |
| Issue / agents automation | `agents:allow-change` label, Code Owner review, and Agents Critical Guard are mandatory before protected YAML merges; orchestrator dispatch only accepts labelled issues. | [Agents Workflow Protection Policy](./AGENTS_POLICY.md), Agents Critical Guard, and repository label configuration. |
| Error checking, linting, and testing topology | Reusable workflows run with signed references; callers must not fork or bypass them; self-test runner is manual and should mirror Gate’s matrix. | Health 42 Actionlint, Health 43 signature guard, and the reusable workflow permissions matrix. |

### Observability surfaces by bucket

Think of these run histories, dashboards, and artifacts as the canonical places
to verify that automation worked—or to capture a permalink for post-mortems and
status updates:

- **PR checks**
  - *Gate summary comment.* Appears automatically on every pull request and is
    the first line of evidence when a contributor wants to share status.
  - *Gate workflow run.* The Checks tab links to
    [pr-00-gate.yml history](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml),
    which exposes reusable job logs and uploaded artifacts for failing runs.
  - *Autofix artifacts.* When the `autofix` label is applied, the workflow
    uploads the formatted patch or commit diff for reviewers to inspect before
    merging.
- **Maintenance & repo health**
  - *Maint 46 comment and artifact bundle.* Each run posts a consolidated
    summary with links to artifacts, making it easy to confirm that post-merge
    hygiene completed.
  - *Health 40–45 dashboards.* The Actions list filtered by `workflow:Health`
    serves as the heartbeat for scheduled enforcement jobs. Failures here are a
    red flag that branch protection or guardrails drifted.
- **Issue / agents automation**
  - *Agents 70 orchestrator timeline.* The orchestrator’s
    [workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-70-orchestrator.yml)
    reveals downstream dispatch history and the inputs supplied by labelled
    issues.
  - *Agents Critical Guard status.* Inspect
    [agents-critical-guard.yml](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-critical-guard.yml)
    whenever a protected YAML edit lands; it should be green before merge.
  - *Agents 63 bridge logs.* These runs attach trace logs showing which issues
    were synced or bootstrapped, invaluable when debugging missed escalations.
- **Error checking, linting, and testing topology**
  - *Reusable job logs.* Because the reusable workflows emit job-level logs for
    each caller, you can open the workflow run from Gate or Maint 46 and expand
    the “Reusable CI” job to see the full lint/test output.
    - *Self-test runner nightly summary.* The scheduled run appends the
      verification table to the job summary so regressions surface without
      paging maintainers.
    - *Self-test runner history artifact.* Manual dispatch uploads the combined
      test report so local reproductions can be compared against CI output.

## Topology at a glance

| Bucket | Where it runs | YAML entry points | Why it exists |
| --- | --- | --- | --- |
| PR checks | Every pull request event (including `pull_request_target` for fork visibility) | `pr-00-gate.yml`, `pr-02-autofix.yml` | Keep the default branch green by running the gating matrix, autofix sweep, and docs-only short circuit before reviewers waste time. |
| Maintenance & repo health | Daily/weekly schedules plus manual dispatch | `maint-46-post-ci.yml`, `maint-45-cosmetic-repair.yml`, `health-4x-*.yml` | Scrub lingering CI debt, enforce branch protection, and surface drift before it breaks contributor workflows. |
| Issue / agents automation | Orchestrator dispatch (`workflow_dispatch`, `workflow_call`, `issues`) | `agents-70-orchestrator.yml`, `agents-63-*.yml`, `agents-64-verify-agent-assignment.yml`, `agents-critical-guard.yml` | Translate labelled issues into automated work while keeping the protected agents surface locked behind guardrails. |
| Error checking, linting, and testing topology | Reusable fan-out invoked by Gate, Maint 46, and manual triggers | `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-16-agents.yml`, `reusable-18-autofix.yml`, `selftest-runner.yml` | Provide a single source of truth for lint/type/test/container jobs so every caller runs the same matrix with consistent tooling. |

Keep this table handy when you are triaging automation: it confirms which workflows wake up on which events, the YAML files to inspect, and the safety purpose each bucket serves.

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
- **Self-test Runner** – `selftest-runner.yml` is the consolidated entry
  point. It runs nightly via cron (06:30 UTC) to rehearse the reusable matrix
  and accepts manual dispatches for summary/comment publication. Inputs:
  - `mode`: `summary`, `comment`, or `dual-runtime` (controls reporting surface
    and Python matrix).
  - `post_to`: `pr-number` or `none` (comment target when `mode == comment`).
  - `enable_history`: `true` or `false` (download the verification artifact for
    local inspection).
  - Optional niceties include `pull_request_number`,
    `summary_title`/`comment_title`, `reason`, and `python_versions` (JSON array
    to override the default matrix).
  - Maint 46 Post CI now serves as the canonical Gate follow-up comment. The
    legacy wrappers `maint-43-selftest-pr-comment.yml`,
    `pr-20-selftest-pr-comment.yml`, and `selftest-pr-comment.yml` were retired
    in Issue #2720 so PR annotations flow through either Maint 46 or this
    manual runner.

## Workflow summary table

**Legend.** `✅` means the workflow must succeed before the associated change can merge; `⚪` covers opt-in, scheduled, or manual automation that supplements the required guardrails.

| Workflow | Trigger | Purpose | Required? | Artifacts / logs |
| --- | --- | --- | --- | --- |
| **Gate** (`pr-00-gate.yml`, PR checks bucket) | `pull_request`, `pull_request_target` | Detect docs-only diffs, orchestrate CI fan-out, and publish the combined status. | ✅ Always | [Gate workflow history](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml) |
| **PR 02 Autofix** (`pr-02-autofix.yml`, PR checks bucket) | `pull_request` (label gated) | Run optional fixers when the `autofix` label is present. | ⚪ Optional | [Autofix runs & artifacts](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-02-autofix.yml) |
| **Maint 46 Post CI** (`maint-46-post-ci.yml`, maintenance bucket) | `workflow_run` (Gate success) | Consolidate CI output, apply small hygiene fixes, and update failure-tracker state. | ⚪ Optional (auto) | [Maint 46 run log](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-46-post-ci.yml) |
| **Maint 45 Cosmetic Repair** (`maint-45-cosmetic-repair.yml`, maintenance bucket) | `workflow_dispatch` | Run pytest + fixers manually and open a labelled PR when changes are required. | ⚪ Manual | [Maint 45 manual entry](https://github.com/stranske/Trend_Model_Project/actions/workflows/maint-45-cosmetic-repair.yml) |
| **Health 40 Repo Selfcheck** (`health-40-repo-selfcheck.yml`, maintenance bucket) | `schedule` (daily) | Capture repository pulse metrics. | ⚪ Scheduled | [Health 40 summary](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-40-repo-selfcheck.yml) |
| **Health 41 Repo Health** (`health-41-repo-health.yml`, maintenance bucket) | `schedule` (weekly) | Perform weekly dependency and repo hygiene sweep. | ⚪ Scheduled | [Health 41 dashboard](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-41-repo-health.yml) |
| **Health 42 Actionlint** (`health-42-actionlint.yml`, maintenance bucket) | `schedule` (daily) | Enforce actionlint across workflows. | ⚪ Scheduled | [Health 42 logs](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-42-actionlint.yml) |
| **Health 43 CI Signature Guard** (`health-43-ci-signature-guard.yml`, maintenance bucket) | `schedule` (daily) | Verify reusable workflow signature pins. | ⚪ Scheduled | [Health 43 verification](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-43-ci-signature-guard.yml) |
| **Health 44 Gate Branch Protection** (`health-44-gate-branch-protection.yml`, maintenance bucket) | `schedule`, `workflow_dispatch` | Ensure Gate and Agents Guard stay required on the default branch. | ⚪ Scheduled (fails if misconfigured) | [Health 44 enforcement logs](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-44-gate-branch-protection.yml) |
| **Health 45 Agents Guard** (`health-45-agents-guard.yml`, maintenance bucket) | `pull_request`, `workflow_dispatch`, `schedule` | Block unauthorized changes to protected agents workflows. | ✅ Required when `agents-*.yml` changes | [Agents Guard run history](https://github.com/stranske/Trend_Model_Project/actions/workflows/health-45-agents-guard.yml) |
| **Agents 70 Orchestrator** (`agents-70-orchestrator.yml`, agents bucket) | `workflow_call`, `workflow_dispatch` | Fan out consumer automation and dispatch work. | ✅ Critical surface | [Orchestrator runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-70-orchestrator.yml) |
| **Agents 63 Codex Issue Bridge** (`agents-63-codex-issue-bridge.yml`, agents bucket) | `issues`, `workflow_dispatch` | Bootstrap branches and PRs from labelled issues. | ✅ Critical surface | [Agents 63 bridge logs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-63-codex-issue-bridge.yml) |
| **Agents 63 ChatGPT Issue Sync** (`agents-63-chatgpt-issue-sync.yml`, agents bucket) | `workflow_dispatch` | Keep curated topic files in sync with issues. | ✅ Critical surface | [Agents 63 sync runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-63-chatgpt-issue-sync.yml) |
| **Agents 64 Verify Agent Assignment** (`agents-64-verify-agent-assignment.yml`, agents bucket) | `schedule`, `workflow_dispatch` | Audit orchestrated assignments and alert on drift. | ⚪ Scheduled | [Agents 64 audit history](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-64-verify-agent-assignment.yml) |
| **Agents Critical Guard** (`agents-critical-guard.yml`, agents bucket) | `pull_request` | Block deletion or renaming of protected agents workflows before maintainer review. | ✅ Required when PR touches protected files | [Agents Critical Guard runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/agents-critical-guard.yml) |
| **Reusable Python CI** (`reusable-10-ci-python.yml`, error-checking bucket) | `workflow_call` | Provide shared lint/type/test matrix for Gate and manual callers. | ✅ When invoked | [Reusable Python CI runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-10-ci-python.yml) |
| **Reusable Docker CI** (`reusable-12-ci-docker.yml`, error-checking bucket) | `workflow_call` | Build and smoke-test container images. | ✅ When invoked | [Reusable Docker runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-12-ci-docker.yml) |
| **Reusable Agents** (`reusable-16-agents.yml`, error-checking bucket) | `workflow_call` | Power orchestrated dispatch. | ✅ When invoked | [Reusable Agents history](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-16-agents.yml) |
| **Reusable Autofix** (`reusable-18-autofix.yml`, error-checking bucket) | `workflow_call` | Centralise formatter + fixer execution. | ✅ When invoked | [Reusable Autofix runs](https://github.com/stranske/Trend_Model_Project/actions/workflows/reusable-18-autofix.yml) |
| **Self-test Runner** (`selftest-runner.yml`, error-checking bucket) | `schedule` (06:30 UTC), `workflow_dispatch` | Rehearse the reusable CI scenarios nightly and publish manual summaries or PR comments on demand. | ⚪ Scheduled/manual | [Self-test runner history](https://github.com/stranske/Trend_Model_Project/actions/workflows/selftest-runner.yml) |

## Policy

- **Required checks.** Gate is mandatory on every PR. Health 45 Agents Guard
  becomes required whenever a change touches the `agents-*.yml` surface. Both
  checks must appear in branch protection.
- **Docs-only detection.** Lives exclusively inside Gate—there is no separate
  docs-only workflow.
- **Autofix.** Maint 46 centralizes automated follow-up fixes. Forks upload
  patch artifacts instead of pushing. Pre-CI autofix (`pr-02-autofix.yml`) must
  stay label-gated and cancel duplicates while Gate runs.
- **Branch protection.** The default branch must require the Gate status context
  (`gate`). Health 44 resolves the current default branch via the REST API and
  either enforces or verifies the rule (requires a `BRANCH_PROTECTION_TOKEN`
  secret with admin scope for enforcement). When agent workflows are in play,
  the rule also enforces **Health 45 Agents Guard** so protected files stay
  gated. Maint 46 Post CI wakes up only after Gate succeeds; it publishes the
  consolidated summary comment but remains informational rather than a required
  status check.
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
6. Reflect the new state in this document and the [Workflow Catalog](WORKFLOWS.md)
   so future contributors inherit an accurate topology and guardrail map. Update
   cross-links in `README.md` / `CONTRIBUTING.md` if the landing surfaces move.

## Verification checklist

- Gate badge in `README.md` and branch protection both show as required for the default branch.
- Health 45 Agents Guard appears as a required check whenever protected workflows change and reports ✅ in the latest run.
- Maintainers can point to the most recent [Workflow System Overview](../ci/WORKFLOW_SYSTEM.md) update in pull-request history, demonstrating that contributors can discover the guardrails without escalation.
- Gate runs and passes on docs-only PRs and appears as a required check.
- Health 45 blocks unauthorized agents workflow edits and reports as the required check whenever `agents-*.yml` files change.
- Health 44 confirms branch protection requires Gate and Agents Guard on the default branch.
- Maint 46 posts a single consolidated summary; autofix artifacts or commits are attached where allowed.

### Required vs informational checks on `phase-2-dev`

- **Required before merge.** Gate / `gate` must finish green on every pull request into `phase-2-dev`. Branch protection enforces this context.
- **Informational after merge.** Maint 46 Post CI fans out once Gate finishes and posts the aggregated summary comment. It mirrors the reusable CI results but does not block merges because it runs post-merge.

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
3. **Configure branch protection manually when adjusting via the UI.**
   - Navigate to **Settings → Branches → Add branch protection rule** and target
     the default branch (`phase-2-dev`).
   - Enable **Require status checks to pass before merging**, then select
     **Gate / gate**. Keep **Health 45 Agents Guard / Enforce agents workflow
     protections** checked so agent-surface edits stay gated.
   - Enable **Require branches to be up to date before merging** to match the
     automation policy.
4. **Run the enforcement script locally when needed.**
   - `python tools/enforce_gate_branch_protection.py --repo <owner>/<repo> --branch <default-branch> --check`
     reports the current status.
   - Add `--require-strict` to fail if the workflow token cannot confirm
     “Require branches to be up to date” (needs admin scope).
   - Add `--apply` to enforce the rule locally (requires admin token in
     `GITHUB_TOKEN`/`GH_TOKEN`). Use `--snapshot path.json` to capture
     before/after state for change control.
5. **Audit the result.**
   - Health 44 uploads JSON snapshots (`enforcement.json`, `verification.json`)
     mirroring the script output and writes a step summary when it runs in
     observer mode.
   - In GitHub settings, confirm that **Gate / gate** appears under required
     status checks, with **Health 45 Agents Guard** retained for agent-surface
     enforcement. Maint 46 Post CI is intentionally absent—it publishes the
     summary comment after merge and remains informational.
6. **Trigger Health 44 on demand.**
   - Kick a manual run with `gh workflow run "Health 44 Gate Branch Protection" --ref <default-branch>`
     whenever you change branch-protection settings.
   - Scheduled executions run daily at 06:00 UTC; a manual dispatch confirms the
     fix immediately after you apply it.
7. **Verify with a test PR.**
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
